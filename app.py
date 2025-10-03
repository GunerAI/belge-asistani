#!/usr/bin/env python3
import io
import os
import json
import shutil
import subprocess
import tempfile
from typing import Tuple, Optional

import streamlit as st
import re
from collections import Counter

import pdfplumber
from pdf2image import convert_from_bytes  # preview (requires Poppler)
from PIL import Image

# --- LangChain / OpenAI for Q&A ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Setup & configuration
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY. Please add it to your .env file.")

MAX_MB = 10
MAX_BYTES = MAX_MB * 1024 * 1024

st.set_page_config(page_title="PDF Uploader & Reader", page_icon="📄", layout="wide")
st.title("📄 PDF Uploader & Reader (Streamlit)")
st.caption(
    f"Upload a PDF (max {MAX_MB} MB). We’ll validate, preview page 1, "
    "and extract text. Then ask questions about it."
)

# Ensure chat history & export flag exist BEFORE rendering sidebar
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "sources_by_turn" not in st.session_state:
    st.session_state["sources_by_turn"] = []  # list[list[dict]] per assistant turn
if "exports_enabled" not in st.session_state:
    st.session_state["exports_enabled"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Error helpers
# ─────────────────────────────────────────────────────────────────────────────
class UserAbort(Exception):
    pass

def add_error(errors: list, level: str, msg: str, detail: str | None = None):
    errors.append({"level": level, "msg": msg, "detail": detail})

def fail(errors: list, level: str, msg: str, detail: str | None = None):
    add_error(errors, level, msg, detail)
    raise UserAbort()

# ─────────────────────────────────────────────────────────────────────────────
# Text cleanup helpers
# ─────────────────────────────────────────────────────────────────────────────
def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def fix_ligatures(text: str) -> str:
    return (text
            .replace("ﬁ", "fi")
            .replace("ﬂ", "fl")
            .replace("ﬀ", "ff")
            .replace("ﬃ", "ffi")
            .replace("ﬄ", "ffl"))

def fix_run_together_words(text: str) -> str:
    """Heuristics to separate words when spaces go missing."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # wordWord → word Word
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)  # A1 → A 1
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)  # 1A → 1 A
    return text

def space_ratio(s: str) -> float:
    """Fraction of spaces among visible chars (ignores newlines)."""
    if not s:
        return 0.0
    visible = re.sub(r"[\r\n]", "", s)
    if not visible:
        return 0.0
    return visible.count(" ") / len(visible)

def remove_repeated_headers_footers(per_page_texts: list[str]) -> list[str]:
    head_candidates, foot_candidates = Counter(), Counter()
    for t in per_page_texts:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if not lines:
            continue
        head_candidates[lines[0]] += 1
        if len(lines) > 1: head_candidates[lines[1]] += 1
        foot_candidates[lines[-1]] += 1
        if len(lines) > 1: foot_candidates[lines[-2]] += 1
    thresh = max(2, int(0.6 * len(per_page_texts)))
    common_heads = {s for s, c in head_candidates.items() if c >= thresh}
    common_foots = {s for s, c in foot_candidates.items() if c >= thresh}
    cleaned = []
    for t in per_page_texts:
        lines = [ln for ln in t.splitlines()]
        while lines and lines[0].strip() in common_heads: lines.pop(0)
        while lines and lines[-1].strip() in common_foots: lines.pop()
        cleaned.append("\n".join(lines))
    return cleaned

def clean_pages_list(per_page_texts: list[str]) -> list[str]:
    pages = []
    for t in per_page_texts:
        t = dehyphenate(t)
        t = normalize_spaces(t)
        t = fix_ligatures(t)
        if space_ratio(t) < 0.05:   # tweak threshold if desired
            t = fix_run_together_words(t)
        pages.append(t)
    return remove_repeated_headers_footers(pages)

def postprocess_pages(per_page_texts: list[str]) -> str:
    return "\n\n".join(clean_pages_list(per_page_texts)).strip()

def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def validate_pdf_bytes(data: bytes) -> Tuple[bool, str]:
    if not data:
        return False, "Empty upload. Please choose a PDF file."
    if len(data) > MAX_BYTES:
        return False, f"File is too large ({human_size(len(data))}). Max is {MAX_MB} MB."
    if data[:5] != b"%PDF-":
        return False, "File is not a valid PDF (missing %PDF header)."
    return True, ""

# ─────────────────────────────────────────────────────────────────────────────
# Word/char rebuilders to fix missing spaces
# ─────────────────────────────────────────────────────────────────────────────
def _median(nums):
    if not nums: return 0.0
    s = sorted(nums)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2.0

def rebuild_page_from_chars(p) -> str:
    """Rebuild page string from char boxes; insert spaces on large gaps."""
    chars = p.chars
    if not chars:
        return ""
    chars.sort(key=lambda c: (round(c["top"]), c["x0"]))
    # group into lines
    lines, current, current_top = [], [], None
    line_tol = 2.0
    for ch in chars:
        if current_top is None or abs(ch["top"] - current_top) <= line_tol:
            current.append(ch)
            if current_top is None:
                current_top = ch["top"]
        else:
            lines.append(current)
            current, current_top = [ch], ch["top"]
    if current:
        lines.append(current)

    out_lines = []
    for line in lines:
        if not line:
            out_lines.append("")
            continue
        char_widths = [(c["x1"] - c["x0"]) for c in line if c["text"].strip()]
        med_char_w = max(_median(char_widths), 0.1)
        gaps = [(b["x0"] - a["x1"]) for a, b in zip(line, line[1:])]
        med_gap = _median([g for g in gaps if g > 0])
        thr = min(0.7 * med_char_w, 0.6 * med_gap if med_gap > 0 else 1.2 * med_char_w)
        buf = [line[0]["text"]]
        for a, b in zip(line, line[1:]):
            gap = b["x0"] - a["x1"]
            if gap > thr:
                buf.append(" ")
            buf.append(b["text"])
        out_lines.append("".join(buf))
    text = "\n".join(out_lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

# ─────────────────────────────────────────────────────────────────────────────
# OCR helpers (ocrmypdf)
# ─────────────────────────────────────────────────────────────────────────────
def have_ocrmypdf() -> bool:
    return shutil.which("ocrmypdf") is not None

@st.cache_data(show_spinner=False)
def run_ocr(data: bytes, lang: str = "eng") -> bytes:
    """Run OCRmyPDF and return the OCR'd PDF (bytes). Cached by input bytes + lang."""
    if not have_ocrmypdf():
        raise RuntimeError("ocrmypdf is not installed or not on PATH.")
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.pdf")
        outp = os.path.join(td, "out.pdf")
        with open(inp, "wb") as f:
            f.write(data)
        cmd = [
            "ocrmypdf", "--skip-text",
            "--optimize", "0", "--fast-web-view", "1",
            "--language", (lang.strip() or "eng"),
            "--quiet", inp, outp,
        ]
        subprocess.run(cmd, check=True)
        with open(outp, "rb") as f:
            return f.read()

# ─────────────────────────────────────────────────────────────────────────────
# Extraction (pdfplumber)
# ─────────────────────────────────────────────────────────────────────────────
def extract_with_pdfplumber(data: bytes, start_page: int, end_page: int) -> list[str]:
    """Return list of per-page texts for pages [start_page..end_page]."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        total_pages = len(pdf.pages)
        if total_pages == 0:
            return []
        start_idx = max(0, min(start_page - 1, total_pages - 1))
        end_idx = max(0, min(end_page - 1, total_pages - 1))
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        per_page = []
        for i in range(start_idx, end_idx + 1):
            p = pdf.pages[i]

            # 1) Try pdfplumber text with layout=True (keeps spaces when possible)
            text = p.extract_text(layout=True) or ""

            # 2) If almost no spaces, rebuild using characters (robust)
            if not text or space_ratio(text) < 0.02:
                rebuilt_chars = rebuild_page_from_chars(p)
                if rebuilt_chars and (len(rebuilt_chars) >= len(text)):
                    text = rebuilt_chars

            per_page.append(text or "")
        return per_page

@st.cache_data(show_spinner=False)
def cached_extract_text(data: bytes, start_page: int, end_page: int) -> tuple[str, list[str], int, dict]:
    """Return (full_text, cleaned_pages, total_pages, metadata_dict)."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        total_pages = len(pdf.pages)
        md = pdf.metadata or {}
    raw_pages = extract_with_pdfplumber(data, start_page, end_page)
    cleaned_pages = clean_pages_list(raw_pages)
    full_text = "\n\n".join(cleaned_pages).strip()
    return full_text, cleaned_pages, total_pages, md

# ─────────────────────────────────────────────────────────────────────────────
# Preview (pdf2image)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def render_first_page_preview(data: bytes, dpi: int = 180) -> Optional[Image.Image]:
    """Return PIL.Image of page 1; requires Poppler on system."""
    try:
        pages = convert_from_bytes(data, first_page=1, last_page=1, dpi=dpi)
        return pages[0] if pages else None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Text statistics
# ─────────────────────────────────────────────────────────────────────────────
def text_stats(s: str) -> dict:
    words = re.findall(r"\b\w+\b", s, flags=re.UNICODE)
    return {
        "words": len(words),
        "chars": len(s),
        "chars_no_spaces": len(re.sub(r"\s+", "", s)),
        "lines": s.count("\n") + (1 if s else 0),
        "avg_word_len": (sum(len(w) for w in words) / len(words)) if words else 0.0,
        "reading_time_min": len(words) / 200.0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Prompt & QA helpers (softer guardrails to avoid overusing "I don't know")
# ─────────────────────────────────────────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "You are a helpful assistant answering questions about a PDF.\n"
        "Base your answer primarily on the provided PDF excerpts. You can synthesize "
        "across excerpts and add brief glue logic when helpful, but don't invent facts "
        "that contradict them. If key details are genuinely missing, say so briefly "
        "and explain the most likely answer from the available excerpts.\n\n"
        "Conversation so far:\n{chat_history}\n\n"
        "PDF excerpts (each begins with its page number):\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer thoroughly with clear steps and cite page numbers when relevant."
    ),
)

@st.cache_resource(show_spinner=False)
def build_vector_index(
    pages: list[str],
    start_page: int,
    embeddings_model: str,
    openai_api_key: str,
):
    # Stamp page number into content so it survives any pipeline
    docs: list[Document] = []
    for i, page_text in enumerate(pages):
        page_no = start_page + i
        if page_text.strip():
            docs.append(
                Document(
                    page_content=f"(Page {page_no})\n{page_text}",
                    metadata={"page": page_no},  # keep metadata too
                )
            )
    if not docs:
        return None, None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=embeddings_model, openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 50, "lambda_mult": 0.5},
    )
    return retriever, vectorstore

def build_qa_chain(
    retriever,
    model_name: str,
    temperature: float,
    max_tokens: int,
    openai_api_key: str,
    memory: ConversationBufferMemory,
):
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=openai_api_key,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT,
            "document_separator": "\n\n",   # no document_prompt; we stamped (Page N) in content
        },
        verbose=False,
    )

def run_chain_with_retry(
    chain,
    retriever,
    vectorstore,
    question: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    openai_api_key: str,
    memory: ConversationBufferMemory,
):
    def _extract(res):
        return (res.get("answer") or res.get("result") or ""), (res.get("source_documents") or [])

    res1 = chain({"question": question})
    answer, sources = _extract(res1)
    need_retry = ("i don't know" in answer.lower()) or (len(sources) < 2)

    if not need_retry or vectorstore is None:
        return answer, sources

    # Wider, more diverse retrieval for the retry
    wide_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 14, "fetch_k": 80, "lambda_mult": 0.4},
    )
    wide_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model=model_name, temperature=temperature, max_tokens=max_tokens, openai_api_key=openai_api_key
        ),
        retriever=wide_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT,
            "document_separator": "\n\n",
        },
        verbose=False,
    )
    res2 = wide_chain({"question": question})
    answer2, sources2 = _extract(res2)

    score1 = (len(sources), 0 if "i don't know" in answer.lower() else 1)
    score2 = (len(sources2), 0 if "i don't know" in answer2.lower() else 1)
    return (answer2, sources2) if score2 > score1 else (answer, sources)

# ─────────────────────────────────────────────────────────────────────────────
# Simple source ranking & snippet helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize(s: str) -> list[str]:
    return re.findall(r"\b\w+\b", s.lower())

def rank_sources(sources, question: str, top_k: int = 3):
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return sources[:top_k]
    scored = []
    for d in sources:
        txt = d.page_content or ""
        s_tokens = set(_tokenize(txt))
        overlap = len(q_tokens & s_tokens)
        scored.append((overlap, d))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored[:top_k]]

def make_snippet(text: str, query: str, max_len: int = 240) -> str:
    if not text:
        return ""
    terms = sorted(set(_tokenize(query)), key=len, reverse=True)
    if not terms:
        return (text[:max_len] + "…") if len(text) > max_len else text
    text_l = text.lower()
    pos = -1
    for t in terms:
        pos = text_l.find(t)
        if pos != -1:
            break
    if pos == -1:
        return (text[:max_len] + "…") if len(text) > max_len else text
    start = max(0, pos - max_len // 2)
    end = min(len(text), start + max_len)
    snippet = text[start:end].strip().replace("\n", " ")
    if start > 0:
        snippet = "… " + snippet
    if end < len(text):
        snippet = snippet + " …"
    return snippet

def _page_from_doc(doc):
    pg = (doc.metadata or {}).get("page")
    if pg is not None:
        return pg
    m = re.match(r"\(Page\s+(\d+)\)", (doc.page_content or "").strip())
    return int(m.group(1)) if m else "—"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
    help="Only .pdf files are accepted. Max size 10 MB."
)

use_ocr = st.sidebar.checkbox("Try OCR if no text found", value=True)

# OCR language selection (only English/Turkish)
OCR_LANG_OPTIONS = {"English": "eng", "Turkish": "tur"}
ocr_lang_label = st.sidebar.selectbox("OCR language", list(OCR_LANG_OPTIONS.keys()), index=0)
ocr_lang = OCR_LANG_OPTIONS[ocr_lang_label]

st.sidebar.markdown("---")

# Model selection (chat)
model_name = st.sidebar.selectbox(
    "OpenAI Chat model",
    options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-5-chat-latest"],
    index=1,
)
temperature = st.sidebar.slider("Generation temperature", 0.0, 1.0, 0.2, 0.05)

# Max tokens control
max_tokens = st.sidebar.slider(
    "Max tokens (response length)",
    min_value=256,
    max_value=4096,
    value=1200,
    step=128,
    help="Controls the maximum length of the model's response"
)

# Embedding model selection
embedding_model = st.sidebar.selectbox(
    "Embedding model",
    options=["text-embedding-3-small", "text-embedding-3-large"],
    index=0,
)

# Sidebar: Export Chat History (after first Q&A)
st.sidebar.markdown("### ⬇️ Export Chat History")
msgs_for_export = st.session_state.get("messages", [])
if st.session_state.get("exports_enabled") and len(msgs_for_export) >= 1:
    txt_history = "\n\n".join(f"{m.get('role','').upper()}: {m.get('content','')}"
                              for m in msgs_for_export)
    st.sidebar.download_button(
        label="💾 Download TXT",
        data=txt_history.encode("utf-8"),
        file_name="chat_history.txt",
        mime="text/plain",
        key="export_txt",
    )
    json_history = json.dumps(msgs_for_export, indent=2, ensure_ascii=False)
    st.sidebar.download_button(
        label="💾 Download JSON",
        data=json_history.encode("utf-8"),
        file_name="chat_history.json",
        mime="application/json",
        key="export_json",
    )
else:
    st.sidebar.caption("Ask your first question to enable exports.")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = []
    st.session_state["sources_by_turn"] = []
    st.session_state["exports_enabled"] = False
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main workflow (upload, parse)
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👆 Upload a PDF to begin.")
    st.stop()

errors: list[dict] = []
try:
    data: bytes = uploaded.getvalue()
    ok, msg = validate_pdf_bytes(data)
    if not ok:
        fail(errors, "error", f"Validation failed: {msg}")

    # Preview
    with st.spinner("Rendering preview…"):
        preview_img = render_first_page_preview(data, dpi=180)

    # Metadata
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        total_pages = len(pdf.pages)
        md = pdf.metadata or {}
        title = md.get("Title") or md.get("title")
        producer = md.get("Producer") or md.get("producer")

    # Top info
    st.success("✅ PDF uploaded and opened successfully.")
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        st.write("**Filename:**", uploaded.name)
        st.write("**Filesize:**", human_size(len(data)))
    with colB:
        st.write("**Total Pages:**", total_pages)
        st.write("**Title:**", title or "—")
    with colC:
        st.write("**Producer:**", producer or "—")
        st.write("**Encrypted?:**", "Unknown")

    with st.expander("👁️ Preview — Page 1", expanded=True):
        if preview_img:
            st.image(preview_img, caption="Page 1 preview", use_column_width=True)
        else:
            st.info("Preview not available (Poppler not installed?) or PDF has 0 pages.")

    # Always read the full PDF
    if total_pages == 0:
        fail(errors, "error", "This PDF has 0 pages. Nothing to extract.")
    else:
        start_page, end_page = 1, total_pages

    # Extract text
    with st.spinner(f"Extracting text from pages {start_page}–{end_page}…"):
        full_text, cleaned_pages, _, _ = cached_extract_text(data, start_page, end_page)

    # OCR fallback
    if use_ocr and (not full_text.strip() or len(full_text.strip()) < 40):
        if not have_ocrmypdf():
            add_error(errors, "warning", "OCR requested but `ocrmypdf` not found on PATH.")
        else:
            with st.spinner(f"Running OCR (lang='{ocr_lang}') and re-extracting…"):
                try:
                    ocr_bytes = run_ocr(data, lang=ocr_lang)
                    full_text, cleaned_pages, _, _ = cached_extract_text(ocr_bytes, start_page, end_page)
                except RuntimeError as e:
                    add_error(errors, "info", "OCR skipped", str(e))

    # Final empty-text check
    if not full_text.strip():
        detail = (
            f"filename={uploaded.name}, size={human_size(len(data))}, "
            f"total_pages={total_pages}, pages_read={start_page}–{end_page}, "
            f"ocr_enabled={use_ocr}, ocr_lang={ocr_lang}"
        )
        fail(errors, "error", "No selectable text found.", detail)

except UserAbort:
    full_text = ""
    cleaned_pages = []
except Exception as e:
    add_error(errors, "error", "Unexpected crash", f"{type(e).__name__}: {e}")
    full_text = ""
    cleaned_pages = []

# ─────────────────────────────────────────────────────────────────────────────
# Error Report (sidebar)
# ─────────────────────────────────────────────────────────────────────────────
if errors:
    st.sidebar.markdown("### 🧰 Error Report")
    for i, err in enumerate(errors, start=1):
        level = err.get("level", "info")
        msg = err.get("msg", "")
        detail = err.get("detail")
        if level == "error":
            st.sidebar.error(f"{i}. {msg}")
        elif level == "warning":
            st.sidebar.warning(f"{i}. {msg}")
        else:
            st.sidebar.info(f"{i}. {msg}")
        if detail:
            with st.sidebar.expander(f"Details #{i}"):
                st.sidebar.code(detail)
    if not full_text.strip():
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Success UI
# ─────────────────────────────────────────────────────────────────────────────
st.success("✅ Text extracted successfully.")
cols = st.columns([1, 1, 2])
with cols[0]:
    st.write("**Total Pages:**", total_pages)
with cols[1]:
    st.write("**Pages Read:**", f"{start_page}–{end_page}")
with cols[2]:
    preview_len = 1200
    preview = full_text[:preview_len] + ("… (truncated)" if len(full_text) > preview_len else "")
    st.write("**Preview (first ~1,200 chars):**")
    st.code(preview)

# Stats
stats = text_stats(full_text)
st.subheader("🧮 Text Statistics")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Words", f"{stats['words']:,}")
with m2:
    st.metric("Characters", f"{stats['chars']:,}")
with m3:
    st.metric("Lines", f"{stats['lines']:,}")
with m4:
    st.metric("Avg Word Len", f"{stats['avg_word_len']:.2f}")
st.caption(f"Approx. reading time: **{stats['reading_time_min']:.1f} min** @ 200 wpm")

with st.expander("🧾 Full Extracted Text"):
    st.text_area("Full text", full_text, height=400)

st.download_button(
    label="⬇️ Download extracted text (.txt)",
    data=full_text.encode("utf-8"),
    file_name=(uploaded.name.rsplit(".", 1)[0] + f"_p{start_page}-{end_page}_extracted.txt"),
    mime="text/plain"
)

# ─────────────────────────────────────────────────────────────────────────────
# Q&A with memory (chat UI)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("💬 Ask Questions about this PDF")

# Reset retriever/chain when file/range/embedding/chat model changes
session_sig = (uploaded.name, start_page, end_page, embedding_model, model_name)
if st.session_state.get("session_sig") != session_sig:
    st.session_state["messages"] = []
    st.session_state["sources_by_turn"] = []
    st.session_state["exports_enabled"] = False
    st.session_state["session_sig"] = session_sig
    st.session_state["retriever"] = None
    st.session_state["vectorstore"] = None
    st.session_state["qa_chain"] = None
    st.session_state["memory"] = None

# Build retriever & chain (with memory) if needed
if st.session_state.get("retriever") is None:
    with st.spinner("Indexing the selected pages for Q&A…"):
        retriever, vectorstore = build_vector_index(
            cleaned_pages,
            start_page,
            embeddings_model=embedding_model,
            openai_api_key=OPENAI_API_KEY,
        )
        st.session_state["retriever"] = retriever
        st.session_state["vectorstore"] = vectorstore

if st.session_state.get("memory") is None:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

if st.session_state.get("qa_chain") is None and st.session_state.get("retriever"):
    st.session_state["qa_chain"] = build_qa_chain(
        st.session_state["retriever"],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY,
        memory=st.session_state["memory"],
    )

# Capture new question first so we can control what we render while "thinking"
question = st.chat_input("Ask a question about this PDF…")

# When a new question exists, suppress *previous* Sources expanders to avoid pop-open
hide_prev_sources = bool(question)

# Render prior chat (optionally hiding their "Sources")
assistant_turn = 0
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if (assistant_turn < len(st.session_state["sources_by_turn"])) and not hide_prev_sources:
                srcs = st.session_state["sources_by_turn"][assistant_turn]
                if srcs:
                    # Unique label per turn prevents UI collisions
                    with st.expander(f"📚 Sources • Q{assistant_turn+1}", expanded=False):
                        for i, item in enumerate(srcs, 1):
                            st.markdown(f"{i}. **Page {item['page']}** — {item['snippet']}")
            assistant_turn += 1

# Handle the new question (if any)
if question:
    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    chain = st.session_state.get("qa_chain")
    if not chain:
        with st.chat_message("assistant"):
            st.warning("Q&A is not ready (no retriever). Try reloading.")
    else:
        # Placeholder for the upcoming assistant message
        assistant_placeholder = st.empty()
        with st.spinner("Thinking…"):
            answer, sources = run_chain_with_retry(
                chain=st.session_state["qa_chain"],
                retriever=st.session_state["retriever"],
                vectorstore=st.session_state.get("vectorstore"),
                question=question,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=OPENAI_API_KEY,
                memory=st.session_state["memory"],
            )

        # Rank and build display items
        top_sources = rank_sources(sources, question, top_k=3)
        source_items = []
        for doc in top_sources:
            # page number is stamped in content and kept in metadata; try both
            m = re.match(r"\(Page\s+(\d+)\)", (doc.page_content or "").strip())
            pg = (doc.metadata or {}).get("page", int(m.group(1)) if m else "—")
            snippet = make_snippet(doc.page_content or "", question, max_len=240)
            source_items.append({"page": pg, "snippet": snippet})

        # Render the assistant message + its (collapsed) Sources
        with assistant_placeholder.container():
            with st.chat_message("assistant"):
                st.markdown(answer)
                if source_items:
                    turn_number = assistant_turn + 1  # previous assistants already counted
                    with st.expander(f"📚 Sources • Q{turn_number}", expanded=False):
                        for i, item in enumerate(source_items, 1):
                            st.markdown(f"{i}. **Page {item['page']}** — {item['snippet']}")

        # Persist assistant turn + its sources
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["sources_by_turn"].append(source_items)

        # Enable export (once) and rerun so the top sidebar shows the buttons immediately
        just_enabled = not st.session_state.get("exports_enabled", False)
        st.session_state["exports_enabled"] = True
        if just_enabled:
            st.rerun()