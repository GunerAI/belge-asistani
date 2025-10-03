# 📄 PDF Uploader & Reader (Streamlit)

Upload a PDF, preview page 1, extract clean text, see quick stats, and **chat with your document** using an OpenAI-backed Q&A chain (LangChain + FAISS).

---

## ✨ Features
- Drag-and-drop **PDF upload** (up to 10 MB)
- **First-page preview** (via `pdf2image` — Poppler required)
- Robust **text extraction & cleanup** (pdfplumber + heuristics, optional OCR fallback)
- **Per-document vector index** (FAISS) with **conversational Q&A** (LangChain)
- **Chat history export** (TXT/JSON)
- Tweakable **models, temperature, tokens**, and **embedding model** from the sidebar

---

## 🚀 Quickstart (Local)

### 1) Clone & enter the project
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```
### 2) Python env & dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 3) Secrets / API key
Create a .env file in the project root:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 4) (Optional) Enable preview / OCR
•	Poppler (for pdf2image preview)
	-	macOS: brew install poppler
	-	Ubuntu/Debian: sudo apt-get install poppler-utils
	-	Windows: install Poppler and add bin/ to PATH.
•	OCR (fallback if the PDF has no selectable text)
	-	ocrmypdf + Tesseract:
	-	macOS: brew install ocrmypdf tesseract
	-	Ubuntu/Debian: sudo apt-get install ocrmypdf tesseract-ocr

### 5) Run
```bash
streamlit run app.py
```

## 🧭 How to Use
1.	Upload a PDF (≤10 MB).
2.	Inspect the page-1 preview, metadata, and extracted text preview.
3.	Review Text Statistics or download the extracted text.
4.	Ask questions in “💬 Ask Questions about this PDF” — answers cite page numbers when relevant.
5.	Export chat history from the sidebar (TXT/JSON).

## 📷 Screenshots








