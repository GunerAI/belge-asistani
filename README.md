# 📄 Belge Asistanı — Streamlit PDF Q&A

This project is a **Streamlit application** that allows users to upload a PDF file, preview it, extract text, and then **ask questions** about the content using **LangChain** and **OpenAI**.  
It also supports chat-style interaction with memory, additional PDF utilities, and extra features for usability.

---

## 🎯 Features

### ✅ Core Requirements (70 pts)
- **PDF Upload**  
  - Accepts only `.pdf` files (max 10 MB)  
  - Validates and confirms successful upload  
  - Rejects other file types  

- **PDF Processing**  
  - Extracts text using `pdfplumber` (or OCR fallback)  
  - Displays extracted text and handles empty/unreadable PDFs  
  - Provides error handling and feedback  

- **Q&A System**  
  - Accepts user questions  
  - Uses LangChain and OpenAI to generate answers based on PDF content  
  - Includes PDF content in the prompt  
  - Maintains **conversation history (memory)**  

- **Streamlit UI**  
  - Clean, user-friendly design  
  - Chat-style messaging with `st.chat_message`  
  - Shows PDF details (filename, size, total pages, metadata)  

### 🌟 Extra Features (30 pts)
Implemented **more than two** extra features:
- 📑 **Page Count Display** — Shows total pages and selected page range  
- 🧠 **Different Models** — Choose OpenAI model & temperature from sidebar  
- 👁 **PDF Preview** — First page image preview (via `pdf2image` + Poppler)  
- 🧹 **Clear Chat** — Button to reset conversation history  
- 🧮 **Character/Word Count** — Text statistics (words, chars, lines, avg word length, estimated reading time)  
- 💾 **Chat Download** — Export chat history as TXT or JSON  

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework  
- [LangChain](https://www.langchain.com/) — LLM orchestration  
- [OpenAI API](https://platform.openai.com/) — LLM provider  
- [pdfplumber](https://pypi.org/project/pdfplumber/) — PDF text extraction  
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search  
- [pdf2image](https://github.com/Belval/pdf2image) — PDF preview rendering  
- [python-dotenv](https://github.com/theskumar/python-dotenv) — Environment variable management  

---

## 📦 Installation

Clone the repository and set up your environment:

```bash
git clone https://github.com/GunerAI/belge-asistani.git
cd belge-asistani
```

# Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

```bash
# Install dependencies
pip install -r requirements.txt
```
