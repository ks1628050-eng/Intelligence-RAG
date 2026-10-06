# Intelligence RAG: Zyro Dynamics HR AI Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FAISS**, **HuggingFace Sentence Transformers**, and **Groq LLMs** to answer internal employee HR queries strictly grounded in company policy documents.

---

## 📌 Project Overview

This project implements an enterprise-grade HR policy question-answering assistant for **Zyro Dynamics**. The assistant processes official company policies, indexes document chunks into a vector database, and synthesizes answers using retrieval-grounded LLM inference with strict guardrails to eliminate hallucinations.

---

## 🚀 Key Features

- **Multi-Document Ingestion**: Parses and extracts multi-page PDF documents across 11 internal HR policies preserving page-level source metadata.
- **Semantic Chunking**: Applies `RecursiveCharacterTextSplitter` (chunk size: 900, overlap: 150) to retain critical numerical criteria, eligibility rules, and procedural steps.
- **Dense Vector Search**: Generates normalized embeddings with `sentence-transformers/all-MiniLM-L6-v2` stored in an in-memory **FAISS** vector database.
- **Maximal Marginal Relevance (MMR) Retrieval**: Balances relevance and diversity (`k=8`, `fetch_k=32`, `lambda=0.70`) to capture multi-aspect policy answers.
- **Strict Grounding & Policy Guardrails**:
  - Direct answers grounded strictly in retrieved context.
  - Preservation of exact figures, percentages, dates, and grade criteria.
  - Out-of-scope refusal handling for non-company or unsupported queries.
  - Accurate citation of source document names and page numbers.
- **Automated Evaluation Pipeline**: Evaluates test queries and generates standardized `submission.csv` outputs.

---

## 📂 Repository Structure

```text
├── zyro-dynamics-hr-corpus/          # Official HR Policy PDFs
│   ├── 00_Company_Profile.pdf
│   ├── 01_Employee_Handbook.pdf
│   ├── 02_Leave_Policy.pdf
│   ├── 03_Work_From_Home_Policy.pdf
│   ├── 04_Code_of_Conduct.pdf
│   ├── 05_Performance_Review_Policy.pdf
│   ├── 06_Compensation_and_Benefits_Policy.pdf
│   ├── 07_IT_and_Data_Security_Policy.pdf
│   ├── 08_Prevention_of_Sexual_Harassment_Policy.pdf
│   ├── 09_Onboarding_and_Separation_Policy.pdf
│   └── 10_Travel_and_Expense_Policy.pdf
├── kedar1628050.ipynb                 # Complete end-to-end RAG pipeline notebook
├── code.ipynb                         # Scratch / setup notebook
├── .gitignore                         # Git ignore rules
└── README.md                          # Project documentation
```

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.10+
- **Framework**: LangChain (`langchain-core`, `langchain-community`, `langchain-groq`, `langchain-huggingface`)
- **Vector Database**: FAISS (`faiss-cpu`)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Inference**: Groq API
- **Document Processing**: `pypdf`, `pandas`

---

## ⚡ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ks1628050-eng/Intelligence-RAG.git
   cd Intelligence-RAG
   ```

2. **Install required packages**:
   ```bash
   pip install langchain langchain-community langchain-core langchain-groq langchain-huggingface faiss-cpu pypdf sentence-transformers pandas
   ```

3. **Set your Groq API Key**:
   ```bash
   # On Linux/macOS
   export GROQ_API_KEY="your_groq_api_key_here"

   # On Windows (PowerShell)
   $env:GROQ_API_KEY="your_groq_api_key_here"
   ```

4. **Run the pipeline**:
   Open and execute [`kedar1628050.ipynb`](kedar1628050.ipynb) in Jupyter Notebook, VS Code, or Kaggle/Colab.