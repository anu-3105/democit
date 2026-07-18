# Local RAG Pipeline Project

This project demonstrates a local Retrieval-Augmented Generation (RAG) pipeline using LangChain and ChromaDB. It allows you to query different types of datasets independently.

## Prerequisites

1.  Make sure you have Python installed.
2.  Obtain a Google Gemini API key if you want generative answers (otherwise, it will perform pure retrieval).

## Step 1: Set Up the Environment

First, open your terminal (PowerShell) and navigate to your project directory.

1.  **Activate the virtual environment** (it has already been created for you):
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
2.  **Set your Google API Key and Model** in the `.env` file to enable generated answers. The scripts will automatically load these settings.

## Step 2: Ingest Data into ChromaDB

Before you can query anything, you must load your data into the vector database. We have two separate datasets and scripts.

**To ingest the E-commerce data:**
```powershell
python ingest_ecommerce.py
```
*(This reads `ecommerce_data.csv` and saves it to the `ecommerce` collection in the `chroma_db` folder).*

**To ingest the Python MCQs data:**
```powershell
python ingest_mcq.py
```
*(This reads `python_mcqs.txt` and saves it to the `python_mcqs` collection in the `chroma_db` folder).*

## Step 3: Query the Data

Once the data is ingested, you can query each dataset independently using its dedicated script.

**Query the E-commerce dataset:**
```powershell
python query_ecommerce.py "What is the revenue for customer 1102?"
```

**Query the Python MCQs dataset:**
```powershell
python query_mcq.py "Which of the following is a mutable data type in Python?"
```

---

### Expected Output
- **With an API key:** The script will print a generated "ANSWER" directly answering your question, followed by the "SOURCES" it retrieved to form that answer.
- **Without an API key:** The script will just print the "RETRIEVED DOCUMENTS" (the chunks of text it found in the database most relevant to your query).
