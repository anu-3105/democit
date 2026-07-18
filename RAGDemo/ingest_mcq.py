import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

TEXT_FILE_PATH = "python_mcqs.txt"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "python_mcqs"

def main():
    print(f"Loading Python MCQs from {TEXT_FILE_PATH}...")
    loader = TextLoader(file_path=TEXT_FILE_PATH, encoding='utf-8')
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents.")
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"Ingesting data into Chroma DB collection '{COLLECTION_NAME}' at {CHROMA_DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
