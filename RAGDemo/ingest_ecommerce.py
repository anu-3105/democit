import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CSV_FILE_PATH = "ecommerce_data.csv"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ecommerce"

def main():
    print(f"Loading ecommerce data from {CSV_FILE_PATH}...")
    loader = CSVLoader(file_path=CSV_FILE_PATH, encoding='utf-8')
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
