import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "python_mcqs"

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Query the Python MCQ RAG pipeline.")
    parser.add_argument("query", type=str, help="The query string to search for.")
    args = parser.parse_args()

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"Loading Chroma DB from {CHROMA_DB_DIR}, collection: {COLLECTION_NAME}...")
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"Error: Chroma database directory '{CHROMA_DB_DIR}' not found. Please run ingest_mcq.py first.")
        sys.exit(1)
        
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    llm_model = os.environ.get("LLM_MODEL", "gemini-1.5-flash")
    
    if google_api_key:
        print(f"\n[Using Google Gemini ({llm_model}) for RAG Generation]")
        llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=google_api_key)
        
        system_prompt = (
            "You are an assistant designed to answer multiple-choice questions on Python. "
            "Use the following pieces of retrieved context to answer the question. "
            "Provide the correct option and a brief explanation based on the context. "
            "If you don't know the answer, just say that you don't know. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        print(f"\nQuerying: '{args.query}'\n")
        
        # Manually retrieve and format context to avoid version-specific chain imports
        docs = retriever.invoke(args.query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        messages = prompt.format_messages(context=context, input=args.query)
        response = llm.invoke(messages)
        
        print("-" * 50)
        print("ANSWER:")
        print(response.content)
        print("-" * 50)
        print("SOURCES (Retrieved Data):")
        for i, doc in enumerate(docs):
            print(f"\nSource {i+1}:\n{doc.page_content}")
            print("-" * 20)
    else:
        print("\n[No GOOGLE_API_KEY found. Performing Document Retrieval only]")
        print(f"\nQuerying: '{args.query}'\n")
        
        docs = retriever.invoke(args.query)
        print("-" * 50)
        print("RETRIEVED DOCUMENTS:")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:\n{doc.page_content}")
            print("-" * 20)
        print("\nNote: To generate a synthesized natural language answer, set the GOOGLE_API_KEY environment variable.")

if __name__ == "__main__":
    main()
