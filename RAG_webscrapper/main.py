import os
import getpass
from bs4 import SoupStrainer
from langchain_groq import ChatGroq
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 1. Load environment variables or prompt for API key


# 2. Initialize the LLM
print("Initializing LLM...")
llm = ChatGroq(model="llama3-8b-8192")
print("LLM initialized successfully.")

# 3. Function to scrape, chunk, and index web pages
def scrape_and_process(urls):
    """
    Scrape and process web pages into chunks for embedding.
    Args:
        urls (list): List of web page URLs to scrape.
    Returns:
        vectorstore: A Chroma vector store for the processed documents.
    """
    print("Scraping and processing web pages...")

    # Scrape web pages
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    documents = loader.load()

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Embed the documents and store them in a vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("Web pages successfully indexed.")
    return vectorstore


# 4. Function to create the RAG chain
def create_rag_chain(vectorstore):
    """
    Create a RAG chain for answering questions.
    Args:
        vectorstore: A Chroma vector store for retrieving documents.
    Returns:
        rag_chain: A RAG chain for answering questions.
    """
    retriever = vectorstore.as_retriever()

    # Load the prompt from LangChain's prompt hub
    prompt = hub.pull("rlm/rag-prompt")

    # Create a runnable chain for RAG
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain successfully created.")
    return rag_chain


# 5. Main function to scrape, process, and answer questions
def main():
    # List of web pages to scrape
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        # Add more URLs here if needed
    ]

    # Scrape and process web pages
    vectorstore = scrape_and_process(urls)

    # Create RAG chain
    rag_chain = create_rag_chain(vectorstore)

    # Query the chain
    queries = [
        "What is Task Decomposition?",
        "What are the components of an agent?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_chain.invoke(query)
        print(f"Answer: {response}")


if __name__ == "__main__":
    main()
