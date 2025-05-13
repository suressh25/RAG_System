import sys
import streamlit as st
import logging
from document_processor import process_uploaded_file
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Constants
LLAMA_MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "./chroma_db"


def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    try:
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Documents split successfully into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error splitting documents: {str(e)}")
        st.error("Error splitting documents. Please try again.")
        return None


def load_or_create_vector_db(chunks=None):
    """Load existing vector database or create a new one."""
    from vector_store import load_or_create_vector_db

    return load_or_create_vector_db(chunks, PERSIST_DIRECTORY, EMBEDDING_MODEL)


def create_retriever(vector_db, llm):
    """Create a retriever for the RAG system."""
    try:
        retriever = vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )

        # Wrap with MultiQueryRetriever
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
        logging.info("Retriever created successfully")
        return retriever_from_llm
    except Exception as e:
        logging.error(f"Error creating retriever: {str(e)}")
        st.error("Error creating retriever. Please try again.")
        return None


def create_chain(retriever, llm):
    """Create the RAG chain."""
    try:
        # Create the prompt template
        template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        Answer: """

        # Create the prompt
        prompt = ChatPromptTemplate.from_template(template)

        # Create the chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logging.info("Chain created successfully")
        return chain
    except Exception as e:
        logging.error(f"Error creating chain: {str(e)}")
        st.error("Error creating chain. Please try again.")
        return None


def main():
    st.title("Document Assistant")

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your document (PDF, Word, Excel, PowerPoint, Image, or Text file)",
        type=["pdf", "docx", "xlsx", "pptx", "png", "jpg", "jpeg", "txt"],
    )

    # Initialize vector_db as None
    vector_db = None

    # Process the uploaded file
    if uploaded_file:
        documents = process_uploaded_file(uploaded_file)
        if documents:
            # Split and index the documents
            chunks = split_documents(documents)
            vector_db = load_or_create_vector_db(chunks)
            if vector_db:
                st.success("Document processed and indexed successfully!")

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input and vector_db:
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model
                llm = ChatOllama(model=LLAMA_MODEL_NAME)

                # Create the retriever
                retriever = create_retriever(vector_db, llm)
                logging.info("Retriever created successfully.")

                # Create the chain
                chain = create_chain(retriever, llm)
                logging.info("Chain created successfully.")

                # Get the response
                response = chain.invoke(user_input)
                logging.info("Response generated successfully.")

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                st.error(
                    "An error occurred while processing your request. Please try again later."
                )
    elif user_input:
        st.warning("Please upload a document first before asking questions.")


if __name__ == "__main__":
    main()
