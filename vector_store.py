import logging
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def load_or_create_vector_db(
    chunks=None, persist_directory="./chroma_db", embedding_model="nomic-embed-text"
):
    """Load existing vector database or create a new one if documents are provided."""
    try:
        embeddings = OllamaEmbeddings(model=embedding_model)

        # If no chunks provided, try to load existing DB
        if chunks is None:
            if Chroma(
                persist_directory=persist_directory, embedding_function=embeddings
            ).get()["ids"]:
                vector_db = Chroma(
                    persist_directory=persist_directory, embedding_function=embeddings
                )
                logging.info("Existing vector database loaded successfully")
                return vector_db
            else:
                st.error("No existing database found. Please upload a document first.")
                return None

        # Create new vector store from documents
        vector_db = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=persist_directory
        )
        logging.info("Vector database created/updated successfully")
        return vector_db

    except (IOError, ValueError) as e:
        logging.error(f"Error with vector database: {str(e)}")
        st.error("Error with vector database. Please try again.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error with vector database: {str(e)}")
        st.error("Unexpected error occurred. Please contact support.")
        return None
