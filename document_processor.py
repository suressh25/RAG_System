from typing import List
import streamlit as st
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader,
    TextLoader,
)
from langchain.docstore.document import Document


def load_document(file_upload) -> List[Document]:
    """Load and process an uploaded document based on its file type."""
    # Create a temporary file to store the uploaded content
    file_type = file_upload.type
    temp_file_path = f"temp_upload_{file_upload.name}"

    try:
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_upload.getvalue())
    except IOError as e:
        st.error(f"Error writing temporary file: {str(e)}")
        raise IOError(f"Error writing temporary file: {str(e)}")  # import logging

    try:
        # Select appropriate loader based on file type
        if file_type == "application/pdf":
            loader = UnstructuredPDFLoader(temp_file_path)
        elif file_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ]:
            loader = Docx2txtLoader(temp_file_path)
        elif file_type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ]:
            loader = UnstructuredExcelLoader(temp_file_path)
        elif file_type in [
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.ms-powerpoint",
        ]:
            loader = UnstructuredPowerPointLoader(temp_file_path)
        elif file_type.startswith("image/"):
            loader = UnstructuredImageLoader(temp_file_path)
        elif file_type == "text/plain":
            loader = TextLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        documents = loader.load()
        return documents

    finally:
        # Clean up the temporary file
        import os

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def process_uploaded_file(uploaded_file):
    """Process an uploaded file and prepare it for the RAG system."""
    if uploaded_file is None:
        return None

    try:
        documents = load_document(uploaded_file)
        st.success(f"Successfully loaded {uploaded_file.name}")
        return documents
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
