import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader
}

def process_file(file_path: str, file_extension: str):
    loader_class = SUPPORTED_EXTENSIONS.get(file_extension.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_extension}")
    return loader_class(file_path).load()

def process_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
        
        try:
            docs = process_file(temp_path, file_extension)
            documents.extend(docs)
        finally:
            os.remove(temp_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)