from langchain_community.document_loaders import TextLoader
import os

def load_document(file_path):
    """
    Loads a text document from the given file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at the specified path: {file_path}")

    loader = TextLoader(file_path)
    documents = loader.load()
    return documents
