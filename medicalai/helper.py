from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from typing import List
from langchain_classic.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings



def document_loader(path: str) -> List[Document]:
    """
    Load all PDF files from a given directory (recursively) and return them as LangChain Document objects.

    This function uses `DirectoryLoader` with `PyPDFLoader` to read PDF files.
    It supports multithreading for faster loading of large datasets and prints the number of pages loaded.

    Args:
        path (str): Path to the directory containing PDF files. Subfolders are also searched recursively.

    Returns:
        List[Document]: A list of `Document` objects, one per page of each PDF.
                        Returns an empty list if no PDFs are found.

    Example:
        >>> documents = document_loader("Data/MedicalPDFs")
        Loaded pages: 123
        >>> len(documents)
        123
    """
    loader = DirectoryLoader(
        path=path,
        glob="**/*.pdf",           # search all PDFs recursively
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )

    documents = loader.load()

    if not documents:
        print(f"No PDFs found in {path}")
    else:
        print(f"Loaded pages: {len(documents)}")

    return documents



def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split a list of LangChain Document objects into smaller chunks.

    Args:
        documents (List[Document]): List of loaded documents (PDF pages, etc.)
        chunk_size (int, optional): Size of each text chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.

    Returns:
        List[Document]: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Chunks created: {len(chunks)}")
    return chunks



def embed_text(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """
    Generate embeddings for a given text using a HuggingFace Sentence-Transformer model.

    Args:
        model_name (str, optional): The HuggingFace model to use. Defaults (384) to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        List[float]: The embedding vector as a list of floats.

    """
    embeddings = HuggingFaceEmbeddings(model=model_name)

    return embeddings


def load_openai_embeddings(model_name: str = "text-embedding-3-small"):
    """
    Generate embeddings for a given text using a OpenAIEmbeddings model.

    Args:
        model_name (str, optional): The OpenAIEmbeddings model to use. Defaults to 'text-embedding-3-small'.

    Returns:
        List[float]: The embedding vector as a list of floats.

    """
    return OpenAIEmbeddings(
        model=model_name
        # model="text-embedding-3-small"
    )
