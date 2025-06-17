# src/pipeline/agent_tools/knowledge_base_tool.py

import os
from typing import List, Union

from langchain.textsplitter.base import TextSplitter
from langchain.textsplitter.recursion import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.documentloaders.pdf import PyPDFLoader
from langchain.documentloaders.web_base import WebBaseLoader
from langchain.prompts.base import PromptTemplate
from langchain.embedding.openai import OpenAIEmbedding


def load_data(sources: List[str]):
    """Load documents from various sources (PDFs, URLs, or text files)."""
    docs = []
    for src in sources:
        if src.startswith("http://") or src.startswith("https://"):
            loader = WebBaseLoader(src)
            docs.extend(loader.load())  
        elif src.endswith(".pdf"):
            loader = PyPDFLoader(src)
            docs.extend(loader.load()) 
        elif src.endswith(".txt"):
            with open(src, "r") as f:
                docs.append(f.readlines()) 
    return docs


def split_data(docs, chunk_size=500, chunk_overlap=50):
    """Split documents into semantic chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def embed_data(splits, embedding=None):
    """Create FAISS vector store from documents with embeddings."""
    if not embedding:
        embedding = OpenAIEmbedding(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_documents(splits, embedding)
    return db


def create_retriever(sources: List[str]):
    """Create a semantic search retriever from documents."""
    docs = load_data(sources)
    splits = split_data(docs)
    db = embed_data(splits)
    return db.as_retriever()


def search_knowledge_base(retriever, question: str) -> List[str]:
    """Search knowledge base for related documents."""
    return retriever.get_relevant_documents(question)
