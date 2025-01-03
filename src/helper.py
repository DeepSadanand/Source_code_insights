import os
from git import Repo

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser # to parse language

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language

from langchain.embeddings import OpenAIEmbeddings

def repo_ingestion(repo_url):
    os.makedirs("/repo",exist_ok = True)
    repo_path = "repo/"
    Repo.clone_from(repo_url,repo_path)

def repo_load():
    loader = GenericLoader(repo_path, 
                             glob="**/**",
                             suffixes = [".py"],
                             parser = LanguageParser(Language = "python",parser_threshold = 200),
                             )
    documents = loader.load()

    return documents


def text_splitter(documents):
    document_splitter = RecursiveCharacterTextSplitter.from_language(
        language = Language.PYTHON, 
        chunk_size = 200, 
        chunk_overlap = 200
    )

    text_chunks = document_splitter.split_documents(documents)
    return text_chunks


def load_embedding():

    embeddings = OpenAIEmbeddings(disallowed_special = 1)
    return embeddings
