import os
from git import Repo

from langchain_community.document_loaders.generic import GenericLoader
#from langchain.document_loaders.parsers import LanguageParser # to parse language

from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language

#from langchain.embeddings import OpenAIEmbeddings # Depreciated

from langchain_openai import OpenAIEmbeddings

def repo_ingestion(repo_url):
    os.makedirs("/repo",exist_ok = True)
    repo_path = "repo/"
    Repo.clone_from(repo_url,to_path=repo_path)

def repo_load(repo_path):
    loader = GenericLoader.from_filesystem(repo_path, 
                             glob="**/[!.]*",
                             suffixes = [".py"],
                             parser = LanguageParser(language=Language.PYTHON, parser_threshold=500),
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
    print(len(text_chunks))
    return text_chunks


def load_embedding():

    embeddings = OpenAIEmbeddings()
    return embeddings
