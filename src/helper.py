import os

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser # to parse language

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language

from langchain.embeddings import OpenAIEmbeddings

