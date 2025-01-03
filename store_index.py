from src.helper import repo_ingestion, repo_load, load_embedding, text_splitter

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma

import os

env_var  = load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

documents = repo_load("repo/")
text_chunks= text_splitter(documents)

print("**********************-----",len(text_chunks))
embeddings = load_embedding()


vectordb = Chroma.from_documents(text_chunks, embedding=embeddings,persist_directory= "/db")
vectordb.persist()
