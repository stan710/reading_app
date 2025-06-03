import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install langchain before using it
install_package("langchain")
install_package("langchain-community")
install_package("langchain_openai")
install_package("streamlit")


from langchain.document_loaders.pdf import PyPDFDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.schema import Document 
from langchain.vectorstores.chroma import Chroma 
from dotenv import load_dotenv 
from langchain.chat_models import ChatOpenAI 
import os 
import shutil 
from typing import List
from langchain.prompts import ChatPromptTemplate
import os
from langchain_openai import ChatOpenAI
import streamlit as st

import requests
import zipfile  

openai_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "https://github.com/stan710/reading_app/db"


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query):
    """
    Query using Chroma database.

    Args:
    - query_text (str): The text to query with.

    Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
    """
    # Use same embedding function as before
    embedding_function = OpenAIEmbeddings()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieving the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query, k=3)
    
    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.8:
        print(f"Unable to find matching results.")

    # Combine context from matching documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Initialize OpenAI chat model
    model = ChatOpenAI(model_name="gpt-4o")
    
    # Generate response text based on the prompt
    response_text = model.predict(prompt)

    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text, sources



# Streamlit UI
st.title("Query Interface - pls enter your question relating to large language models (but I'm just a baby so may not make sense)")
query = st.text_input("Enter your question:")

if st.button("Submit"):
    formatted_response, response_text, sources = query_rag(query)
    st.write("### Response:")
    st.write(response_text)

    if sources:
        st.write("### Sources:")
        for source in sources:
            st.write(f"- {source}")
