import subprocess
import sys
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Install necessary packages
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package("langchain")
install_package("langchain-community")
install_package("langchain_openai")
install_package("streamlit")

# Load API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Define Chroma database path
CHROMA_PATH = "https://github.com/stan710/reading_app/chroma_db"

from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.

    Args:
    - query_text (str): The text to query the RAG system with.

    Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
    """
    # YOU MUST - Use same embedding function as before
    embedding_function = OpenAIEmbeddings()
    
    # Prepare the database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, distance_metric="cosine")

    # Retrieving the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.8:
        print(f"Unable to find matching results.")

    # Combine context from matching documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize OpenAI chat model
    model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_key)
    
    # Generate response text based on the prompt
    response_text = model.predict(prompt)

    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text


# Streamlit UI
st.title("Query Interface - Ask about Large Language Models")
query = st.text_input("Enter your question:")
if st.button("Submit"):
    formatted_response, response_text = query_rag(query)
    st.write(response_text)