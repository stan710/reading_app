# Chromadb Streamlit App

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and Streamlit.

## Features

- Loads and splits PDF documents
- Embeds and stores data in ChromaDB
- Provides a Streamlit interface for querying with OpenAI models

## Setup

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Set your OpenAI API key in a `.env` file:
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run streamlit.py
    ```

## Usage

- Enter your question in the Streamlit interface and get answers based on your PDF data.

## Project Structure

- `Setup.ipynb`: Data loading, chunking, and vector store creation
- `streamlit.py`: Streamlit web interface
- `requirements.txt`: Python dependencies
