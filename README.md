# Complaints-Classifier-RAG
Using retrieval augmented generation to classify customer complaints into L1, L2 and L3 reported problem codes

## Process Overview

### Storing Data in Vector DB
![Storing Data](/images/storing_data.png)

### Retrieval, Reranking, and Sending to LLM
![RAG Pipeline](/images/rag_pipeline.png)

1. **Storing Data**: Data is stored in a vector database for efficient retrieval.
2. **Retrieval**: Relevant data is retrieved from the vector database.
3. **Reranking**: Retrieved data is reranked based on relevance.
4. **LLM Processing**: The reranked data is sent to the LLM for generating the final output.

## Prerequisites
- Install [Ollama](https://ollama.com)
- Download models: `deepseek-r1`, `llalma3.2` etc..

## Setup

### Windows
1. Create a virtual environment:
    ```sh
    python -m venv venv
    ```
2. Activate the virtual environment:
    ```sh
    .\venv\Scripts\activate
    ```
3. Install necessary packages:
    ```sh
    pip install -r requirements.txt
    ```

### Linux
1. Create a virtual environment:
    ```sh
    python3 -m venv venv
    ```
2. Activate the virtual environment:
    ```sh
    source venv/bin/activate
    ```
3. Install necessary packages:
    ```sh
    pip3 install -r requirements.txt
    ```

## Running the application

- In the terminal after you have activated the virtual env run `streamlit run streamlit_app.py`.
- This will launch the web app on `localhost:8051`.
- Press the `Poulate DB` button on the sidebar to create and store the embeddings from the excel into the vector DB.
- Select one of the LLM models from the selection box and ask your query.
