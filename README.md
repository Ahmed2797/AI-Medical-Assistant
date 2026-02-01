# AI-Medical-Assistant

***AI-powered medical chatbot using RAG architecture to provide accurate healthcare responses, built with LLMs, Pinecone vector database, and deployed on AWS.***

## step 1

### Clone the repository

    git clone https://github.com/Ahmed2797/AI-Medical-Assistant.git

## step 2

### Create enverionment & install

    conda create -n ai python=3.10 -y
    conda activate ai

    pip install -r requirements.txt

## Dataset(pdf)

    https://drive.google.com/file/d/1iVvTfm3sxA1xmaXbcWHmbcGNJbghi3xn/view?usp=sharing

## step 3

### Create a .env file in the root directory and add your Pinecone & openai credentials as follows

    PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxx"
    OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxx"

    run:
    python store_pinecone.py
    python app.py
