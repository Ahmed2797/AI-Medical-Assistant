from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from medicalai.prompt import prompt_template
from medicalai.chatmodel import openai_chat_model
from medicalai.helper import *

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.prompts import PromptTemplate


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = openai_chat_model()
index_name = "medicalrag"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template_obj = PromptTemplate(
    input_variables=["context", "input"],
    template=prompt_template
)

combine_chain = create_stuff_documents_chain(llm, prompt_template_obj)
rag_chain = create_retrieval_chain(retriever, combine_chain)

def get_medical_answer(query: str):
    response = rag_chain.invoke({"input": query})
    return response["answer"]