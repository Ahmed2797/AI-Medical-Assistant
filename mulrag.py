import os
import base64
import shutil
# import streamlit as st
from PIL import Image
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- CONFIGURATION & DIRECTORIES ---
CHROMA_PATH = "./chroma_db"
IMAGE_DIR = "figures"

for folder in [IMAGE_DIR, CHROMA_PATH]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class MultiModalRAG:
    def __init__(self):
        # High-reasoning model (Cheap & Fast)
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        self.vectorstore = Chroma(
            collection_name="multirag",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=CHROMA_PATH
        )

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def summarize_image(self, image_path):
        """Standardizes visual data into searchable text."""
        b64_image = self.encode_image(image_path)
        response = self.llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Analyze this image from a document. Describe charts, data points, or visual content for a search index."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ])
        ])
        return response.content

    def process_pdf(self, file_path):
        """Industry-standard partitioning for PDF extraction."""
        elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1500,
            image_output_dir_path=IMAGE_DIR
        )
        
        for el in elements:
            metadata = {"source": file_path, "type": "text"}
            if "Image" in str(type(el)):
                img_path = el.metadata.image_path
                content = self.summarize_image(img_path)
                metadata.update({"type": "image", "image_path": img_path})
            else:
                content = el.text
            
            if content:
                self.vectorstore.add_texts(texts=[content], metadatas=[metadata])

    def query_system(self, user_input, is_image=False):
        """Cross-modal retrieval logic."""
        search_query = user_input
        if is_image:
            search_query = self.summarize_image(user_input)
        
        # Retrieve top matches
        docs = self.vectorstore.similarity_search(search_query, k=3)
        
        # Formulate answer based on context
        context_text = "\n\n".join([d.page_content for d in docs])
        prompt = [
            # SystemMessage(content="Answer the question using ONLY the provided context. Show expertise and clarity."),
            # SystemMessage(content="Answer based ONLY on the provided context. If an image is relevant, mention it."),
            SystemMessage(content="You are a helpful assistant. Answer the question using ONLY the provided context and if an image is relevant, mention it. If the answer isn't in the context, say you don't know."),

            HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {search_query}")
        ]
        answer = self.llm.invoke(prompt).content
        return answer, docs
