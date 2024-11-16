import streamlit as st
from langchain_community.llms import ctransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import BedrockChat
import boto3
from langchain_community.llms.openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Khai báo biến
api_key = os.getenv("openai_api_key")

vector_db_path = "vectorstores/db_faiss"

# Tạo prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return llm_chain

def read_vectors_db():
    embedding_model = OpenAIEmbeddings(api_key=api_key)
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Read vector DB and load model
db = read_vectors_db()
llm = OpenAI(api_key=api_key, temperature=0)

template = """Answer the question based only on the following context:

{context}

---

Answer the question based on the above context, answer by vietnamese: {question}
"""
prompt = creat_prompt(template)

llm_chain = create_qa_chain(prompt, llm, db)

# Streamlit interface
st.title("AI Question Answering System")

st.write("""
    Nhập câu hỏi của bạn vào ô dưới đây và mô hình sẽ trả lời dựa trên thông tin đã được huấn luyện.
""")

question = st.text_input("Câu hỏi của bạn:")

if question:
    response = llm_chain.invoke({"query": question})
    st.write("Câu trả lời:")
    st.write(response)
