from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
# Khai bao bien
api_key = os.getenv("openai_api_key")
data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# Ham1: Tao ra vector DB tu 1 doan text
def create_db_from_text():
    raw_text = """ Luật này quy định về quy hoạch, đầu tư, xây dựng, bảo vệ, quản lý, bảo trì và phát triển kết cấu 
    hạ tầng đường sắt; công nghiệp đường sắt, phương tiện giao thông đường sắt; tín hiệu, quy tắc 
    giao thông và bảo đảm trật tự, an toàn giao thông đường sắt; kinh doanh đường sắt; quyền và 
    nghĩa vụ của tổ chức, cá nhân có liên quan đến hoạt động đường sắt; quản lý nhà nước về hoạt 
    động đường sắt """
    
    text_splitter = CharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_text(raw_text)

    #Embedding
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    #Tao Faiss vector db
    db = FAISS.from_texts(texts = chunks, embedding = embedding_model)
    db.save_local(vector_db_path)

    return db

def create_db_from_files():
    document_loader = PyPDFDirectoryLoader(data_path)
    document = document_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len,
        separators=["##", "###", "#", "\n#", "\n###", "\n##"]
    )
    chunks = text_splitter.split_documents(document)
    # Lấy 3 chunk đầu tiên
    first_three_chunks = chunks[:10]
    
    # In từng chunk
    for i, chunk in enumerate(first_three_chunks, start=1):
        print(f"Chunk {i}:")
        print(chunk)
        print("-" * 20)
    #embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    # Doi model embedding
    #embedding_model = get_embedding_function()
    embedding_model = OpenAIEmbeddings(api_key=api_key)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_files()

