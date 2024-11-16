# Chatbot Trả Lời Các Câu Hỏi Về Luật Pháp Việt Nam

Chatbot này được phát triển để cung cấp các câu trả lời cho các câu hỏi liên quan đến luật pháp Việt Nam, dựa trên kiến trúc RAG

## Cách cài đặt
Trước tiên hãy lấy openai_api_key tại [API Keys của OpenAI](https://platform.openai.com/settings/organization/api-keys)

### 1. Cài đặt các thư viện cần thiết
Sử dụng câu lệnh sau để cài đặt các thư viện yêu cầu:
```bash
pip install -r setup.txt
```
### 2. Chunking và embedding
```bash
python prepare_vector_db.py
```

### 3. Tạo chatbot
```bash
streamlit run qabot.py
