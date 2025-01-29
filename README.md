**Hybrid Search with FAISS and BM25**

📌 Overview
This project implements a hybrid search engine combining FAISS (vector-based similarity search) and BM25 (lexical search) to retrieve the most relevant documents for a given query. It utilizes LangChain, Groq’s Llama3-8B, FAISS, and BM25 to process and analyze documents efficiently.

🚀 Features
Multi-format Document Processing: Supports PDF, TXT, JSON, and DOCX files.
Text Chunking & Embedding: Uses RecursiveCharacterTextSplitter for chunking and Ollama embeddings for vector representation.
FAISS Vector Search: Finds semantically similar documents.
BM25 Lexical Search: Matches keywords and phrases for better retrieval.
Hybrid Retrieval Strategy: Combines FAISS and BM25 scores for better accuracy.
LLM-Powered Answering: Uses Groq’s Llama3-8B to generate human-like responses.
JSON Output & Document Insights: Displays retrieved documents and structured responses.
🛠️ Installation
Clone the repository:

**git clone https://github.com/your-repo.git  
cd your-repo**

Create and activate a virtual environment:

**python -m venv venv  
source venv/bin/activate**  # On Windows: **venv\Scripts\activate**  

Install dependencies:

**pip install -r requirements.txt**  

Set up environment variables:

**cp .env.example .env**  # Rename and add your API keys 

📜 Usage
Run the Streamlit app:

**streamlit run app.py**

Upload documents (PDF, TXT, JSON, DOCX).
Click "Initialize Document Embeddings" to process files.
Enter a query to search within the uploaded documents.
The system will retrieve relevant content and provide an AI-generated response.
