# 🤖 Agentic RAG Chatbot for Research Papers & Current News

This project is an **Agentic Retrieval-Augmented Generation (RAG) Chatbot** built with **Streamlit**, **FAISS**, **LangChain**, **CrewAI**, and the **Groq API (LLaMA3-70B)**.  
It allows users to:
- **Query the top 100 research papers** (stored in a FAISS vector database)
- **Get up-to-date answers** from the web using **CrewAI agents** and **DuckDuckGo Search**

## 🚀 Features
- **Hybrid Query Handling**  
  - Academic queries → Answered using semantic search over research paper embeddings  
  - General news queries → Answered using real-time web search + LLM reasoning
- **FAISS Vector Database** for fast semantic search
- **CrewAI Agent** for web-based, real-time information retrieval
- **LLaMA3-70B (Groq API)** for high-quality LLM responses
- **Streamlit UI** for interactive chat experience

---

## 📂 Project Structure

├── app_2.py                   # Main Streamlit application
├── arxiv_papers/              # Folder containing 100 research papers from arXiv
├── top100_papers_vector_db/   # FAISS vector store with research paper embeddings
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

---

## 🔑 Prerequisites

1. **Python** 3.10+
2. **Groq API Key**  
   Sign up at [https://groq.com](https://groq.com) and get your API key.

3. **Top 100 Research Papers Vector DB**  
   You must already have the FAISS vector database stored in `top100_papers_vector_db/`  
   (Generated from embeddings using `sentence-transformers/all-MiniLM-L6-v2`)

---

## ⚙️ Installation

'''# 1️⃣ Clone the repository
git clone https://github.com/mhdafrith/-Agentic-RAG-Chatbot-for-Research-Papers-Live-News.git
cd -Agentic-RAG-Chatbot-for-Research-Papers-Live-News

# 2️⃣ Create & activate a virtual environment
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Set up environment variables
# Create a `.env` file in the project root and add:
GROQ_API_KEY=your_groq_api_key_here

# 5️⃣ Run the application
streamlit run app_2.py'''
