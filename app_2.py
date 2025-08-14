from crewai import Agent, Task, Crew
from crewai.llm import LLM
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Constants
VECTOR_DB_PATH = "top100_papers_vector_db"
TOP_K = 7

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": TOP_K})

# Initialize Groq LLM instance
llm = LLM(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Create a CrewAI agent for news analysis
news_agent = Agent(
    role="News Analyst",
    goal="Provide up-to-date information from the web about current events",
    backstory="Expert in current affairs, uses tools to gather the latest info.",
    allow_delegation=False,
    llm=llm
)

# DuckDuckGo web search function to get text snippets
def web_search(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        # Join snippets, fallback if 'body' missing
        return "\n\n".join([r.get("body", "") for r in results])

# Run Crew with web search results to generate answer
def run_news_crew(query):
    search_results = web_search(query)
    task_description = f"""Here is the latest info I found on the web:\n\n{search_results}\n\n
Based on this, answer the user's question: {query}"""

    task = Task(
        description=task_description,
        expected_output="An up-to-date and accurate answer from recent web results.",
        agent=news_agent
    )
    crew = Crew(
        agents=[news_agent],
        tasks=[task],
        verbose=False
    )
    result = crew.kickoff()

    # Extract answer text from task output safely
    try:
        if result and hasattr(result, "tasks_output") and len(result.tasks_output) > 0:
            answer = result.tasks_output[0].description
            return answer.strip()
        else:
            return "Sorry, no answer could be generated from web search."
    except Exception as e:
        return f"Error processing the response: {e}"

# Helper to detect general queries to use web search + news agent
def is_general_query(query: str) -> bool:
    general_keywords = ["news", "today", "recent", "latest", "IPL", "weather", "score", "happening"]
    return any(word.lower() in query.lower() for word in general_keywords)

# Streamlit UI setup
st.set_page_config(page_title="RAG Chatbot with Groq + CrewAI", layout="wide")
st.title("🤖 RAG Chatbot for Research Papers and Current News")

user_query = st.text_input("Ask a question about the top 100 research papers or general news:")

if user_query:
    with st.spinner("Thinking..."):
        if is_general_query(user_query):
            st.info("🌐 Using Web Search via CrewAI for this general query.")
            answer = run_news_crew(user_query)
            st.markdown("### 📌 Answer")
            st.write(answer)
        else:
            # Retrieve relevant docs from vector DB
            retrieved_docs = retriever.get_relevant_documents(user_query)
            relevant_docs = [doc for doc in retrieved_docs if len(doc.page_content.strip()) > 100]

            if not relevant_docs:
                st.warning("❌ No relevant answer found in the research papers.")
            else:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                prompt = f"""You are a research assistant. Use only the context below to answer the question.
Context:
{context}

Question: {user_query}
Answer:"""

                # Call Groq LLM with prompt
                try:
                    response =llm.call(prompt)

                    st.markdown("### 📌 Answer")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error from LLM: {e}")
