from dotenv import load_dotenv
import os
import json
import numpy as np
from google.oauth2 import service_account
from llama_index.core import SQLDatabase, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load environment variables
env_path = r"C:\Users\HP\Videos\AI Projects\Chatbot LLama\.env"
load_dotenv(dotenv_path=env_path)

# Load BigQuery credentials
credentials_path = os.getenv("bq_service_accout_key")
project_id = os.getenv("bq_project_id")
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Set up SQLAlchemy engine for BigQuery
engine = create_engine(
    f"bigquery://{project_id}",
    credentials_path=credentials_path
)

# Define table
tables = {
    "tbproddb.Exam_Generator": {
        "description": "Exam Generator table that stores exam info",
        "columns": [
            "exam_id", "user_id", "type", "school_name", "sector", "Grade",
            "Subject", "Checkpoint_Name", "total_questions",
            "exercise_questions", "conceptual_questions"
        ]
    }
}

# Load LLM & embedding
llm = Groq(
    model="llama3-70b-8192",
    system_prompt="You are an SQL expert that can understand data..."  # same as before
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

# SQL Query Engine
sql_database = SQLDatabase(engine, include_tables=tables, sample_rows_in_table_info=50)
query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=tables, llm=llm)

# Chat Memory File
MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

# --- Vector DB Setup with Chroma ---
chroma_client = chromadb.Client()
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
chroma_collection = chroma_client.get_or_create_collection(
    name="chat_history",
    embedding_function=embedding_fn
)

def add_to_vector_store(query, response):
    chroma_collection.add(
        documents=[query],
        metadatas=[{"response": response}],
        ids=[str(hash(query))]
    )

def query_vector_store(prompt, threshold=0.85):
    results = chroma_collection.query(query_texts=[prompt], n_results=1)
    if results and results['documents'][0]:
        return results['metadatas'][0][0]['response']
    return None

# Detect corrections
def is_correction(text):
    return any(kw in text.lower() for kw in ["wrong", "not correct", "should be", "actually"])

def ask_query(query, query_engine, memory):
    print("\nThinking...\n")

    if any(greet in query.lower() for greet in ["hi", "hello", "salam"]):
        response = "Hello! How can I assist you today with your data-related queries?"
        memory.append({"user": query, "bot": response})
        save_memory(memory)
        return {"response": response, "sql": None}

    if not any(kw in query.lower() for kw in ["exam", "assessment", "grade", "checkpoint", "subject"]):
        fallback = "Your question does not match the exam data. Please ask a relevant question."
        memory.append({"user": query, "bot": fallback})
        save_memory(memory)
        return {"response": fallback, "sql": None}

    # Correction handling
    if memory and is_correction(query):
        last_entry = memory[-1]
        revised_prompt = f"Earlier: '{last_entry['user']}', Bot: '{last_entry['bot']}', Correction: '{query}'"
        corrected = llm.complete(revised_prompt).text.strip()
        memory[-1]["bot"] = corrected
        save_memory(memory)
        return {"response": corrected, "sql": None}

    # Vector DB search
    similar_response = query_vector_store(query)
    if similar_response:
        memory.append({"user": query, "bot": similar_response})
        save_memory(memory)
        return {"response": similar_response, "sql": None}

    # Run actual SQL engine query
    try:
        response = query_engine.query(query)
        sql_result = response.metadata.get("sql_query")
        memory.append({"user": query, "bot": response.response})
        add_to_vector_store(query, response.response)
        save_memory(memory)
        return {"response": response.response, "sql": sql_result}
    except Exception as e:
        fallback = "There was an error processing your query. Please rephrase or try again."
        return {"response": fallback, "sql": None}

# CLI
if __name__ == "__main__":
    print("🧠 Welcome to Vector-Enhanced SQL Chatbot (LLaMA + BigQuery + ChromaDB)")
    memory = load_memory()
    while True:
        user_input = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break
        output = ask_query(user_input, query_engine, memory)
        print(f"\n🔍 Answer: {output['response']}")
        if output['sql']:
            print(f"\n🧾 SQL Query: {output['sql']}")
