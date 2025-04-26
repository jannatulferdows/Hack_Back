# app.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import pickle
import re
import time
from better_profanity import profanity
from sentence_transformers import SentenceTransformer
from together import Together
from config import TOGETHER_API_KEY

# --- Load resources at startup ---

# Load multilingual BERT model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Load FAISS index
index = faiss.read_index("wdc_product_faiss.index")

# Load product metadata
with open("wdc_product_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Initialize Together client
client = Together(api_key=TOGETHER_API_KEY)

# Load profanity filter
profanity.load_censor_words()

# --- Define FastAPI app ---
app = FastAPI()

# --- Utility functions ---
def sanitize_input(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def together_translate_and_guard(text):
    prompt = f"""
You are a professional translator for an e-commerce platform.

Instructions:
- If the input contains offensive or abusive language, respond exactly: Bad language detected
- If the input is already in proper English, return it unchanged
- If the input contains Bangla or Banglish, translate it into fluent English
- Prefer using "dress" instead of "shirt" for the Bangla word "jama"
- Return only the translated sentence, no extra text

Input: "{text}"
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def preprocess_and_translate_only(raw_text):
    text = sanitize_input(raw_text)
    if profanity.contains_profanity(text):
        return "Bad language detected"
    translated = together_translate_and_guard(text)
    return translated

def search_products(query, k=5):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")
    D, I = index.search(query_vector, k)

    results = []
    for idx in I[0]:
        results.append(metadata[idx])
    return results

def filter_relevant_products(query, results):
    product_list_text = "\n".join(
        [f"{i+1}. {r['title']} (Price: {r['price']})" for i, r in enumerate(results)]
    )
    prompt = f"""
You are a smart product assistant. Your job is to decide which products from a list are relevant to a user's shopping intent or similar product.

User intent: "{query}"

Product list:
{product_list_text}

Return only the product numbers of the relevant ones (e.g. 1, 3, 5).
Do not explain. Do not output anything else.
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()
    relevant_indexes = {int(i) - 1 for i in answer.split(",") if i.strip().isdigit()}
    return [results[i] for i in relevant_indexes if i < len(results)]

# --- API Endpoints ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search/")
def search_endpoint(request: QueryRequest):
    translated_query = preprocess_and_translate_only(request.query)
    if translated_query == "Bad language detected":
        return {"error": "Bad language detected"}
    results = search_products(translated_query, k=request.top_k)
    filtered = filter_relevant_products(translated_query, results)
    return {"query": translated_query, "results": filtered}

@app.get("/")
def root():
    return {"message": "Welcome to the Product Search API"}
