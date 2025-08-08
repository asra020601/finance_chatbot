import streamlit as st
import pymupdf  
import re
import traceback
import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import torch
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(page_title="Financial Insights Chatbot", page_icon="📊", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
try:
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=GROQ_API_KEY)
    st.success("✅ LLM initialized successfully. Using llama3-70b-8192")
except Exception as e:
    st.error("❌ Failed to initialize Groq LLM.")
    traceback.print_exc()

embedding_model = SentenceTransformer("baconnier/Finance2_embedding_small_en-V1.5")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def fetch_financial_data(company_ticker):
    if not company_ticker:
        return "No ticker symbol provided. Please enter a valid company ticker."

    try:
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={company_ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        overview_response = requests.get(overview_url)

        if overview_response.status_code == 200:
            overview_data = overview_response.json()
            market_cap = overview_data.get("MarketCapitalization", "N/A")
        else:
            return "Error fetching company overview."

        income_url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={company_ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        income_response = requests.get(income_url)

        if income_response.status_code == 200:
            income_data = income_response.json()
            annual_reports = income_data.get("annualReports", [])
            revenue = annual_reports[0].get("totalRevenue", "N/A") if annual_reports else "N/A"
        else:
            return "Error fetching income statement."

        return f"Market Cap: ${market_cap}\nTotal Revenue: ${revenue}"

    except Exception as e:
        traceback.print_exc()
        return "Error fetching financial data."

def extract_and_embed_text(pdf_file):
    """Processes PDFs and generates embeddings with GPU acceleration using pymupdf."""
    try:
        docs, tokenized_texts = [], []

        with pymupdf.open(stream=pdf_file.read(), filetype="pdf") as doc:
            full_text = "\n".join(page.get_text("text") for page in doc)
            chunks = text_splitter.split_text(full_text)
            for chunk in chunks:
                docs.append(chunk)
                tokenized_texts.append(chunk.split())

        embeddings = embedding_model.encode(docs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

        embedding_dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(embedding_dim, 32)
        index.add(embeddings)

        bm25 = BM25Okapi(tokenized_texts)

        return docs, embeddings, index, bm25
    except Exception as e:
        traceback.print_exc()
        return [], [], None, None

def retrieve_relevant_docs(user_query, docs, index, bm25):
    """Hybrid search using FAISS cosine similarity & BM25 keyword retrieval."""
    query_embedding = embedding_model.encode(user_query, convert_to_numpy=True, normalize_embeddings=True)
    _, faiss_indices = index.search(np.array([query_embedding]), 8)
    bm25_scores = bm25.get_scores(user_query.split())
    bm25_indices = np.argsort(bm25_scores)[::-1][:8]
    combined_indices = list(set(faiss_indices[0]) | set(bm25_indices))

    return [docs[i] for i in combined_indices[:3]]

def generate_response(user_query, pdf_ticker, ai_ticker, mode, uploaded_file):
    try:
        if mode == "📄 PDF Upload Mode":
            docs, embeddings, index, bm25 = extract_and_embed_text(uploaded_file)
            if not docs:
                return "❌ Error extracting text from PDF."

            retrieved_docs = retrieve_relevant_docs(user_query, docs, index, bm25)
            context = "\n\n".join(retrieved_docs)
            prompt = f"Summarize the key financial insights for {pdf_ticker} from this document:\n\n{context}"

        elif mode == "🌍 Live Data Mode":
            financial_info = fetch_financial_data(ai_ticker)
            prompt = f"Analyze the financial status of {ai_ticker} based on:\n{financial_info}\n\nUser Query: {user_query}"
        else:
            return "Invalid mode selected."

        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        traceback.print_exc()
        return "Error generating response."

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📄 FinQuery RAG Chatbot</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h5 style='text-align: center; color: #666;'>Analyze financial reports or fetch live financial data effortlessly!</h5>", 
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏢 **Choose Your Analysis Mode**")
    mode = st.radio("", ["📄 PDF Upload Mode", "🌍 Live Data Mode"], horizontal=True)

with col2:
    st.markdown("### 🔎 **Enter Your Query**")
    user_query = st.text_input("💬 What financial insights are you looking for?")

st.markdown("---")
if mode == "📄 PDF Upload Mode":
    st.markdown("### 📂 Upload Your Financial Report")
    uploaded_file = st.file_uploader("🔼 Upload PDF (Only for PDF Mode)", type=["pdf"])
    pdf_ticker = st.text_input("🏢 Enter Company Ticker for PDF Insights", placeholder="e.g., INFY, TCS")
    ai_ticker = None
else:
    st.markdown("### 🌍 Live Market Data")
    ai_ticker = st.text_input("🏢 Enter Company Ticker for AI Insights", placeholder="e.g., AAPL, MSFT")
    uploaded_file = None
    pdf_ticker = None

if st.button("Analyze Now"):
    if mode == "📄 PDF Upload Mode" and (not uploaded_file or not pdf_ticker):
        st.error("❌ Please upload a PDF and enter a company ticker for insights.")
    elif mode == "🌍 Live Data Mode" and not ai_ticker:
        st.error("❌ Please enter a valid company ticker for AI insights.")
    else:
        with st.spinner("🔍 Your Query is Processing, this can take up to 5 - 7 minutes ⏳"):
            response = generate_response(user_query, pdf_ticker, ai_ticker, mode, uploaded_file)
            st.markdown("---")
            st.markdown("<h3 style='color: #4CAF50;'>💡 AI Response</h3>", unsafe_allow_html=True)
            st.write(response)

st.markdown("---")

