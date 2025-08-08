import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse
import uvicorn
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
from dotenv import load_dotenv 
load_dotenv()  # Load environment variables from .env

app = FastAPI()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=GROQ_API_KEY)
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
async def extract_and_embed_text(pdf_file):
    """Processes PDFs and generates embeddings with GPU acceleration using pymupdf."""
    try:
        docs, tokenized_texts = [], []
        pdf_content = await pdf_file.read()  # Await the async read operation
        with pymupdf.open(stream=pdf_content, filetype="pdf") as doc:
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

async def generate_response(user_query, pdf_ticker, ai_ticker, mode, uploaded_file):
    try:
        if mode == "📄 PDF Upload Mode":
            docs, embeddings, index, bm25 = await extract_and_embed_text(uploaded_file)
            if not docs:
                return "❌ Error extracting text from PDF."

            retrieved_docs = retrieve_relevant_docs(user_query, docs, index, bm25)
            context = "\n\n".join(retrieved_docs)
            prompt = f"Summarize the key financial insights for {pdf_ticker} from this document:\n\n{context}"

        elif mode == "🌍 Live Data Mode":
            financial_info = fetch_financial_data(ai_ticker)
            prompt = f"Analyze the financial status of {ai_ticker} based on:\n{financial_info}\n\nUser Query: {user_query}"
        else:
            return "it is not working"

        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        traceback.print_exc()
        return "Error generating response."

# HTML template as a Python function returning a formatted string.
def render_page(error: str = "", response: str = "", form_data: dict = {}):
    # Use default empty strings for form values if not provided.
    user_query = form_data.get("user_query", "")
    mode = form_data.get("mode", "PDF")
    pdf_ticker = form_data.get("pdf_ticker", "")
    ai_ticker = form_data.get("ai_ticker", "")
    
    # The HTML template.
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>📄 FinQuery RAG Chatbot</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .center {{ text-align: center; }}
            .green {{ color: #4CAF50; }}
            .gray {{ color: #666; }}
            .container {{ display: flex; justify-content: space-between; }}
            .column {{ flex: 1; padding: 10px; }}
            .error {{ color: red; margin-top: 10px; }}
            hr {{ margin: 20px 0; }}
            label {{ font-weight: bold; }}
            input[type="text"], input[type="file"] {{ width: 100%; padding: 8px; margin: 5px 0; }}
        </style>
    </head>
    <body>
        <h1 class="center green">📄 FinQuery RAG Chatbot</h1>
        <h5 class="center gray">Analyze financial reports or fetch live financial data effortlessly!</h5>
        
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <div class="container">
                <div class="column">
                    <h3>🏢 Choose Your Analysis Mode</h3>
                    <label>
                        <input type="radio" name="mode" value="📄 PDF Upload Mode" {"checked" if mode=="📄 PDF Upload Mode" else ""}> 📄 PDF Upload Mode
                    </label>
                    <label style="margin-left: 15px;">
                        <input type="radio" name="mode" value="🌍 Live Data Mode" {"checked" if mode=="🌍 Live Data Mode" else ""}> 🌍 Live Data Mode
                    </label>
                </div>
                <div class="column">
                    <h3>🔎 Enter Your Query</h3>
                    <input type="text" name="user_query" placeholder="💬 What financial insights are you looking for?" value="{user_query}" required>
                </div>
            </div>
            <hr>
            <div id="pdf-section" style="display: {"block" if mode=="📄 PDF Upload Mode" else "none"};">
                <h3>📂 Upload Your Financial Report</h3>
                <input type="file" name="pdf_upload" accept=".pdf">
                <input type="text" name="pdf_ticker" placeholder="🏢 Enter Company Ticker for PDF Insights (e.g., INFY, TCS)" value="{pdf_ticker}">
            </div>
            <div id="live-section" style="display: {"block" if mode=="🌍 Live Data Mode" else "none"};">
                <h3>🌍 Live Market Data</h3>
                <input type="text" name="ai_ticker" placeholder="🏢 Enter Company Ticker for AI Insights (e.g., AAPL, MSFT)" value="{ai_ticker}">
            </div>
            <hr>
            <div class="center">
                <button type="submit">Analyze Now</button>
            </div>
        </form>
        
        {"<p class='error center'>" + error + "</p>" if error else ""}
        {f"<hr><h3 class='green center'>💡 AI Response</h3><p>{response}</p><hr>" if response else ""}
        
        <script>
            // Toggle between PDF and Live sections based on the selected mode.
            const radios = document.getElementsByName('mode');
            const pdfSection = document.getElementById('pdf-section');
            const liveSection = document.getElementById('live-section');
            radios.forEach(radio => {{
                radio.addEventListener('change', () => {{
                    if (radio.value === "PDF" && radio.checked) {{
                        pdfSection.style.display = "block";
                        liveSection.style.display = "none";
                    }} else if (radio.value === "Live" && radio.checked) {{
                        pdfSection.style.display = "none";
                        liveSection.style.display = "block";
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/", response_class=HTMLResponse)
async def read_form():
    return HTMLResponse(content=render_page(), status_code=200)

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    mode: str = Form(...),
    user_query: str = Form(...),
    pdf_ticker: str = Form(None),
    ai_ticker: str = Form(None),
    pdf_upload: UploadFile = File(None)
):
    form_data = {"mode": mode, "user_query": user_query, "pdf_ticker": pdf_ticker, "ai_ticker": ai_ticker}
    error = ""
    
    if mode == "📄 PDF Upload Mode":
        if not pdf_upload or not pdf_ticker:
            error = "❌ Please upload a PDF and enter a company ticker for insights."
            return HTMLResponse(content=render_page(error=error, form_data=form_data), status_code=400)
    else:
        if not ai_ticker:
            error = "❌ Please enter a valid company ticker for AI insights."
            return HTMLResponse(content=render_page(error=error, form_data=form_data), status_code=400)
    
    # Simulate processing (replace with your actual function)
    response_text = await generate_response(user_query, pdf_ticker, ai_ticker, mode, pdf_upload)
    return HTMLResponse(content=render_page(response=response_text, form_data=form_data), status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)