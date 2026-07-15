# 🛒 GreenMart Assistant — AI Supermarket Chatbot

GreenMart Assistant is an intelligent customer service chatbot designed for supermarket websites. It utilizes machine learning for customer intent detection, computes semantic product searches against the store catalog, and leverages the Gemini API to formulate accurate, conversational, and policy-compliant replies.

---

## 🌟 Features

- **Intent Classification:** Recognizes customer intent (e.g., product search, store hours, policies, promotions, order tracking) using a local scikit-learn classifier.
- **Semantic Product Search:** Searches over a 6,000+ product catalog using vector embeddings (`SentenceTransformers` `all-MiniLM-L6-v2`) and cosine similarity.
- **Generative AI Responses:** Passes matched context to Google Gemini for warm, human-like answers.
- **Unified Deployment:** Frontend files (`index.html`, `chat.html`) are served directly by the FastAPI backend to simplify hosting and avoid CORS issues.

---

## 🏗️ Architecture

```mermaid
graph TD
    Client[Web Browser Frontend] -->|HTTP POST /api/chat| API[FastAPI Backend]
    API -->|1. Predict Intent| Classifier[Intent Classifier]
    API -->|2. Semantic Search (if relevant)| VectorDB[Product Index]
    API -->|3. Generate Response| Gemini[Gemini API]
    Gemini -->|Reply Text| API
    API -->|JSON Response| Client
```

---

## 🚀 Local Development Setup

### 1. Prerequisites
- Python 3.10+
- A Gemini API Key (obtained from [Google AI Studio](https://aistudio.google.com/))

### 2. Installation

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/Fathima-Fazlina/SUPERMARKET_GREENMART_CHATBOT.git
cd SUPERMARKET_GREENMART_CHATBOT
```

Set up a virtual environment and activate it:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory based on the `.env.example` template:
```bash
cp .env.example .env
```
Open `.env` and fill in your actual key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 4. Running the Application
Start the FastAPI server using `uvicorn`:
```bash
uvicorn backend.main:app --reload --port 8000
```
Open your browser and navigate to:
- **Landing Page:** [http://localhost:8000/](http://localhost:8000/)
- **Full Chat UI:** [http://localhost:8000/chat.html](http://localhost:8000/chat.html)
- **API Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🔮 Model Training & Index Building

If you update the product list or add new intents, you can re-build the search index or retrain the classifier:

- **Build Product Index:**
  ```bash
  python build_index.py
  ```
  *(Reads `data/raw/products.csv` and embeds product data to `retrieval/product_index.pkl`)*

- **Train Intent Model:**
  ```bash
  python train_intent_model.py
  ```
  *(Trains the classifier on `classifier/labeled_intents.csv` and saves to `classifier/intent_model.pkl`)*

---

## ☁️ Vercel Deployment

This project is fully prepared for serverless deployment on Vercel.

### Critical Deployment Step
Because the semantic search features use the PyTorch and SentenceTransformers packages, the deployment exceeds Vercel's default 500MB serverless function limit.

To deploy successfully:
1. Link your repository to **Vercel**.
2. Go to your Vercel Project Settings -> **Environment Variables**.
3. Add the following environment variables:
   - **`GEMINI_API_KEY`**: Your Google Gemini API Key.
   - **`VERCEL_SUPPORT_LARGE_FUNCTIONS`**: `1` (This enables Vercel's Large Functions beta which raises the size limit to 5GB).
4. Deploy/Redeploy the project. Vercel will automatically build the FastAPI app using the configured `pyproject.toml` and `vercel.json` routing rules.
