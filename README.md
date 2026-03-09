



# 🤖 Research Paper Q&A Bot
<img width="1915" height="896" alt="image" src="https://github.com/user-attachments/assets/70e8ad09-746e-471c-a045-75111d49e7e4" />
# Research Paper Q&A Bot


A RAG (Retrieval-Augmented Generation) chatbot that lets you ask questions about research papers and get answers with cited sources.

**Built by:** Prem Pochiraju  
**Tech stack:** Python · LangChain · ChromaDB · OpenAI  
**Purpose:** Portfolio AI project demonstrating LLM integration and data pipeline engineering



---

## What It Does

Upload your PDF research papers, ask questions in plain English, and get accurate answers with page-level citations pulled directly from the papers.

```
You: What method did the paper use to detect ambulances?

Bot: The paper uses YOLOv10, a real-time object detection model fine-tuned
     on a custom dataset of emergency vehicles. It achieves 94.3% precision
     at 30 FPS on standard traffic camera feeds.

Sources:
  - YOLOv10_Emergency_Detection.pdf (page 3)
  - YOLOv10_Emergency_Detection.pdf (page 5)
```

---

## How It Works

```
PDF Papers
    |
    v
Text Chunks (1000 tokens, 150 overlap)
    |
    v
OpenAI Embeddings  -->  ChromaDB (Vector Store)
                               |
User Question  -->  Embed  -->  Similarity Search (top 4 chunks)
                                       |
                          Chunks + Question  -->  GPT-3.5  -->  Answer + Sources
```

1. PDFs are split into overlapping chunks and embedded using OpenAI
2. Vectors are stored locally in ChromaDB — no cloud database needed
3. Each question is embedded and matched against the stored chunks
4. The top 4 matching chunks are sent to GPT-3.5 with the question to generate answers 

---

## Quickstart

**Requirements:** Python 3.11+, OpenAI API key

```bash
# Clone the repository
git clone https://github.com/prempochiraju/research-qa-bot
cd research-qa-bot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."          # Mac/Linux
$env:OPENAI_API_KEY="sk-..."            # Windows PowerShell

# Add your PDF papers to the papers/ folder, then run
python app.py
```

To run the web UI instead:

```bash
pip install streamlit
streamlit run streamlit_app.py
```

---

## Project Structure

```
research-qa-bot/
├── app.py                  # Terminal chatbot
├── streamlit_app.py        # Web UI (Streamlit)
├── requirements.txt        # Python dependencies
├── README.md
└── papers/                 # Add your PDF papers here
```

---

## Cost

| Action | Approx. Cost |
|---|---|
| Index 10 papers (~100 pages) | $0.02 |
| Per question | $0.001 |
| 100 questions | $0.12 |

Total to build and demo this project: under $1.00

---

## Papers Used

- Prem Swaroop Pochiraju et al., "YOLOv10 for Real-Time Emergency Vehicles Detection in Intelligent Traffic Systems," IEEE IC_ASET, 2025
- Prem Swaroop Pochiraju et al., "Development of a Machine Learning-Based Model for an Intelligent Ambulance Detection System using CNN," WAMS, 2025

---

## Skills Demonstrated

- RAG pipeline design — chunking strategy, embedding, semantic retrieval
- Vector database — ChromaDB for local semantic search
- LLM integration — OpenAI API orchestrated via LangChain
- Python engineering — modular, production-style code
- Data pipelines — document ingestion and transformation

---

## Author

**Prem Pochiraju**  
M.S. Computer Engineering, Florida Institute of Technology  
[LinkedIn](https://www.linkedin.com/in/prem-pochiraju/) · [GitHub](https://github.com/prempochiraju) · prempochiraju@gmail.com
