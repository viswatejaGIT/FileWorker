# PDF Chat - RAG Chatbot

A Streamlit-based chatbot that lets you upload PDFs and ask questions about them using Retrieval-Augmented Generation (RAG).

## Features

- **PDF Upload**: Upload any PDF document (up to 10MB)
- **RAG Pipeline**: Chunks documents, creates embeddings, and retrieves relevant context for answers
- **Conversational Memory**: Maintains chat history for natural follow-up questions
- **Customizable Responses**: 
  - Response Style: Precise / Balanced / Creative (controls temperature)
  - Detail Level: Short / Normal / Detailed
- **Source Citations**: "Show thinking" expander reveals which pages were used to answer

## Tech Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyPDF2
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **LLM**: NVIDIA NIM API (`nemotron-3.5-lightning-30b-a3b`)

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your NVIDIA API key:
   ```
   NVIDIA_API_KEY=your_key_here
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Evaluation

The project includes a RAGAS-based evaluation script (`evaluate.py`) with test cases (`test_questions.py`) designed to test:
- Factual accuracy
- Handling of contradicting information
- Similar number confusion
- Hallucination detection

> Note: `evaluate.py` requires RAGAS API migration to work with the installed version.

## Project Structure

```
├── app.py              # Main Streamlit application
├── evaluate.py         # RAGAS evaluation script
├── test_questions.py   # Test cases with ground truth
├── test_document.pdf   # Sample document for testing
├── requirements.txt    # Python dependencies
└── .env                # API keys (not committed)
```
