import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

st.title("pdf QandA with AI")

uploaded_file = st.sidebar.file_uploader("upload a PDF", type="pdf")

if uploaded_file:

    if "vector_store" not in st.session_state:
        # made pdf to one long string
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
        #convert that long string to chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        # now convert those chunks to vectors
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
     
        # store those vectors in FAISS database
        vector_store = FAISS.from_texts(chunks, embeddings)

        # store it in a session so that it wont reload everytime
        st.session_state["vector_store"] = vector_store
        
question = st.chat_input("ask a question about the pdf")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if question:
    # 1. Save and display user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 2. Search, build prompt, call Gemini
    results = st.session_state["vector_store"].similarity_search(question,k=4)
    context = ""
    for result in results:
        context += result.page_content + "\n"
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context.If the answer is not in the context, say "I don't have enough information."
    Context: {context} Question: {question}"""
    try:
        response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        # 3. Save and display assistant message
        st.session_state["messages"].append({"role": "assistant", "content": response.text})
        with st.chat_message("assistant"):
            st.write(response.text)
    except Exception as e:
        st.write(f"error: {e}")
else:
    st.write("please upload a file to start conversation")


        




        


    
    


