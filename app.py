import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from google import genai

#load the key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_and_chunk(uploaded_file):
    if uploaded_file.name != st.session_state.get("name"):
        if "vector_store" in st.session_state:
            del st.session_state["vector_store"]
        st.session_state["name"] = uploaded_file.name

    if "vector_store" not in st.session_state:
        # made pdf to one long string
        reader = PdfReader(uploaded_file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = []
        all_metadatas = []
        for i,page in enumerate(reader.pages):
            page_text = page.extract_text()
            page_chunks = splitter.split_text(page_text)
            for chunk in page_chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"page": i+1})
        if not all_chunks:
            st.warning("please input text")
            st.stop()
        # now convert those chunks to vectors
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # store those vectors in FAISS database
        vector_store = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadatas)
        # store it in a session so that it wont reload everytime
        st.session_state["vector_store"] = vector_store


def processing(question,style):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    message = "previous conversation:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"][-6:]])
    detail_map = {
            "Short": "Keep your answer to 1-2 sentences maximum",
            "Normal": "Give a clear, appropriately sized answer",
            "Detailed": "Give a thorough answer with examples and reasoning"
        }

    if "vector_store" in st.session_state:
        results = st.session_state["vector_store"].similarity_search(question, k=4)
        context = "\n---\n".join([r.page_content for r in results])          
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context.
        If the answer is not in the context, say "I don't have enough information" and dont say "Based on provided context", just answer the question in {detail_map[style["output_level"]]} style.

        Context:
        {context}

        chat History for smooth flow of conversation:
        {message}

        Question:
        {question}

        Answer:"""
    else:
        prompt = f"""You are a helpful assistant. just answer question in {detail_map[style["output_level"]]} style.

        chat History for smooth flow of conversation:
        {message}

        Question:
        {question}

        Answer:"""
        results = None
    


    try:
        stream = client.models.generate_content_stream(model="gemini-flash-latest", contents=prompt, config={"temperature": style["temperature"]})
        with st.chat_message("assistant"):
            full_response = ""
            placeholder = st.empty()
            for chunk in stream:
                full_response += chunk.text
                placeholder.markdown(full_response)
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
        
        if results:
            with st.expander("Show thinking"):
                for r in results:
                    st.write(f"Page {r.metadata['page']}:")
                    st.write(r.page_content)
                    st.write("---")
    except Exception as e:
        st.write(f"error: {e}")


if __name__ == "__main__":
    
    st.title("chat with AI")

    uploaded_file = st.sidebar.file_uploader("upload a PDF", type="pdf")
    
    tempe = st.sidebar.radio("Response Style", options=["Precise", "Balanced", "Creative"])
    temp_map = {"Precise": 0.2, "Balanced": 0.5, "Creative": 0.9}
    
    output_level = st.sidebar.radio("Detail level", options=["Short","Normal","Detailed"])
    style = {"temperature":temp_map[tempe], "output_level": output_level}
    
    if uploaded_file:
        extract_and_chunk(uploaded_file)
    
    question = st.chat_input("ask a question about the pdf")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if question:
        processing(question,style)
