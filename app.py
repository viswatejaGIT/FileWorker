import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def extract_and_chunk(uploaded_file):
        """ this function reads the uploaded_file, split into chunks, convert them to vectors 
        and store it in vector store and return that vector store"""
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
            return None
        # now convert those chunks to vectors
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # store those vectors in FAISS database
        vector_store = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadatas)
        return vector_store

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    st.warning("API key issue")
    st.stop()
else:
    nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
    )


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
        results = st.session_state["vector_store"].similarity_search_with_score(question, k=4)
        context = "\n---\n".join([r[0].page_content for r in results])          
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
        stream = nim_client.chat.completions.create(model="nvidia/nemotron-3.5-lightning-30b-a3b", messages=[{"role": "user", "content": prompt}], temperature = style["temperature"], stream = True)
        with st.chat_message("assistant"):
            full_response = ""
            placeholder = st.empty()
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    placeholder.markdown(full_response)
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
        
        if results:
            with st.expander("Show thinking"):
                for r in results:
                    st.write(f"Page {r[0].metadata['page']}:")
                    st.write(r[0].page_content)
                    st.write("---")
                    st.write(f"score{r[1]}")
    except Exception as e:
        st.warning(f"could not process your question: {e}")

        


if __name__ == "__main__":
    
    st.title("chat with AI")

    uploaded_file = st.sidebar.file_uploader("upload a PDF", type="pdf")
    if uploaded_file:
        if  uploaded_file.size > 10 * 1024 * 1024:
            st.warning("please upload a file which is less then 10mb")
            st.stop()
    
    tempe = st.sidebar.radio("Response Style", options=["Precise", "Balanced", "Creative"])
    temp_map = {"Precise": 0.2, "Balanced": 0.5, "Creative": 0.9}
    
    output_level = st.sidebar.radio("Detail level", options=["Short","Normal","Detailed"])
    style = {"temperature":temp_map[tempe], "output_level": output_level}
    
    if uploaded_file:
        if uploaded_file.name !=st.session_state.get("name"):
            if "vector_store" in st.session_state:
                del st.session_state["vector_store"]
            st.session_state["name"] = uploaded_file.name
        
        if "vector_store" not in st.session_state:
            vector_store = extract_and_chunk(uploaded_file)
            if vector_store is None:
                st.warning("please input text")
                st.stop()
            st.session_state["vector_store"] = vector_store
    
    question = st.chat_input("ask a question about the pdf")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if question:
        processing(question,style)