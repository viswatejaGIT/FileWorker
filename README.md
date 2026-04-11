# Summary of project
This is a Retrieval Augmented Generation(RAG) based web application where you can upload a pdf(of certain pages, not a book..etc) and ask questions in it like you ask it to someone who read the pdf. you can write the question in chatbox and bot will answer the question and also provides you with the pages it referred to answer the question. try it :)

## features
- pdf upload
- chat with conversation memory
- Page source reference
- Thinking expander

## tech Stack
Streamlit, Google Gemini, LangChain, FAISS, HuggingFace Embeddings, PyPDF2

## How to run locally
- Clone the repo
- Install dependencies(in requirements file)
- Create .env with the API key
- Run 'streamlit run app.py'

## how it works
As soon as you upload a document, the document is stored locally in your system. then the content in document will be turned to embeddings and stored in FAISS database. when you enter a question, the question will also be converted to embeds and will be matched with the embeds present in the database, then it will take the top 4 best matched embeds to your question, considering this as a context to answer your question, this will be given to gemini llm with a detailed prompt. then LLM process it and produce the output which is shown to you. You can also see the top 4 embed text chosen to answer your question in "show thinking" drop down after the answer.

