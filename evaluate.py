from app import extract_and_chunk
from test_questions import test_cases
from dotenv import load_dotenv
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
import os
from ragas.llms import llm_factory
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# initializing llm for evaluation
openai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)
llm = llm_factory("nvidia/nemotron-3.5-lightning-30b-a3b", client=openai_client)

# test_document file is extracted,chuncked and stored in vector_store variable
with open("test_document.pdf", "rb") as f:
    vector_store = extract_and_chunk(f)

data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# generate the response for the question
def generate(prompt, temperature=0.5):
    response = openai_client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

# retrieve k most similar chunks to the question in vector store
def retrieve(vector_store, question, k=4):
    return vector_store.similarity_search_with_score(question, k=k)

# ragas need question, ground_truth, llm answer and content to evaluate, so storing all in a data dictionary
for i in test_cases:
    results = retrieve(vector_store,i["question"])
    context = "\n---\n".join([r[0].page_content for r in results])
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context.
        If the answer is not in the context, say "I don't have enough information" and dont say "Based on provided context", just answer the question.
        Context:
        {context}

        Answer:"""
    rag_output = generate(prompt)

    data["question"].append(i["question"]) # question    
    data["ground_truth"].append(i["ground_truth"]) # actual answer
    data["answer"].append(rag_output) # llm's answer
    data["contexts"].append([r[0].page_content for r in results]) # content llm used to answer

# converting data dictionary to dataset as ragas need dataset
dataset = Dataset.from_dict(data)
print(dataset)

# evaluating using ragas
os.environ["RAGAS_MAX_WORKERS"] = "1"
faithfulness.llm = llm
answer_relevancy.llm = llm
answer_relevancy.embeddings = embeddings
print(evaluate(dataset, metrics=[faithfulness, answer_relevancy]))