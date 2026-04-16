# brain_module.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize DB
client = chromadb.Client(Settings(persist_directory="./db"))
collection = client.get_or_create_collection("dormnet")

def add_documents(text_list):
    embeddings = model.encode(text_list).tolist()

    collection.add(
        documents=text_list,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(text_list))]
    )

def query(query_text):
    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    return results["documents"]

if __name__ == "__main__":
    docs = ["Derivative of x^2 is 2x", "Binary search is O(log n)"]
    add_documents(docs)

    print(query("What is derivative?"))