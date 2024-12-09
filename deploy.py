from qdrant_client import QdrantClient
from model import generate_answer

QDRANT_URI = "https://fc94b6ab-f5e6-4b45-8e7c-51ed48367a37.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "r7tCBpWwRoewwOOIP-VuZgRoka_YZGvwY24SzEK1K85CyU1-Ii772Q"
qdrant_client = QdrantClient(url=QDRANT_URI, api_key=QDRANT_API_KEY)

def query_rag(question):
    # Retrieve relevant data from Qdrant
    search_results = qdrant_client.search(
        collection_name="rag_collection",
        query_vector=[0.5] * 384,  # Dummy vector for demo
        limit=5
    )
    context = " ".join([result.payload["content"] for result in search_results])
    # Generate an answer using the model
    return generate_answer(question, context)
