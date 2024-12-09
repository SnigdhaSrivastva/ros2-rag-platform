from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from pymongo import MongoClient
import datetime

MONGO_URI = "mongodb+srv://and8995:Aniks777@cluster0.zqzqz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
QDRANT_URI = "https://fc94b6ab-f5e6-4b45-8e7c-51ed48367a37.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "r7tCBpWwRoewwOOIP-VuZgRoka_YZGvwY24SzEK1K85CyU1-Ii772Q"
client = MongoClient(MONGO_URI)
db = client["ros2_rag"]
youtube_collection = db["youtube_data"]
docs_collection = db["documentation_data"]
featurized_collection = db["featurized_data"]
instruction_collection = db["instruction_data"]

qdrant_client = QdrantClient(url=QDRANT_URI, api_key=QDRANT_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

def featurize_data():
    # Get data from both collections
    youtube_data = list(youtube_collection.find())
    docs_data = list(docs_collection.find())
    
    processed_data = []
    
    # Process YouTube data with chunking
    for doc in youtube_data:
        chunks = chunk_text(doc["content"], chunk_size=512)
        for i, chunk in enumerate(chunks):
            processed_data.append({
                "id": f"{str(doc['_id'])}_chunk_{i}",
                "content": chunk,
                "url": doc["url"],
                "source": "youtube",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "processed_date": datetime.datetime.utcnow()
            })
    
    # Process documentation data with chunking
    for doc in docs_data:
        chunks = chunk_text(doc["content"], chunk_size=512)
        for i, chunk in enumerate(chunks):
            processed_data.append({
                "id": f"{str(doc['_id'])}_chunk_{i}",
                "content": chunk,
                "url": doc["url"],
                "source": "github",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "processed_date": datetime.datetime.utcnow()
            })
    
    # Store processed data in MongoDB
    if processed_data:
        featurized_collection.insert_many(processed_data)
    
    return processed_data

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end > text_len:
            end = text_len
        # Add chunk to list
        chunks.append(text[start:end])
        # Move start position, accounting for overlap
        start = end - overlap
    
    return chunks

def store_embeddings(data):
    # Generate embeddings for all chunks
    texts = [item["content"] for item in data]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    points = []
    for i, item in enumerate(data):
        points.append({
            "id": item["id"],
            "vector": embeddings[i].tolist(),
            "payload": {
                "content": item["content"],
                "url": item["url"],
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "total_chunks": item["total_chunks"],
                "processed_date": item["processed_date"].isoformat()
            }
        })
    
    # Store in Qdrant
    qdrant_client.upsert(collection_name="rag_collection", points=points)

def retrieve_and_create_instruction_dataset(query, top_k=5):
    # Encode the query
    query_vector = model.encode(query).tolist()
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name="rag_collection",
        query_vector=query_vector,
        limit=top_k
    )
    
    # Create instruction dataset
    instruction_data = []
    for result in search_results:
        instruction_entry = {
            "instruction": query,
            "input": result.payload["content"],
            "context": {
                "source": result.payload["source"],
                "url": result.payload["url"],
                "chunk_index": result.payload["chunk_index"],
                "total_chunks": result.payload["total_chunks"],
                "score": result.score,
                "processed_date": datetime.datetime.fromisoformat(result.payload["processed_date"])
            },
            "created_at": datetime.datetime.utcnow()
        }
        instruction_data.append(instruction_entry)
    
    # Store in MongoDB
    if instruction_data:
        instruction_collection.insert_many(instruction_data)
        
    return instruction_data
