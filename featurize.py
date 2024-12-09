#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pymongo
from transformers import AutoTokenizer, AutoModel
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import torch
from bson.objectid import ObjectId


# In[2]:


#Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# In[3]:


# MongoDB Configuration
MONGO_URI = "mongodb+srv://and8995:Aniks777@cluster0.4voet.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo_client = pymongo.MongoClient(f"{MONGO_URI}")
db = mongo_client["ros2_rag"]


# In[4]:


# Qdrant Configuration
QDRANT_HOST = "https://fc94b6ab-f5e6-4b45-8e7c-51ed48367a37.us-east4-0.gcp.cloud.qdrant.io:6333"  # Replace with your Qdrant Cloud endpoint
QDRANT_API_KEY = "j31ExXzFJ5xD77C1OFwnx21Aa5gFUSpSEQhCO4bHxIaGfSd6TNXRVQ"  # Replace with your Qdrant Cloud API Key
qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)


# In[5]:


# Model for Featurization
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


# In[6]:


# Collection name in Qdrant
QDRANT_COLLECTION_NAME = "star_charts"


# In[7]:


def featurize_text(text):
    """Convert text into a vector representation using a transformer model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings.squeeze()


# In[8]:


def ensure_qdrant_collection(collection_name, vector_size):
    """Ensure a Qdrant collection exists, or create it if it doesn't."""
    collections = qdrant_client.get_collections()
    collection_names = [collection.name for collection in collections.collections]
    
    if collection_name not in collection_names:
        logging.info(f"Creating Qdrant collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    else:
        logging.info(f"Collection '{collection_name}' already exists in Qdrant.")


# In[12]:


"""def push_data_to_qdrant(mongo_collection_name):
    Fetch data from MongoDB and push it to Qdrant.
    collection = db[mongo_collection_name]
    data = collection.find()

    for item in data:
        try:
            content = item.get("content", "")
            # Convert ObjectId to string in metadata
            metadata = {
                k: str(v) if isinstance(v, ObjectId) else v 
                for k, v in item.items() 
                if k != "content"
            }
            
            if not content.strip():
                logging.warning(f"Skipping item with missing content: {item}")
                continue
            
            # Featurize the text content
            vector = featurize_text(content)
            
            # Ensure Qdrant collection exists
            ensure_qdrant_collection(QDRANT_COLLECTION_NAME, vector_size=len(vector))
            
            # Insert data into Qdrant
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    {
                        "id": str(item["_id"]),  # Convert ObjectId to string
                        "vector": vector.tolist(),
                        "payload": metadata
                    }
                ]
            )
            logging.info(f"Inserted document ID {item['_id']} into Qdrant.")
        except Exception as e:
            logging.error(f"Failed to process item {item['_id']}: {e}")"""

def push_data_to_qdrant(mongo_collection_name):
    """Fetch data from MongoDB and push it to Qdrant."""
    collection = db[mongo_collection_name]
    data = collection.find()
    
    # Add a counter for generating numeric IDs
    point_id = 1

    for item in data:
        try:
            content = item.get("content", "")
            metadata = {
                k: str(v) if isinstance(v, ObjectId) else v 
                for k, v in item.items() 
                if k != "content"
            }
            
            if not content.strip():
                logging.warning(f"Skipping item with missing content: {item}")
                continue
            
            vector = featurize_text(content)
            ensure_qdrant_collection(QDRANT_COLLECTION_NAME, vector_size=len(vector))
            
            # Use the numeric ID instead of ObjectId string
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    {
                        "id": point_id,  # Use numeric ID
                        "vector": vector.tolist(),
                        "payload": {
                            "mongo_id": str(item["_id"]),  # Store original MongoDB ID in payload
                            **metadata
                        }
                    }
                ]
            )
            logging.info(f"Inserted document ID {item['_id']} into Qdrant with point_id {point_id}")
            point_id += 1  # Increment the counter
        except Exception as e:
            logging.error(f"Failed to process item {item['_id']}: {e}")


# In[13]:


if __name__ == "__main__":
    # Specify the MongoDB collection to fetch data from
    MONGO_COLLECTION_NAME = "raw_data"
    push_data_to_qdrant(MONGO_COLLECTION_NAME)


# In[ ]:




