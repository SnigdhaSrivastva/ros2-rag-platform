#!/usr/bin/env python
# coding: utf-8

# <h2> Import all relevant libraries</h2>

# In[1]:


import re
import requests
import pymongo
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi
from clearml import Task
import logging
import urllib.parse


# <h2> Configure logging </h2>

# In[2]:


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# <h1>Featurization</h1>

# In[3]:


db_string="mongodb+srv://and8995:Uc3vk8VXHoXuKgvv@cluster0.4voet.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


# In[4]:


def featurize_data():
    client = pymongo.MongoClient(f"{db_string}")
    db = client["ros2_rag"]
    raw_data = db["raw_data"].find()

    qdrant_client = QdrantClient(host="qdrant")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    for data in raw_data:
        text = data["url"]
        tokens = tokenizer(text, return_tensors="pt")
        vector = model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()

        qdrant_client.upload_collection(
            collection_name="rag_vectors",
            vectors=vector,
            payload={"url": data["url"]}
        )
    print("Featurization Completed!")


# <h2>ClearML Setup</h2>

# In[5]:


# !clearml-init
task = Task.init(project_name="RAG_ETL_Pipeline", task_name="ETL Pipeline")


# <h2>MongoDB Setup</h2>

# In[6]:


client = pymongo.MongoClient(f"{db_string}")
db = client["ros2_rag"]
raw_data_collection = db["raw_data"]


# <h2>Step - 1: Ingest Data</h2>

# In[15]:


def extract_and_validate_urls(text):
    """
    Extracts URLs from a text string, validates them, and returns a list of valid URLs.
    """
    url_pattern = r"(https?://[^\s>`]*)"  # Updated pattern to exclude '>' and other markup
    urls = re.findall(url_pattern, text)
    valid_urls = []
    
    for url in urls:
        # Clean the URL by removing markup syntax
        url = re.sub(r'[>`_\)].*$', '', url)  # Remove '>`__' and similar markup
        url = url.rstrip('.')  # Remove trailing periods
        
        # Skip empty URLs after cleaning
        if not url:
            continue
            
        try:
            # Perform a HEAD request to check the URL's validity
            response = requests.head(url, timeout=5)
            response.raise_for_status()
            valid_urls.append(url)
        except requests.exceptions.RequestException as e:
            print(f"Invalid or inaccessible URL: {url}, Error: {e}")

    return valid_urls


# In[7]:


def extract_and_validate_urls(text):
    """
    Extracts URLs from a text string, validates them, and returns a list of valid URLs.
    """
    url_pattern = r"(https?://\S+)"  # Regular expression to match URLs
    urls = re.findall(url_pattern, text)
    valid_urls = []
    for url in urls:
        url = url.strip('>')
        try:
            response = requests.head(url, timeout=5)  # Use HEAD request for efficiency
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            valid_urls.append(url)
        except requests.exceptions.RequestException as e:
            print(f"Invalid or inaccessible URL: {url}, Error: {e}")  # Log invalid URLs
    return valid_urls


# In[16]:


def fetch_ros2_docs(repo_url):
    """Fetches ROS2 documentation URLs from a GitHub repository."""
    try:
        response = requests.get(repo_url, timeout=10)  # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text.splitlines()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch ROS2 documentation: {e}")
        return []


# In[17]:


def fetch_youtube_transcript(video_id):
    """Fetches YouTube video transcript by video ID."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except YouTubeTranscriptApi.CouldNotRetrieveTranscriptException as e:
        logging.error(f"Could not retrieve transcript for {video_id}: {e}")
        return ""
    except Exception as e:
        logging.exception(f"An unexpected error occurred while fetching transcript for {video_id}: {e}")
        return ""


# <h2> Step 2: Transform Data</h2>

# In[18]:


def clean_text(text):
    """Cleans and tokenizes raw text."""
    text = text.replace("\n", " ").strip()
    return text


# In[19]:


def structure_data(source, content, metadata=None):
    """Structures data for MongoDB insertion."""
    return {
        "source": source,
        "content": content,
        "metadata": metadata or {}
    }


# <h2> Step 3: Load Data</h2>

# In[20]:


# Step 3: Load Data
def load_to_mongodb(data):
    """Loads structured data into MongoDB."""
    try:
        if data:
            result = raw_data_collection.insert_one(data)
            logging.info(f"Data inserted successfully with ID: {result.inserted_id}")
        else:
            logging.warning("No data to insert")
    except pymongo.errors.PyMongoError as e:
        logging.error(f"Failed to insert data into MongoDB: {e}")


# <h1> Main ETL Function</h1>

# In[21]:


def etl_pipeline():
    # Ingest GitHub Docs
    ros2_docs_urls = fetch_ros2_docs("https://raw.githubusercontent.com/kartikj69/Finetuned-RAG-Systems-Engineering/refs/heads/main/ros.txt")
    ros2_docs_urls_string = "\n".join(ros2_docs_urls)  
    valid_urls = extract_and_validate_urls(ros2_docs_urls_string)
    for url in valid_urls:
        try:
            content = requests.get(url, timeout=10) #Added timeout for robustness
            content.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            content = content.text #Get the text content of the response
            cleaned_content = clean_text(content)
            structured_doc = structure_data(source=url, content=cleaned_content, metadata={"type": "ROS2 Documentation"})
            load_to_mongodb(structured_doc)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch or process URL {url}: {e}")
        except Exception as e:
            logging.exception(f"An unexpected error occurred while processing URL {url}: {e}")

    # Ingest YouTube Transcripts
    youtube_video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0"]  # Example video IDs
    for video_id in youtube_video_ids:
        transcript = fetch_youtube_transcript(video_id)
        cleaned_transcript = clean_text(transcript)
        structured_transcript = structure_data(source=f"YouTube Video ID: {video_id}", content=cleaned_transcript, metadata={"type": "YouTube Transcript"})
        load_to_mongodb(structured_transcript)


# In[22]:


if __name__ == "__main__":
    etl_pipeline()


# In[ ]:





# In[ ]:




