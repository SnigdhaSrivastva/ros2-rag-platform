import requests
from pymongo import MongoClient
from clearml import Task
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re

MONGO_URI = "mongodb+srv://and8995:Aniks777@cluster0.zqzqz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["ros2_rag"]
youtube_collection = db["youtube_data"]
docs_collection = db["documentation_data"]

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_github_docs_content():
    # Focus on LTS releases documentation
    lts_releases = ["humble", "iron"]  # Current ROS2 LTS releases
    docs_content = []
    
    for release in lts_releases:
        api_url = f"https://api.github.com/repos/ros2/ros2_documentation/contents/source/{release}"
        response = requests.get(api_url).json()
        if isinstance(response, list):
            for doc in response:
                if doc["type"] == "file" and doc["name"].endswith(('.rst', '.md')):
                    # Get raw content
                    raw_content = requests.get(doc["download_url"]).text
                    docs_content.append({
                        "name": doc["name"],
                        "url": doc["html_url"],
                        "content": raw_content,
                        "source": "github",
                        "release": release
                    })
    
    return docs_content

def process_transcript_with_llm(transcript_text):
    # Split long transcript into chunks of 1000 characters with overlap
    chunk_size = 1000
    overlap = 100
    chunks = []
    
    for i in range(0, len(transcript_text), chunk_size - overlap):
        chunk = transcript_text[i:i + chunk_size]
        chunks.append(chunk)
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    
    # Combine summaries
    return " ".join(summaries)

def get_youtube_content():
    # Predefined list of ROS2 tutorial videos
    ros2_video_urls = [
        "https://www.youtube.com/watch?v=0aPbWsyENA8",  # ROS2 Tutorial for Beginners
        "https://www.youtube.com/watch?v=bFWKYZR5Y28",  # ROS2 Navigation Tutorial
        "https://www.youtube.com/watch?v=T4iRJqESQAk",  # ROS2 Manipulation Basics
        "https://www.youtube.com/watch?v=QR1v-4_HDBk",  # ROS2 Perception Pipeline
        "https://www.youtube.com/watch?v=Gg25GfA456o",  # ROS2 SLAM Tutorial
    ]
    
    videos_content = []
    
    for url in ros2_video_urls:
        try:
            # Extract video ID from URL
            video_id = url.split("v=")[1]
            
            # Get video transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            # Combine transcript text
            full_transcript = " ".join([entry["text"] for entry in transcript])
            
            # Process transcript with LLM
            processed_content = process_transcript_with_llm(full_transcript)
            
            videos_content.append({
                "name": f"ROS2 Tutorial Video {video_id}",
                "url": url,
                "content": processed_content,
                "raw_transcript": full_transcript,
                "source": "youtube"
            })
        except Exception as e:
            continue  # Skip videos without transcripts
    
    return videos_content

def run_etl():
    # Initialize ClearML task
    task = Task.init(project_name="ros2_rag", task_name="etl_pipeline")
    
    try:
        # Get content from both sources
        github_data = get_github_docs_content()
        task.get_logger().report_text("GitHub docs content fetched successfully")
        
        youtube_data = get_youtube_content()
        task.get_logger().report_text("YouTube transcripts processed with LLM successfully")
        
        # Store in separate MongoDB collections
        if github_data:
            docs_collection.insert_many(github_data)
            task.get_logger().report_text(f"Stored {len(github_data)} documents in documentation collection")
            
        if youtube_data:
            youtube_collection.insert_many(youtube_data)
            task.get_logger().report_text(f"Stored {len(youtube_data)} videos in YouTube collection")
        
        # Log URLs for verification
        all_urls = [doc["url"] for doc in github_data + youtube_data]
        task.get_logger().report_text("\n".join(all_urls))
        
        return all_urls
        
    except Exception as e:
        task.get_logger().report_text(f"Error in ETL pipeline: {str(e)}")
        raise e