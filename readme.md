RAG System for ROS2 Subdomains

Overview

This project implements a Retrieval-Augmented Generation (RAG) system to assist developers in ROS2 subdomains, such as:

ros2: Robotics middleware
nav2: Navigation
moveit2: Motion planning
gazebo: Simulation
The system combines retrieval-based and generation-based approaches to answer domain-specific questions using MongoDB Atlas, Qdrant, and T5-based fine-tuned models.

Features

ETL Pipeline: Extracts raw data from GitHub (ROS2 documentation) and YouTube, processes it, and loads it into MongoDB Atlas.
Featurization Pipeline: Converts raw data into vector embeddings using SentenceTransformers and stores them in Qdrant for fast retrieval.
Fine-tuned Model: Uses a T5-small model fine-tuned on instruction datasets for domain-specific question-answering.
Interactive API: A FastAPI backend to process queries and return relevant answers.
Gradio UI (Optional): An interactive web interface for testing the RAG system.
Architecture

The system consists of the following pipelines:

ETL Pipeline: Ingests and stores raw data.
Featurization Pipeline: Converts data into embeddings and stores them in Qdrant.
Inference Pipeline: Combines retrieval from Qdrant and T5 model generation to answer user queries.


Technologies Used

Python: Core programming language
FastAPI: Backend framework for APIs
MongoDB Atlas: Cloud-based NoSQL database
Qdrant Cloud: Vector search engine
SentenceTransformers: For embedding generation
Huggingface Transformers: For T5 fine-tuning
ClearML: Experiment tracking and orchestration
Gradio: Web interface for query testing
Setup Instructions

1. Prerequisites
Python 3.8 or higher
MongoDB Atlas account (Sign up here)
Qdrant Cloud account (Sign up here)
Docker (optional, for containerization)
2. Clone the Repository
git clone https://github.com/your-username/rag-project.git
cd rag-project
3. Install Dependencies
pip install -r app/requirements.txt
4. Configure Environment Variables
Create a .env file in the project root and add the following:

MONGO_URI=mongodb+srv://and8995:Aniks777@cluster.mongodb.net/<dbname>
QDRANT_URL=https://fc94b6ab-f5e6-4b45-8e7c-51ed48367a37.us-east4-0.gcp.cloud.qdrant.io:6333/dashboard#/collections
QDRANT_API_KEY=r7tCBpWwRoewwOOIP-VuZgRoka_YZGvwY24SzEK1K85CyU1-Ii772Q
5. Run the App
Run the FastAPI application locally:

python app/main.py
The app will be available at http://127.0.0.1:8000.

6. Test the API
ETL Endpoint: Extract and load data
curl -X POST http://127.0.0.1:8000/etl
Featurization Endpoint: Generate embeddings and store them in Qdrant
curl -X POST http://127.0.0.1:8000/featurize
Query Endpoint: Retrieve and generate answers
curl -X POST "http://127.0.0.1:8000/query?question=How+do+I+navigate"
Docker Deployment

Build and Run the Docker Containers
docker-compose up --build
Verify Running Services
docker ps
The app will be accessible at http://localhost:8000.

Gradio UI (Optional)

To run the Gradio interface for testing:

Add the following code to main.py:
import gradio as gr
from model import generate_answer

def gradio_interface(question):
    return generate_answer(question, "Sample ROS2 context")

iface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text")
iface.launch()
Start the app:
python app/main.py
Open the Gradio interface URL provided in the terminal.
Future Improvements

Fine-tuning: Enhance the T5 model with additional domain-specific instruction datasets.
Interactive UI: Add more interactivity and predefined questions for ease of use.
Scalability: Expand the system to handle larger datasets and more complex queries.
Contributors

Member 1 – Aryan Donde
Member 2 – Dhruv Pando 
