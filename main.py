from fastapi import FastAPI, Query
from etl import run_etl
from featurization import featurize_data, store_embeddings
from deploy import query_rag

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG System is running"}

@app.post("/etl")
def etl_pipeline():
    urls = run_etl()
    return {"message": "ETL completed", "urls": urls}

@app.post("/featurize")
def featurize_pipeline():
    data = featurize_data()
    store_embeddings(data)
    return {"message": "Featurization completed"}

@app.post("/query")
def query(question: str = Query(...)):
    response = query_rag(question)
    return {"response": response}