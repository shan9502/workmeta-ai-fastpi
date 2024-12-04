from fastapi import FastAPI
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_vertexai import VertexAIEmbeddings
# import vertexai

# PROJECT_ID = "workmeta-ai"  # @param {type:"string"}
# LOCATION = "us-central1"
# vertexai.init(project=PROJECT_ID, location=LOCATION)

class Body(BaseModel):
    query: str

app = FastAPI()

# Initialize the a specific Embeddings Model version
# embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@app.post("/embedd_qurey/")
def read_root(body:Body):
    single_vector = embeddings.embed_query(str(body.query))
    return {"embeddings":single_vector}
