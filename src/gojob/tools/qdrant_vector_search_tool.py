from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import streamlit as st
import os
from dotenv import load_dotenv


class QdrantVectorSearchTool:
    def __init__(self):
        # Charger les variables d'environnement
        load_dotenv()
        
        # Récupérer les informations sensibles depuis les variables d'environnement
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Vérifier que les variables d'environnement sont définies
        if not qdrant_url or not qdrant_api_key:
            st.error("QDRANT_URL ou QDRANT_API_KEY non définis dans les variables d'environnement")
            return
            
        # Initialize the Qdrant client
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.limit = 5
        self.collection_name = "job_broker"  # Set the collection name to job_broker
        # Use a model that produces 3072-dimensional vectors
        self.model = SentenceTransformer("sentence-transformers/roberta-large-nli-stsb-mean-tokens")
        self.MAX_RESULTS = 5

    def search(self, query):
        # Get the embeddings for the query
        query_embedding = self.model.encode(query).tolist()  # Convert to Python list

        # Combine embeddings of multiple fields for a more comprehensive query vector
        combined_embedding = query_embedding
        fields = ["company_description", "profile", "skills", "tools", "experiences", "recruitment_process"]
        for field in fields:
            field_embedding = self.model.encode(field).tolist()
            combined_embedding = [sum(x) for x in zip(combined_embedding, field_embedding)]

        # Search for the top 8 similar vectors in the collection
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=combined_embedding,
            limit=self.limit,
        )
        # Extract required fields from the search results
        results = [
            {
                "poste": result.payload["metadata"].get("poste"),
                "company_name": result.payload["metadata"].get("company_name"),
                "local_address": result.payload["metadata"].get("local_address"),
                "contract_type": result.payload["metadata"].get("contract_type"),
                "benefits": result.payload["metadata"].get("benefits"),
                "skills": result.payload["metadata"].get("skills"),
                "tools": result.payload["metadata"].get("tools"),
                "remote": result.payload["metadata"].get("remote"),
                "url": result.payload["metadata"].get("url"),
                "company_description": result.payload["metadata"].get("company_description"),
                "profile": result.payload["metadata"].get("profile"),
                "recruitment_process": result.payload["metadata"].get("recruitment_process"),
                "experience_level": result.payload["metadata"].get("experience_level"),
                "score": result.score
            }
            for result in search_results
        ]
        return results