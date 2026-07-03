import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()   

def get_pgvector_store(client: str, embeddings: OpenAIEmbeddings = None) -> PGVector:
 
    connection = (
        f"postgresql+psycopg://{os.getenv("PG_USER")}:{os.getenv("PG_PASSWORD")}"
        f"@{os.getenv("PG_HOST")}:{os.getenv("PG_PORT")}/{os.getenv("PG_DATABASE")}"
        # "?sslmode=require"
    )

    if not embeddings:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=f"{client}_vector_docs",
        connection=connection, 
        use_jsonb=True
    )
    return vector_store