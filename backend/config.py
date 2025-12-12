"""Application configuration management."""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment based configuration."""
    
    # Database
    postgres_db: str = "ragdb"
    postgres_user: str = "raguser"
    postgres_password: str = "ragpass"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    
    # Ollama
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.1"
    
    # Embeddings
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768
    
    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_n: int = 5
    
    # Phoenix
    phoenix_endpoint: str = "http://phoenix:4317"
    phoenix_project_name: str = "abu-dhabi-rag"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
