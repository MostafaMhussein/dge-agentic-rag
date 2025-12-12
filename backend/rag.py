"""Core RAG components for document processing and retrieval.

Handles PDF ingestion, embedding, and hybrid search with reranking.
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict

from llama_index.core import VectorStoreIndex, StorageContext, Settings as LlamaSettings, Document
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever

from .config import get_settings
from .tracing import trace_span

logger = logging.getLogger(__name__)

_embed_model: Optional[HuggingFaceEmbedding] = None
_reranker: Optional[FlagEmbeddingReranker] = None
_llm: Optional[Ollama] = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Initialize the embedding model."""
    global _embed_model
    
    if _embed_model is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embed_model}")
        _embed_model = HuggingFaceEmbedding(
            model_name=settings.embed_model,
            trust_remote_code=True,
        )
        LlamaSettings.embed_model = _embed_model
    
    return _embed_model


def get_reranker() -> FlagEmbeddingReranker:
    """Initialize the cross-encoder reranker."""
    global _reranker
    
    if _reranker is None:
        settings = get_settings()
        logger.info("Loading reranker: BAAI/bge-reranker-base")
        _reranker = FlagEmbeddingReranker(
            model="BAAI/bge-reranker-base",
            top_n=settings.rerank_top_n,
        )
    
    return _reranker


def get_llm() -> Ollama:
    """Initialize the Ollama LLM."""
    global _llm
    
    if _llm is None:
        settings = get_settings()
        logger.info(f"Configuring LLM: {settings.ollama_model}")
        _llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=300.0,
            temperature=0.1,
        )
        LlamaSettings.llm = _llm
    
    return _llm


def get_vector_store() -> PGVectorStore:
    """Create PGVector store connection."""
    settings = get_settings()
    return PGVectorStore.from_params(
        database=settings.postgres_db,
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        table_name="document_chunks",
        embed_dim=settings.embed_dim,
    )


def get_index() -> VectorStoreIndex:
    """Get the vector store index."""
    with trace_span("get_index"):
        get_embed_model()
        get_llm()
        
        vector_store = get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )


def load_documents(doc_dir: str = "/app/documents") -> List[Document]:
    """Load PDF documents using Docling.
    
    Converts PDFs to Markdown preserving structure for better chunking.
    """
    from llama_index.readers.docling import DoclingReader
    
    documents: List[Document] = []
    doc_path = Path(doc_dir)
    
    if not doc_path.exists():
        logger.warning(f"Documents directory not found: {doc_dir}")
        return []
    
    files = list(doc_path.glob("*.pdf")) + list(doc_path.glob("*.PDF"))
    files += list(doc_path.glob("*.docx")) + list(doc_path.glob("*.DOCX"))
    
    logger.info(f"Found {len(files)} documents to process")
    
    reader = DoclingReader()
    
    for file_path in files:
        filename = file_path.name
        if filename.startswith("."):
            continue
        
        try:
            logger.info(f"Processing: {filename}")
            docs = reader.load_data(file_path=str(file_path))
            
            for doc in docs:
                doc.metadata.update({
                    "filename": filename,
                    "doc_type": _classify_document(filename),
                    "source": str(file_path),
                })
                documents.append(doc)
            
            logger.info(f"Loaded: {filename} ({len(docs)} sections)")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    return documents


def _classify_document(filename: str) -> str:
    """Classify document type from filename."""
    name = filename.lower()
    
    if "procurement" in name and "standard" in name:
        return "procurement_standards"
    elif "hr" in name or "bylaw" in name:
        return "hr_bylaws"
    elif "security" in name:
        return "information_security"
    elif "procurement" in name:
        return "procurement_manual"
    return "government_policy"


def create_splitter() -> SemanticSplitterNodeParser:
    """Create semantic splitter for topic-based chunking."""
    return SemanticSplitterNodeParser.from_defaults(
        embed_model=get_embed_model(),
        buffer_size=1,
        breakpoint_percentile_threshold=95,
    )


class HybridRetriever(BaseRetriever):
    """Combines vector search with BM25 keyword matching."""
    
    def __init__(self, vector_index: VectorStoreIndex, nodes: Optional[List[TextNode]] = None, top_k: int = 10):
        super().__init__()
        
        self.vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
        self.bm25_retriever = None
        
        if nodes and len(nodes) > 0:
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
            logger.info(f"BM25 initialized with {len(nodes)} nodes")
        else:
            logger.info("Vector-only retrieval (no BM25)")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Execute hybrid retrieval."""
        with trace_span("hybrid_retrieval"):
            vector_nodes = self.vector_retriever.retrieve(query_bundle)
            
            if self.bm25_retriever is None:
                return vector_nodes
            
            bm25_nodes = self.bm25_retriever.retrieve(query_bundle.query_str)
            
            merged: Dict[str, NodeWithScore] = {}
            for node in vector_nodes + bm25_nodes:
                node_id = node.node.node_id
                if node_id not in merged or node.score > merged[node_id].score:
                    merged[node_id] = node
            
            return list(merged.values())
