#!/usr/bin/env python3
"""Document indexer for the RAG system.

Processes PDF documents and indexes them into PGVector for retrieval.
"""
import argparse
import logging
import sys
import time
from typing import List

from llama_index.core.schema import TextNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument("--documents-dir", default="/app/documents", help="Path to documents")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Document Indexer")
    print("=" * 60)
    
    start_time = time.time()
    
    from backend.rag import load_documents, get_index, get_embed_model, create_splitter
    from backend.tracing import init_phoenix_tracing
    
    init_phoenix_tracing()
    
    print(f"\n[1/3] Loading documents from: {args.documents_dir}")
    docs = load_documents(args.documents_dir)
    
    if not docs:
        print(f"No documents found in {args.documents_dir}")
        sys.exit(1)
    
    print(f"Loaded {len(docs)} documents:")
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown")
        chars = len(doc.text)
        print(f"  - {filename}: {chars:,} chars")
    
    print(f"\n[2/3] Chunking documents...")
    get_embed_model()
    splitter = create_splitter()
    
    all_nodes: List[TextNode] = []
    
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown")
        print(f"  Processing: {filename}")
        
        try:
            nodes = splitter.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata.update({
                    "filename": doc.metadata.get("filename"),
                    "doc_type": doc.metadata.get("doc_type"),
                    "source": doc.metadata.get("source"),
                })
            all_nodes.extend(nodes)
            print(f"    -> {len(nodes)} chunks")
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"Total chunks: {len(all_nodes)}")
    
    print(f"\n[3/3] Indexing to vector store...")
    
    try:
        index = get_index()
        batch_size = 50
        
        for i in range(0, len(all_nodes), batch_size):
            batch = all_nodes[i:i + batch_size]
            index.insert_nodes(batch)
            batch_num = i // batch_size + 1
            total_batches = (len(all_nodes) + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches}")
        
        print("Indexing complete!")
        
    except Exception as e:
        print(f"Indexing error: {e}")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(all_nodes)}")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
