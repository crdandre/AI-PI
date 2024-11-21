import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient, models
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import DataParallel
from qdrant_client.http.models import PointStruct
import uuid

from embedding_models import get_text_embeddings

@dataclass
class DocumentMetadata:
    doc_id: str
    version: int
    author: str
    timestamp: str
    type: str  # 'document', 'comment', or 'revision'


class DocumentVectorStore:
    def __init__(
        self,
        collection_name: str,
        vector_store_path: str = "vector_store",
        vector_dim: int = 4096
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        os.makedirs(vector_store_path, exist_ok=True)
        
        try:
            self.client = QdrantClient(path=vector_store_path)
            self._check_create_collection()
        except Exception as e:
            raise ValueError(f"Error initializing vector store: {e}")
            
    def _check_create_collection(self):
        """Check if collection exists, create if it doesn't."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE
                )
            )
            
    def store_document_embeddings(self, embeddings: List[np.ndarray], metadata: List[DocumentMetadata]):
        """Store document embeddings in vector DB."""
        points = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            # Generate a UUID based on the document metadata to ensure consistency
            point_id = uuid.uuid5(
                uuid.NAMESPACE_DNS, 
                f"{meta.doc_id}_{meta.type}_{i}"
            )
            
            point = PointStruct(
                id=str(point_id),  # Convert UUID to string
                payload={
                    'doc_id': meta.doc_id,
                    'version': meta.version,
                    'author': meta.author,
                    'timestamp': meta.timestamp,
                    'type': meta.type
                },
                vector=embedding.tolist()
            )
            points.append(point)

        # Batch upsert to vector store
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
            
    def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        return [{'id': hit.id, 'score': hit.score, 'metadata': hit.payload} for hit in results]

class DocumentEmbedder:
    """Handles embedding generation for document histories."""
    
    def __init__(
        self,
        vector_store: Optional[DocumentVectorStore] = None
    ):
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.vector_store = vector_store
        
    def embed_batch(self, texts: List[str], max_workers: int = 5) -> Tuple[np.ndarray, int]:
        """Batch process multiple texts for embedding."""
        return get_text_embeddings(
            texts=texts,
            max_workers=max_workers,
            embedding_cache=self.embedding_cache
        )
        
    def _prepare_comment_text(self, comment: Dict) -> str:
        """Format comment data for embedding."""
        text = (
            f"{comment['referenced_text']} "
            f"Comment: {comment['text']} "
            f"Author: {comment['author']} "
            f"Date: {comment['date']}"
        )
        if comment['replies']:
            replies_text = " Replies: " + " ".join(
                f"{reply['text']} by {reply['author']}"
                for reply in comment['replies']
            )
            text += replies_text
        return text
    
    def _prepare_revision_text(self, revision: Dict) -> str:
        """Format revision data for embedding."""
        return (
            f"{revision['referenced_text']} "
            f"Revision: {revision['text']} "
            f"Type: {revision['type']} "
            f"Author: {revision['author']} "
            f"Date: {revision['date']}"
        )

    def embed_document_history(self, document_history: Dict) -> Dict:
        """Generate embeddings for all comments and revisions and store in vector DB."""
        embedded_history = document_history.copy()
        
        # Prepare batches for embedding
        texts = []
        metadata = []
        
        # Add document
        texts.append(document_history['content'])
        metadata.append(DocumentMetadata(
            doc_id=document_history['document_id'],
            version=document_history.get('version', 1),
            author=document_history.get('author', 'unknown'),
            timestamp=document_history.get('metadata', {}).get('last_modified', ''),
            type='document'
        ))
        
        # Add comments and revisions
        for comment in document_history['comments']:
            texts.append(self._prepare_comment_text(comment))
            metadata.append(DocumentMetadata(
                doc_id=document_history['document_id'],
                version=document_history.get('version', 1),
                author=comment['author'],
                timestamp=comment['date'],
                type='comment'
            ))
            
        for revision in document_history['revisions']:
            texts.append(self._prepare_revision_text(revision))
            metadata.append(DocumentMetadata(
                doc_id=document_history['document_id'],
                version=1,
                author=revision['author'],
                timestamp=revision['date'],
                type='revision'
            ))
            
        # Batch embed
        embeddings, total_tokens = self.embed_batch(texts)
        
        # Store in vector DB if available
        if self.vector_store:
            self.vector_store.store_document_embeddings(embeddings, metadata)
            
        # Update document history with embeddings
        for i, comment in enumerate(embedded_history['comments']):
            comment['embedding'] = embeddings[i + 1].tolist()  # +1 because document is first
            
        offset = len(embedded_history['comments']) + 1
        for i, revision in enumerate(embedded_history['revisions']):
            revision['embedding'] = embeddings[i + offset].tolist()
            
        embedded_history['metadata']['total_tokens'] = total_tokens
        return embedded_history

    def find_similar(self, query: str, embedded_doc: Dict) -> List[Dict]:
        """Find similar items to query in embedded document."""
        query_embedding, _ = self.embedding_model.get_embedding(query, is_query=True)
        
        results = []
        for item_type in ['document', 'comments', 'revisions']:
            if item_type in embedded_doc:
                items = embedded_doc[item_type]
                for item in items:
                    if 'embedding' in item:
                        sim_score = cosine_similarity([query_embedding], [item['embedding']])[0][0]
                        results.append({
                            'type': item_type.rstrip('s'),  # Remove 's' from type
                            'item': item,
                            'similarity': sim_score
                        })
        
        # Sort by similarity score
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        return results

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import os
    
    parser = argparse.ArgumentParser(description='Document Embedding System')
    parser.add_argument('--collection', type=str, default="documents",
                       help='Name of the vector collection')
    parser.add_argument('--store_path', type=str, default="vector_store",
                       help='Path to store vector database')
    parser.add_argument('--encoder_type', type=str, default="hf",
                       help='Type of encoder to use (hf, openai, azure, together)')
    parser.add_argument('--model', type=str, default="sentence-transformers/all-mpnet-base-v2",
                       help='Model name (if using HuggingFace)')
    args = parser.parse_args()

    # Set environment variables for model selection
    os.environ["ENCODER_API_TYPE"] = args.encoder_type
    if args.encoder_type == "hf":
        os.environ["HF_MODEL"] = args.model

    # Initialize vector store with appropriate dimensions
    vector_dimensions = {
        "hf": 768,  # all-mpnet-base-v2 dimension
        "openai": 1536,  # text-embedding-3-small dimension
        "azure": 1536,  # Azure OpenAI dimension
        "together": 1024,  # BGE-large dimension
    }
    
    vector_dim = vector_dimensions.get(args.encoder_type, 768)  # default to HF dimension

    # Initialize vector store
    vector_store = DocumentVectorStore(
        collection_name=args.collection,
        vector_store_path=args.store_path,
        vector_dim=vector_dim  # Add this parameter to DocumentVectorStore
    )

    # Initialize document embedder
    embedder = DocumentEmbedder(vector_store=vector_store)

    # Example document history
    example_doc = {
        "document_id": "example_paper.docx",
        "content": "This is the main content of the academic paper discussing the methodology.",
        "revisions": [
            {
                "id": "rev_0",
                "type": "insertion",
                "text": "We conducted additional experiments to validate our findings.",
                "author": "Dr. Smith",
                "date": datetime.now().isoformat(),
                "position": {
                    "start": 50,
                    "end": 100,
                    "expanded_start": 45,
                    "expanded_end": 105
                },
                "referenced_text": "the methodology. We conducted additional experiments",
                "formatting": {"bold": True},
                "parent_id": None
            }
        ],
        "comments": [
            {
                "id": "comment_1",
                "text": "Consider adding statistical significance analysis here.",
                "author": "Dr. Johnson",
                "date": datetime.now().isoformat(),
                "position": {
                    "start": 80,
                    "end": 100,
                    "expanded_start": 75,
                    "expanded_end": 105
                },
                "referenced_text": "additional experiments to validate",
                "resolved": False,
                "replies": [
                    {
                        "id": "comment_2",
                        "text": "Agreed, I'll add p-values for all experiments.",
                        "author": "Dr. Smith",
                        "date": datetime.now().isoformat(),
                        "position": {
                            "start": 80,
                            "end": 100,
                            "expanded_start": 75,
                            "expanded_end": 105
                        },
                        "referenced_text": "additional experiments to validate",
                        "resolved": True,
                        "replies": []
                    }
                ],
                "related_revision_id": "rev_0"
            }
        ],
        "metadata": {
            "last_modified": datetime.now().isoformat(),
            "revision_count": 1,
            "comment_count": 2,
            "contributors": ["Dr. Smith", "Dr. Johnson"]
        }
    }

    # Embed document history
    embedded_doc = embedder.embed_document_history(example_doc)
    print(f"\nEmbedded document with {embedded_doc['metadata']['total_tokens']} tokens")

    # Example similarity search
    query = "Who suggested to add statistical significance analysis?"
    similar_items = embedder.find_similar(query, embedded_doc)
    
    print("\nSimilar items to query:", query)
    for item in similar_items:
        print(f"\nType: {item['type']}")
        print(f"Author: {item['item']['author']}")
        print(f"Text: {item['item']['text']}")
        print(f"Referenced text: {item['item']['referenced_text']}")
        print(f"Similarity: {item['similarity']:.3f}")

