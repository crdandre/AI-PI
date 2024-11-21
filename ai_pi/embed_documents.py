"""
Uses llamaindex's inbuilt NVIDIA-NIM embedding functionality

for testing ContextStorageInterface.storage_context is None for in-memory vector storage,
but for storing this data in a file, see:
https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/#using-vectorstoreindex
"""
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding


class ContextStorageInterface:
    """
    Handles embedding model selection, database type and location,
    data import, chunk size and overlap, 
    """
    def __init__(
        self,
        data_dir="processed_documents",
        default_top_k = 10,
        embedding_model="nvidia/nv-embedqa-mistral-7b-v2",
        chunk_size = 1000,
        chunk_overlap = 200,
        storage_dir = "vector_db",
        storage_context = None, #see top docstring for more on this
        store_embeddings = False,
    ):
        self.data = SimpleDirectoryReader(data_dir).load_data()
        self.default_top_k = default_top_k
        
        self.vector_index = VectorStoreIndex.from_documents(
            documents=self.data,
            storage_context=storage_context
        )        
        self.summary_index = DocumentSummaryIndex.from_documents(
            documents=self.data,
            storage_context=storage_context
        )
        if storage_context is None and store_embeddings:
            self.vector_index.storage_context.persist(persist_dir=storage_dir)
            self.summary_index.storage_context.persist(persist_dir=storage_dir)
        elif storage_context is not None and store_embeddings:
            raise ValueError("Cannot pre-define storage_context while trying to create it")
        
        #TODO: improve with reranking?
        self.vector_retriever = self.vector_index.as_retriever(
            similarity_top_k=self.default_top_k
        )
        
        #These parameters can be optimized
        Settings.text_splitter=SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        #this could be set many ways for remote or local models
        #i.e. HuggingFaceEmbedding
        Settings.embed_model=NVIDIAEmbedding(
            model=embedding_model,
            truncate="END"
        )
        
    def retrieve(self, query):
        raw_response = self.retriever.retrieve(query)
        retrieved_texts = [node.node.text for node in raw_response]
        return "\n\n".join(retrieved_texts)