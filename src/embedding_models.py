import os
from typing import List, Tuple, Union, Optional, Dict
import numpy as np
from abc import ABC, abstractmethod
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        """Get embedding for text. Returns (embedding, token_count)."""
        pass

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.model = model

    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        data = {"input": text, "model": self.model}
        response = requests.post(self.url, headers=self.headers, json=data)
        if response.status_code == 200:
            data = response.json()
            embedding = np.array(data["data"][0]["embedding"])
            token = data["usage"]["prompt_tokens"]
            return embedding, token
        else:
            response.raise_for_status()


class HFEmbeddingModel(EmbeddingModel):
    def __init__(self, model: str = "nvidia/NV-Embed-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self.model.eval()  # Set to evaluation mode
        
    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Convert the output dictionary's tensor to numpy
            if isinstance(outputs, dict):
                # Get the first tensor from the dictionary
                first_tensor = next(iter(outputs.values()))
                embedding = first_tensor.numpy()[0]
            else:
                embedding = outputs.numpy()[0]
        return embedding, len(inputs.input_ids[0])
    

def get_embedding_model() -> EmbeddingModel:
    """Get embedding model based on environment configuration."""
    encoder_type = os.getenv("ENCODER_API_TYPE", "hf").lower()
    if encoder_type == "openai":
        return OpenAIEmbeddingModel()
    elif encoder_type == "hf":
        return HFEmbeddingModel()
    else:
        raise ValueError(
            f"Invalid encoder type: {encoder_type}. "
            "Check environment variable ENCODER_API_TYPE. "
            "Valid values are: openai, azure, together, hf"
        )

def get_text_embeddings(
    texts: Union[str, List[str]],
    max_workers: int = 5,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, int]:
    """Get embeddings for text(s) using configured embedding model."""
    embedding_model = get_embedding_model()

    def fetch_embedding(text: str) -> Tuple[str, np.ndarray, int]:
        if embedding_cache is not None and text in embedding_cache:
            return text, embedding_cache[text], 0
        embedding, token_usage = embedding_model.get_embedding(text)
        return text, embedding, token_usage

    if isinstance(texts, str):
        _, embedding, tokens = fetch_embedding(texts)
        return np.array(embedding), tokens

    embeddings = []
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_embedding, text): text for text in texts}
        for future in as_completed(futures):
            try:
                text, embedding, tokens = future.result()
                embeddings.append((text, embedding, tokens))
                total_tokens += tokens
            except Exception as e:
                print(f"Error processing text: {futures[future]}")
                print(e)

    embeddings.sort(key=lambda x: texts.index(x[0]))
    if embedding_cache is not None:
        for text, embedding, _ in embeddings:
            embedding_cache[text] = embedding
    
    return np.array([e[1] for e in embeddings]), total_tokens
