"""Replaceable text and image embedding wrappers."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from schema import DEFAULT_TEXT_VECTOR_DIM, IMAGE_VECTOR_DIM


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def hash_text_embedding(text: str, dim: int = DEFAULT_TEXT_VECTOR_DIM) -> list[float]:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        return vector.tolist()
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return _l2_normalize(vector).tolist()


def hash_image_embedding(blob: bytes | None, dim: int = IMAGE_VECTOR_DIM) -> list[float]:
    vector = np.zeros(dim, dtype=np.float32)
    if not blob:
        return vector.tolist()
    digest = hashlib.blake2b(blob, digest_size=64).digest()
    for index, byte in enumerate(digest):
        vector[index % dim] += (byte / 255.0) - 0.5
    return _l2_normalize(vector).tolist()


@dataclass
class TextEmbedder:
    backend: str = "hash"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int | None = None
    batch_size: int = 128

    def __post_init__(self) -> None:
        self._model = None
        self._client = None
        self._cache: dict[str, list[float]] = {}
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            dim = int(self._model.get_sentence_embedding_dimension())
            self.dimensions = self.dimensions or dim
            if self.dimensions != dim:
                raise ValueError(
                    f"{self.model_name} returns {dim}-d vectors, not {self.dimensions}"
                )
        elif self.backend == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for --embedding-backend openai")
            from openai import OpenAI

            self._client = OpenAI()
            self.dimensions = self.dimensions or _default_openai_dimensions(self.model_name)
        elif self.backend == "hash":
            self.dimensions = self.dimensions or DEFAULT_TEXT_VECTOR_DIM
        else:
            raise ValueError(f"unknown text embedding backend: {self.backend}")

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        if self._model is None:
            if self._client is not None:
                vector = self._embed_openai([text])[0]
            else:
                vector = hash_text_embedding(text, self.dimensions or DEFAULT_TEXT_VECTOR_DIM)
            self._cache[text] = vector
            return vector
        vector = self._model.encode(text or "", normalize_embeddings=True)
        result = np.asarray(vector, dtype=np.float32).tolist()
        self._cache[text] = result
        return result

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        if all(text in self._cache for text in text_list):
            return [self._cache[text] for text in text_list]
        if self._client is not None:
            vectors = []
            missing = [text for text in text_list if text not in self._cache]
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                batch_vectors = self._embed_openai(batch)
                for text, vector in zip(batch, batch_vectors, strict=True):
                    self._cache[text] = vector
            return [self._cache[text] for text in text_list]
        if self._model is None:
            dim = self.dimensions or DEFAULT_TEXT_VECTOR_DIM
            for text in text_list:
                if text not in self._cache:
                    self._cache[text] = hash_text_embedding(text, dim)
            return [self._cache[text] for text in text_list]
        vectors = self._model.encode(text_list, normalize_embeddings=True)
        results = np.asarray(vectors, dtype=np.float32).tolist()
        for text, vector in zip(text_list, results, strict=True):
            self._cache[text] = vector
        return results

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialized")
        kwargs = {
            "model": self.model_name,
            "input": [text or "" for text in texts],
            "encoding_format": "float",
        }
        if self.dimensions is not None and self.model_name.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [np.asarray(item.embedding, dtype=np.float32).tolist() for item in ordered]


@dataclass
class ImageEmbedder:
    backend: str = "hash"
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    dimensions: int = IMAGE_VECTOR_DIM

    def __post_init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None
        if self.backend == "open-clip":
            import open_clip
            import torch

            self._torch = torch
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self._model.eval()
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        elif self.backend != "hash":
            raise ValueError(f"unknown image embedding backend: {self.backend}")

    def embed_text_query(self, text: str) -> list[float]:
        """Embed a text query into the CLIP image space, for text-to-image search."""
        if self._model is None or self._tokenizer is None or self._torch is None:
            # Offline/hash fallback: no meaningful text-to-image signal.
            return hash_image_embedding(None, self.dimensions)
        tokens = self._tokenizer([text or ""])
        with self._torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        vector = features.squeeze(0).cpu().numpy().astype(np.float32)
        return vector.tolist()

    def embed_blob(self, blob: bytes | None) -> list[float]:
        if self._model is None or self._preprocess is None or self._torch is None:
            return hash_image_embedding(blob, self.dimensions)
        if not blob:
            return [0.0] * self.dimensions
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(blob)).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        vector = features.squeeze(0).cpu().numpy().astype(np.float32)
        if len(vector) != self.dimensions:
            self.dimensions = int(len(vector))
        return vector.tolist()


def _default_openai_dimensions(model_name: str) -> int:
    if model_name == "text-embedding-3-large":
        return 3072
    if model_name == "text-embedding-3-small":
        return 1536
    if model_name == "text-embedding-ada-002":
        return 1536
    return DEFAULT_TEXT_VECTOR_DIM
