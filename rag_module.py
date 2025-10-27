"""Módulo RAG para cargar, procesar y crear el índice vectorial.

Funciones:
- _load_kb_files: carga archivos .txt y .json desde la carpeta de Base de conocimientos.
- _chunk_documents: divide el texto en fragmentos usando RecursiveCharacterTextSplitter.
- _create_vector_store: crea un índice FAISS o Chroma en memoria y retorna un retriever.
- get_retriever: orquesta el flujo y usa caché en memoria para evitar recomputación.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Intentar importar FAISS; si no está disponible (p. ej., en Windows), haremos fallback a Chroma
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # ImportError u otros
    FAISS = None  # type: ignore
    _FAISS_AVAILABLE = False
# Embeddings con Gemini para evitar dependencias pesadas de PyTorch
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    KB_DIR,
    BASE_CHUNK_SIZE,
    BASE_CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    TOP_K_RESULTS,
    GEMINI_API_KEY,
)

# Caché simple en memoria del retriever
_cached_retriever = None


def _load_kb_files(kb_dir: Path) -> List[Dict[str, Any]]:
    """Carga archivos .txt y .json de la base de conocimientos.

    Devuelve una lista de diccionarios con claves: text (str) y metadata (dict).
    """
    docs: List[Dict[str, Any]] = []
    if not kb_dir.exists():
        return docs

    for p in kb_dir.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in {".txt", ".md"}:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            docs.append({
                "text": text,
                "metadata": {"source": str(p), "type": "text"},
            })
        elif suffix == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                data = {}

            def _flatten(obj, prefix: str = "") -> List[str]:
                lines: List[str] = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                        lines.extend(_flatten(v, key))
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        key = f"{prefix}[{i}]"
                        lines.extend(_flatten(v, key))
                else:
                    val = str(obj).replace("\n", " ").strip()
                    lines.append(f"{prefix}: {val}")
                return lines

            flat_lines = _flatten(data)
            text = "\n".join(flat_lines)
            docs.append({
                "text": text,
                "metadata": {"source": str(p), "type": "json"},
            })
    return docs


def _chunk_documents(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Divide documentos en fragmentos y adjunta metadatos mínimos."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASE_CHUNK_SIZE,
        chunk_overlap=BASE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
    )

    chunks: List[Dict[str, Any]] = []
    for i, d in enumerate(raw_docs):
        text = d.get("text", "")
        meta = d.get("metadata", {})
        parts = [c for c in splitter.split_text(text) if c.strip()]
        for j, content in enumerate(parts):
            chunks.append({
                "text": content,
                "metadata": {
                    **meta,
                    "chunk_id": f"{i}-{j}",
                },
            })
    return chunks


def _create_vector_store(chunks: List[Dict[str, Any]]):
    """Crea un índice vectorial en memoria (FAISS si está disponible; si no, Chroma)."""
    if not chunks:
        return None

    # Usar embeddings de Gemini (evita instalar torch/sentence-transformers)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_document",
        google_api_key=GEMINI_API_KEY,
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Intentar FAISS primero
    if _FAISS_AVAILABLE:
        try:
            vectordb = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
            return vectordb.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
        except Exception:
            # Si falla la creación con FAISS, continuar al fallback
            pass

    # Fallback: Chroma en memoria
    try:
        from langchain_community.vectorstores import Chroma  # import local para no requerirlo si FAISS funciona
        vectordb = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name="ecomarket_kb",
        )
        return vectordb.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    except Exception as e:
        # Último recurso: sin índice
        return None


def get_retriever():
    """Obtiene un retriever FAISS cacheado basado en la carpeta de Base de conocimientos."""
    global _cached_retriever
    if _cached_retriever is not None:
        return _cached_retriever

    raw_docs = _load_kb_files(KB_DIR)
    chunks = _chunk_documents(raw_docs)
    retriever = _create_vector_store(chunks)
    _cached_retriever = retriever
    return retriever


if __name__ == "__main__":
    # Prueba rápida del módulo
    r = get_retriever()
    if r is None:
        print("No hay datos para indexar en la Base de conocimientos.")
    else:
        res = r.get_relevant_documents("¿Cuál es la política de devoluciones?")
        print(f"Documentos recuperados: {len(res)}")
        if res:
            print("Ejemplo fuente:", res[0].metadata.get("source"))
