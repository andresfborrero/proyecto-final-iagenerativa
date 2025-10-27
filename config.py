"""Configuración y constantes del proyecto EcoMarket.

Este módulo centraliza variables de configuración para el agente y el RAG.
"""
from __future__ import annotations

import os
from pathlib import Path

# Claves y modelos
# Reemplazar con tu API Key de Gemini o configurar la variable de entorno GEMINI_API_KEY.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyCSmA1cjqB1NXqoZH_dCsIxVJHx6Xza6cg")
GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"

# Rutas base
BASE_DIR: Path = Path(__file__).resolve().parent
KB_DIR: Path = BASE_DIR / "Base de conocimientos"

# Parámetros de RAG
BASE_CHUNK_SIZE: int = 800
BASE_CHUNK_OVERLAP: int = 150
TOP_K_RESULTS: int = 10

# Otras constantes útiles
APP_NAME: str = "proyecto_final_ecomarket"
STREAMLIT_PAGE_TITLE: str = "EcoBot - Devoluciones y Consultas"
STREAMLIT_PAGE_ICON: str = "🛒"
