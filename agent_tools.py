"""Herramientas del agente para EcoMarket.

Incluye:
- buscar_en_base_de_conocimiento: herramienta RAG que consulta la Base de conocimientos.
- verificar_elegibilidad_producto: herramienta simulada para validar devoluciones.
- generar_etiqueta_devolucion: herramienta simulada para crear una etiqueta de envío.
"""
from __future__ import annotations

import json
import random
import string
from typing import List

from langchain.tools import tool

from rag_module import get_retriever


@tool
def buscar_en_base_de_conocimiento(query: str) -> str:
    """Busca información en la Base de conocimientos con un retriever semántico.

    Parámetros:
    - query: consulta del usuario en texto plano.

    Devuelve texto con fragmentos relevantes y sus fuentes. Si no hay índice o resultados, devuelve un mensaje informativo.
    """
    retriever = get_retriever()
    if retriever is None:
        return "RAG no disponible: no hay índice construido sobre la Base de conocimientos."

    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        return f"Error al recuperar información: {e}"

    if not docs:
        return "No se encontró información relevante en la Base de conocimientos."

    lines: List[str] = ["Resultados de la Base de conocimientos:"]
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "desconocido")
        snippet = d.page_content.strip().replace("\n", " ")
        lines.append(f"[{i}] ({src}) {snippet}")
    return "\n".join(lines)


@tool
def verificar_elegibilidad_producto(numero_orden: str, dias_desde_compra: int) -> str:
    """Verifica si una orden es elegible para devolución.

    Condición: elegible si dias_desde_compra <= 30.
    Devuelve un JSON con el resultado de elegibilidad y motivo.
    """
    elegible = dias_desde_compra <= 30
    motivo = (
        "Elegible: dentro de 30 días desde la compra." if elegible else "No elegible: superó los 30 días permitidos."
    )
    result = {
        "numero_orden": numero_orden,
        "dias_desde_compra": dias_desde_compra,
        "elegible": elegible,
        "motivo": motivo,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def generar_etiqueta_devolucion(numero_orden: str, direccion_cliente: str) -> str:
    """Genera datos simulados de una etiqueta de devolución.

    Devuelve un JSON con número de guía y URL de descarga.
    Requiere haber verificado elegibilidad antes de su uso (validación lógica en el agente).
    """
    # Número de guía simulado: ECOM seguido de 10 caracteres alfanuméricos
    guia = "ECOM" + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    url = f"https://devoluciones.ecomarket.example/etiqueta/{guia}"
    result = {
        "numero_orden": numero_orden,
        "direccion_cliente": direccion_cliente,
        "guia": guia,
        "url_etiqueta": url,
        "estado": "generada",
    }
    return json.dumps(result, ensure_ascii=False)


# Lista agregada de herramientas para el agente
all_tools = [
    buscar_en_base_de_conocimiento,
    verificar_elegibilidad_producto,
    generar_etiqueta_devolucion,
]
