"""Aplicación principal de Streamlit para EcoBot.

Provee una interfaz de chat con un agente basado en LangChain que:
- Usa RAG sobre la carpeta "Base de conocimientos" para preguntas generales.
- Simula el flujo de devoluciones con herramientas de verificación y generación de etiqueta.
"""
from __future__ import annotations

import os
import streamlit as st

# Silenciar mensaje ruidoso del SDK de Google sobre 'title' en el schema ANTES de importar el cliente
import logging
import warnings
class _SuppressSchemaTitle(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Key 'title' is not supported in schema" not in record.getMessage()

# Silenciar logger de google.generativeai completamente (además del filtro)
logger_gemini = logging.getLogger("google.generativeai")
logger_gemini.handlers.clear()
logger_gemini.propagate = False
logger_gemini.addFilter(_SuppressSchemaTitle())
logger_gemini.setLevel(logging.CRITICAL)

# Suprimir también como warning estándar, apuntando al módulo específico
warnings.filterwarnings(
    "ignore",
    message=r"Key 'title' is not supported in schema.*",
    category=UserWarning,
    module=r".*google\.generativeai.*",
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    KB_DIR,
)
from agent_tools import all_tools

# Configuración de página
st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, page_icon=STREAMLIT_PAGE_ICON)
st.title("EcoBot - Asistente de Devoluciones y Consultas")

# Asegurar variable esperada por SDK
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)

# Validaciones y avisos
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    st.sidebar.error(
        "Configura la variable de entorno GEMINI_API_KEY o actualiza config.py con tu API key."
    )

if not KB_DIR.exists():
    st.sidebar.warning(
        f"No se encontró la carpeta de Base de conocimientos en: {KB_DIR}. Crea la carpeta y añade archivos .txt/.json."
    )

# Inicialización del LLM Gemini
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0, api_key=GEMINI_API_KEY)

# Prompt del agente
system_instructions = (
    "Eres EcoBot, asistente de EcoMarket. Responde en español, de forma breve, clara y empática. "
    "Dispones de herramientas. Sigue estas reglas: "
    "1) Para preguntas generales, prioriza 'buscar_en_base_de_conocimiento'. "
    "2) Para devoluciones, primero solicita numero_orden y dias_desde_compra. "
    "   Solo llama 'verificar_elegibilidad_producto' cuando tengas ambos. "
    "3) Nunca generes una etiqueta antes de verificar elegibilidad. "
    "   Llama 'generar_etiqueta_devolucion' únicamente si la orden es elegible y el usuario provee direccion_cliente. "
    "4) Si no hay información suficiente, indícalo y ofrece usar la base de conocimientos. "
    "5) No inventes datos como números de guía o políticas no documentadas."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    # Requerido por create_tool_calling_agent para registrar pasos intermedios
    MessagesPlaceholder("agent_scratchpad"),
])

# Construcción del agente con herramientas
agent = create_tool_calling_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=False)

# Estado de sesión para historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List[BaseMessage]

# Render del historial
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Entrada del usuario
user_input = st.chat_input("Escribe tu mensaje...")
if user_input:
    # Añadir mensaje del usuario
    st.session_state.chat_history.append(HumanMessage(user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invocar agente
    try:
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history,
        })
        output_text = result.get("output", "")
    except Exception as e:
        output_text = f"Ocurrió un error al ejecutar el agente: {e}"

    # Mostrar respuesta y guardar en historial
    with st.chat_message("assistant"):
        st.markdown(output_text)
    st.session_state.chat_history.append(AIMessage(output_text))
