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
from time import time
from typing import List
import re
# Nuevos imports para PDF
from datetime import datetime
try:
    from fpdf import FPDF
except Exception:  # módulo opcional; si no está, omitimos PDF
    FPDF = None  # type: ignore

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
# Importar herramientas específicas para uso directo (acceso a .func)
from agent_tools import verificar_elegibilidad_producto, generar_etiqueta_devolucion

# ----------------------
# Logger de la aplicación
# ----------------------
_ecobot_logger = logging.getLogger("ecobot")
if not _ecobot_logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _handler.setFormatter(_formatter)
    _ecobot_logger.addHandler(_handler)
_ecobot_logger.setLevel(logging.INFO)


def _log_session_state(context: str) -> None:
    """Imprime un resumen del estado de sesión en los logs.

    Evita serializar objetos complejos; para chat_history muestra longitud y último mensaje.
    """
    try:
        summary: dict = {}
        for k, v in st.session_state.items():
            if k == "chat_history" and isinstance(v, list):
                last = None
                if v:
                    last_obj = v[-1]
                    # Los mensajes de LangChain suelen tener atributo .content
                    last = getattr(last_obj, "content", str(last_obj))
                    # Limitar tamaño del log
                    if isinstance(last, str) and len(last) > 300:
                        last = last[:300] + "…"
                summary[k] = {
                    "type": "list",
                    "length": len(v),
                    "last": last,
                    "message_types": sorted({type(m).__name__ for m in v}),
                }
            else:
                summary[k] = type(v).__name__
        _ecobot_logger.info("Session state [%s]: %s", context, summary)
    except Exception as e:
        _ecobot_logger.exception("Error registrando session_state [%s]: %s", context, e)

# Configuración de página
st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, page_icon=STREAMLIT_PAGE_ICON)
st.title("EcoBot - Asistente de Devoluciones y Consultas")

# Controles en la barra lateral
st.sidebar.subheader("Sesión")
if "ttl_minutes" not in st.session_state:
    st.session_state.ttl_minutes = 15
if "max_ctx_messages" not in st.session_state:
    st.session_state.max_ctx_messages = 12
if "agent_verbose" not in st.session_state:
    st.session_state.agent_verbose = False
# Estado para PDF de etiqueta
if "etiqueta_pdf" not in st.session_state:
    st.session_state.etiqueta_pdf = None
st.session_state.ttl_minutes = int(st.sidebar.number_input("TTL inactividad (min)", 5, 120, st.session_state.ttl_minutes))
st.session_state.max_ctx_messages = int(st.sidebar.number_input("Mensajes al modelo (contexto)", 4, 40, st.session_state.max_ctx_messages))
st.session_state.agent_verbose = bool(st.sidebar.checkbox("Modo depuración (verbose)", value=st.session_state.agent_verbose))
if st.sidebar.button("Reiniciar conversación"):
    st.session_state.chat_history = []
    st.session_state.last_activity_ts = time()
    # Limpiar completamente el contexto de devoluciones para evitar estados obsoletos
    st.session_state.devolucion_fields = {
        "numero_orden": None,
        "dias_desde_compra": None,
        "direccion_cliente": None,
    }
    st.session_state.elegibilidad_result = None
    st.session_state.etiqueta_result = None
    st.session_state.etiqueta_pdf = None
    st.sidebar.success("Conversación reiniciada.")

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
    "5) Si el usuario responde solo con un número y estabas pidiendo los días, interprétalo como dias_desde_compra. "
    "6) Usa el contexto de devolución proporcionado para no repetir preguntas innecesarias. "
    "7) No inventes datos como números de guía o políticas no documentadas. "
    "\n\nContexto de la devolución actual: {devolucion_context}"
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
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=st.session_state.agent_verbose, max_iterations=6, early_stopping_method="generate")

# Estado de sesión para historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List[BaseMessage]
if "last_activity_ts" not in st.session_state:
    st.session_state.last_activity_ts = time()
if "devolucion_fields" not in st.session_state:
    st.session_state.devolucion_fields = {
        "numero_orden": None,
        "dias_desde_compra": None,
        "direccion_cliente": None,
    }
if "elegibilidad_result" not in st.session_state:
    st.session_state.elegibilidad_result = None
if "etiqueta_result" not in st.session_state:
    st.session_state.etiqueta_result = None


def _update_devolucion_fields(text: str) -> None:
    t = text.strip()
    # numero de orden con etiqueta
    m = re.search(r"(?:numero\s*de\s*orden|numero_?orden|orden)\s*[:=]?\s*([A-Za-z]{2,5}-?\d{3,8})", t, re.IGNORECASE)
    if m:
        st.session_state.devolucion_fields["numero_orden"] = m.group(1).upper()
    else:
        # numero de orden sin etiqueta (p.ej., "ORD-1001")
        m2 = re.search(r"\b([A-Za-z]{2,5}-?\d{3,8})\b", t)
        if m2:
            st.session_state.devolucion_fields["numero_orden"] = m2.group(1).upper()

    # dias desde compra (etiqueta antes del número)
    m = re.search(r"(?:dias\s*(?:desde\s*(?:la\s*)?compra)?)\s*[:=]?\s*(\d{1,3})\b", t, re.IGNORECASE)
    if m:
        st.session_state.devolucion_fields["dias_desde_compra"] = int(m.group(1))
    else:
        # número seguido de 'dia/dias' (con o sin tilde), ej: "5 dias" o "tiene 5 días"
        m2 = re.search(r"\b(\d{1,3})\s*d[ií]a[s]?\b", t, re.IGNORECASE)
        if m2:
            st.session_state.devolucion_fields["dias_desde_compra"] = int(m2.group(1))
        elif re.fullmatch(r"\d{1,3}", t):
            # mensaje numérico únicamente
            st.session_state.devolucion_fields["dias_desde_compra"] = int(t)

    # direccion (preferir clave=valor)
    m = re.search(r"(?:direccion(?:_cliente)?)\s*[:=]\s*(.+)$", t, re.IGNORECASE)
    if m:
        st.session_state.devolucion_fields["direccion_cliente"] = m.group(1).strip()
        return

    # Si ya es elegible y se estaba pidiendo la dirección, tomar el texto libre como dirección
    elig = st.session_state.get("elegibilidad_result")
    if (
        st.session_state.devolucion_fields.get("direccion_cliente") in (None, "")
        and isinstance(elig, dict) and elig.get("elegible") is True
    ):
        if len(t) >= 5 and not re.fullmatch(r"\d{1,3}", t):
            st.session_state.devolucion_fields["direccion_cliente"] = t
            return

    # Heurística genérica de dirección (detecta palabras comunes y presencia de números)
    if st.session_state.devolucion_fields.get("direccion_cliente") in (None, ""):
        addr_keywords = r"\b(?:calle|cll|carrera|cra|avenida|av|transv|tv|diag|dg|km|kil[oó]metro|mz|manzana|apto|apartamento|edif|bloque|#|n°|no\.?|sector)\b"
        if re.search(addr_keywords, t, re.IGNORECASE) and re.search(r"\d", t):
            st.session_state.devolucion_fields["direccion_cliente"] = t
            return


def _build_devolucion_context() -> str:
    f = st.session_state.devolucion_fields
    def fmt(v):
        return "ND" if v in (None, "") else str(v)
    return (
        f"numero_orden={fmt(f.get('numero_orden'))}; "
        f"dias_desde_compra={fmt(f.get('dias_desde_compra'))}; "
        f"direccion_cliente={fmt(f.get('direccion_cliente'))}"
    )


# Helper: construir PDF en memoria con datos de la etiqueta
def _build_label_pdf(etiqueta: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Encabezado
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "EcoMarket - Etiqueta de Devolución", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    # Datos de la etiqueta
    orden = etiqueta.get("numero_orden", "-")
    guia = etiqueta.get("guia", "-")
    direccion = etiqueta.get("direccion_cliente", "-")

    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 8, "N° de orden:")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, str(orden), ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 8, "Guía:")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, str(guia), ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 8, "Dirección:")
    pdf.set_font("Arial", size=12)
    # Dividir dirección en varias líneas si es larga
    for line in pdf.multi_cell(0, 8, str(direccion), split_only=True):
        pdf.cell(0, 8, line, ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 6, "Pega esta etiqueta en el paquete. Llévalo al punto de envío autorizado.")

    # Pie de página simple
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 6, "EcoMarket - Devoluciones", ln=True)

    # Retornar bytes
    # fpdf2 devuelve str (latin-1) con dest='S'; convertir a bytes
    try:
        return pdf.output(dest="S").encode("latin1")
    except Exception:
        # Fallback básico: guardar a temp y reabrir (menos ideal, pero robusto)
        import tempfile, os as _os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        tmp.close()
        pdf.output(tmp_path)
        with open(tmp_path, "rb") as fh:
            data = fh.read()
        try:
            _os.remove(tmp_path)
        except Exception:
            pass
        return data


def _maybe_handle_devolucion() -> str | None:
    f = st.session_state.devolucion_fields
    numero = f.get("numero_orden")
    dias = f.get("dias_desde_compra")
    direccion = f.get("direccion_cliente")

    # Normalizar/depurar resultados previos: si la elegibilidad almacenada no corresponde
    # al caso actual (otro número de orden o días distintos), reiniciar para recalcular
    prev = st.session_state.elegibilidad_result if isinstance(st.session_state.elegibilidad_result, dict) else None
    try:
        if prev and (
            (numero and str(prev.get("numero_orden", "")).upper() != str(numero).upper()) or
            (isinstance(dias, int) and isinstance(prev.get("dias_desde_compra"), (int, float)) and int(prev.get("dias_desde_compra")) != int(dias))
        ):
            st.session_state.elegibilidad_result = None
            st.session_state.etiqueta_result = None
            st.session_state.etiqueta_pdf = None
            prev = None
    except Exception:
        # Si el formato no es el esperado, limpiar por seguridad
        st.session_state.elegibilidad_result = None
        st.session_state.etiqueta_result = None
        st.session_state.etiqueta_pdf = None
        prev = None

    # Paso 1: si ya tenemos elegibilidad calculada y es no elegible, informar y resetear contexto mínimo
    if st.session_state.elegibilidad_result and not st.session_state.elegibilidad_result.get("elegible", False):
        res = st.session_state.elegibilidad_result
        # Luego de informar, limpiar solo los campos de devolución para permitir nuevos casos
        st.session_state.devolucion_fields = {"numero_orden": None, "dias_desde_compra": None, "direccion_cliente": None}
        st.session_state.elegibilidad_result = None
        st.session_state.etiqueta_result = None
        st.session_state.etiqueta_pdf = None
        return f"Tu orden {res.get('numero_orden')} no es elegible para devolución. Motivo: {res.get('motivo')}"

    # Paso 2: si tenemos número y días pero aún no hemos calculado elegibilidad
    if numero and isinstance(dias, int) and st.session_state.elegibilidad_result is None:
        try:
            raw = verificar_elegibilidad_producto.func(numero_orden=numero, dias_desde_compra=int(dias))
        except Exception as e:
            _ecobot_logger.exception("Error en verificar_elegibilidad_producto: %s", e)
            return "Ocurrió un error verificando la elegibilidad. Intenta nuevamente."
        import json as _json
        try:
            res = _json.loads(raw)
        except Exception:
            res = {"numero_orden": numero, "dias_desde_compra": dias, "elegible": dias <= 30, "motivo": "Resultado parseado por fallback"}
        st.session_state.elegibilidad_result = res
        if res.get("elegible"):
            if not direccion:
                return "Tu orden es elegible para devolución. Por favor, indícame la direccion_cliente para generar la etiqueta."
            # Si ya hay dirección, pasamos a generar etiqueta en el siguiente bloque
        else:
            # No elegible: respondemos aquí mismo (también cubierto por Paso 1 en próximas vueltas)
            st.session_state.devolucion_fields = {"numero_orden": None, "dias_desde_compra": None, "direccion_cliente": None}
            st.session_state.elegibilidad_result = None
            st.session_state.etiqueta_result = None
            st.session_state.etiqueta_pdf = None
            return f"Tu orden {res.get('numero_orden')} no es elegible para devolución. Motivo: {res.get('motivo')}"

    # Paso 3: si es elegible y tenemos dirección, generamos etiqueta
    if st.session_state.elegibilidad_result and st.session_state.elegibilidad_result.get("elegible") and direccion:
        try:
            raw = generar_etiqueta_devolucion.func(numero_orden=st.session_state.elegibilidad_result.get("numero_orden"), direccion_cliente=direccion)
        except Exception as e:
            _ecobot_logger.exception("Error en generar_etiqueta_devolucion: %s", e)
            return "Ocurrió un error generando la etiqueta. Intenta nuevamente."
        import json as _json
        try:
            etiqueta = _json.loads(raw)
        except Exception:
            etiqueta = {"numero_orden": numero, "direccion_cliente": direccion, "guia": "ECOMERROR", "url_etiqueta": "N/A", "estado": "generada"}
        st.session_state.etiqueta_result = etiqueta
        # Generar PDF local y guardarlo en sesión (si FPDF disponible)
        has_pdf = False
        if FPDF is not None:
            try:
                pdf_bytes = _build_label_pdf(etiqueta)
                fname = f"Etiqueta_{etiqueta.get('guia', 'ECOM')}.pdf"
                st.session_state.etiqueta_pdf = {"bytes": pdf_bytes, "filename": fname}
                has_pdf = True
            except Exception as e:
                _ecobot_logger.exception("Error generando PDF de etiqueta: %s", e)
                st.session_state.etiqueta_pdf = None
        else:
            st.session_state.etiqueta_pdf = None
        # Reset para nuevo caso (conservamos etiqueta_result/pdf para el render actual)
        st.session_state.devolucion_fields = {"numero_orden": None, "dias_desde_compra": None, "direccion_cliente": None}
        st.session_state.elegibilidad_result = None
        msg_lines = [
            "Etiqueta de devolución generada:",
            f"- Orden: {etiqueta.get('numero_orden')}",
            f"- Guía: {etiqueta.get('guia')}",
        ]
        if has_pdf:
            msg_lines.append("- Descarga: usa el botón de abajo para obtener el PDF.")
        else:
            msg_lines.append("- Nota: no se pudo generar el PDF local en este entorno.")
        return "\n".join(msg_lines)

    # Si no podemos manejarlo determinísticamente, dejar al agente
    return None

# TTL por inactividad
now_ts = time()
elapsed = now_ts - float(st.session_state.last_activity_ts or 0)
if elapsed > st.session_state.ttl_minutes * 60 and st.session_state.chat_history:
    st.info("La conversación se reinició por inactividad.")
    st.session_state.chat_history = []
    # Limpiar también el flujo de devolución para evitar residuos de una sesión previa
    st.session_state.devolucion_fields = {"numero_orden": None, "dias_desde_compra": None, "direccion_cliente": None}
    st.session_state.elegibilidad_result = None
    st.session_state.etiqueta_result = None
    st.session_state.etiqueta_pdf = None
    _ecobot_logger.info("TTL expirado: chat_history borrado tras %.1f segundos", elapsed)

# Log inicial del estado de sesión
_log_session_state("startup")

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
    _update_devolucion_fields(user_input)
    st.session_state.last_activity_ts = time()
    _log_session_state("after_user_message")
    with st.chat_message("user"):
        st.markdown(user_input)

    # Manejo determinista del flujo de devolución (short-circuit)
    handled = _maybe_handle_devolucion()
    if handled is not None:
        with st.chat_message("assistant"):
            st.markdown(handled)
            # Si se generó un PDF, ofrecer botón de descarga
            if st.session_state.etiqueta_pdf and isinstance(st.session_state.etiqueta_pdf, dict):
                st.download_button(
                    label="Descargar etiqueta (PDF)",
                    data=st.session_state.etiqueta_pdf.get("bytes", b""),
                    file_name=st.session_state.etiqueta_pdf.get("filename", "Etiqueta_EcoMarket.pdf"),
                    mime="application/pdf",
                )
                # Limpiar para no repetir en respuestas futuras
                st.session_state.etiqueta_pdf = None
        st.session_state.chat_history.append(AIMessage(handled))
        st.session_state.last_activity_ts = time()
        _log_session_state("after_ai_message")
    else:
        # Historial limitado para el modelo
        max_ctx = int(st.session_state.max_ctx_messages)
        ctx_history: List = st.session_state.chat_history[-max_ctx:]

        # Invocar agente
        try:
            _log_session_state("before_agent")
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": ctx_history,
                "devolucion_context": _build_devolucion_context(),
            })
            output_text = result.get("output", "")
            if not isinstance(output_text, str) or output_text.strip() == "":
                _ecobot_logger.warning("Respuesta vacía del agente; aplicando fallback. Result=%s", result)
                output_text = "Lo siento, no pude generar una respuesta. ¿Puedes reformular o reiniciar la conversación desde la barra lateral?"
        except Exception as e:
            output_text = f"Ocurrió un error al ejecutar el agente: {e}"

        # Mostrar respuesta y guardar en historial
        with st.chat_message("assistant"):
            st.markdown(output_text)
        st.session_state.chat_history.append(AIMessage(output_text))
        st.session_state.last_activity_ts = time()
        _log_session_state("after_ai_message")
