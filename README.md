# Proyecto Final: EcoBot (proyecto_final_ecomarket)

Asistente web con agente de IA que usa RAG sobre la carpeta "Base de conocimientos" y herramientas simuladas para gestionar devoluciones.

## Estructura
- Base de conocimientos/
  - inventario.json
  - politicas de devoluciones.txt
  - FAQ.txt
- app.py (Streamlit, interfaz de chat y agente)
- agent_tools.py (herramientas del agente: RAG + simuladas de devoluciones)
- rag_module.py (carga, chunking y retriever FAISS)
- config.py (configuración de rutas, modelos y parámetros RAG)
- requirements.txt

## Fase 1: Diseño de Arquitectura del Agente
- Marco: LangChain para agentes y herramientas. Justificación: continuidad con el Taller 2 (migración directa del RAG), ecosistema maduro y soporte de herramienta de llamada estructurada.
- Interfaz: Streamlit por simplicidad y despliegue rápido.
- LLM: Gemini (gemini-2.5-flash-lite) vía langchain-google-genai con temperatura 0 para respuestas consistentes.
- RAG:
  - Ingesta: archivos .txt y .json desde "Base de conocimientos".
  - Chunking: RecursiveCharacterTextSplitter (800, solapamiento 150).
  - Embeddings: models/text-embedding-004 (Google Generative AI).
  - Índice: FAISS en memoria; fallback a Chroma si FAISS no está disponible; retriever con k=3 (configurable).
- Herramientas (Tools):
  1) buscar_en_base_de_conocimiento(query): usa el retriever semántico y retorna fragmentos con fuentes.
  2) verificar_elegibilidad_producto(numero_orden, dias_desde_compra): simula la política (<=30 días) y devuelve JSON.
  3) generar_etiqueta_devolucion(numero_orden, direccion_cliente): genera guía simulada y URL de etiqueta en JSON.
- Reglas del agente (prompt del sistema):
  - Priorizar RAG para preguntas generales.
  - Exigir numero_orden y dias_desde_compra antes de verificar elegibilidad.
  - Nunca generar etiqueta sin verificar elegibilidad primero.

- Diagrama de flujo del proceso de devolución:
  - Usuario solicita devolución
    -> ¿proporcionó numero_orden y dias_desde_compra?
      - No: solicitar los datos faltantes (no llamar herramientas todavía).
      - Sí: llamar herramienta verificar_elegibilidad_producto.
        -> ¿la orden es elegible (<=30 días)?
          - No: informar no elegible y el motivo; ofrecer consulta de políticas con RAG.
          - Sí: ¿proporcionó direccion_cliente?
            - No: solicitar direccion_cliente.
            - Sí: llamar herramienta generar_etiqueta_devolucion
                 -> devolver JSON con guía y URL de etiqueta al usuario.

## Diagrama de arquitectura (alto nivel)
```text
Usuario
  |
  v
+---------------------------+
| Streamlit UI (app.py)     |
| - chat_input/chat_output  |
| - st.session_state (hist) |
+-------------+-------------+
              |
              v
+---------------------------+
| AgentExecutor (LangChain) |
| - create_tool_calling_agent
| - Prompt con reglas       |
+------+------+-------------+
       |    |
       |    +------------------------------+
       |                                   |
       v                                   v
+-------------------+            +---------------------------+
| LLM (Gemini)      |            | Tools (agent_tools.py)    |
| gemini-2.5-flash- |            | - buscar_en_base...       |
| lite              |            | - verificar_elegibilidad  |
+---------+---------+            | - generar_etiqueta        |
          |                      +------------+--------------+
          |                                   |
          |                                   v
          |                         +-------------------------+
          |                         | RAG (rag_module.py)     |
          |                         | - load kb files         |
          |                         | - chunking              |
          |                         | - embeddings:           |
          |                         |   models/text-embedding-004
          |                         | - Vector store:         |
          |                         |   FAISS | Chroma (fb)   |
          |                         +------------+------------+
          |                                      |
          |                                      v
          |                         +-------------------------+
          |                         | Base de conocimientos   |
          |                         | (.txt, .md, .json)      |
          |                         +-------------------------+
          |
          +---- Respuestas del LLM y de Tools -> Agent -> UI
```

## Fase 2: Implementación y Conexión de Componentes
- Migración del notebook RAG (carga, chunking, embeddings FAISS) a rag_module.py.
- Herramientas definidas en agent_tools.py y expuestas en all_tools.
- Agente construido con create_tool_calling_agent y ejecutado con AgentExecutor.
- Historial de chat gestionado con st.session_state.

## Fase 3: Análisis Crítico y Propuestas de Mejora
- Riesgos de seguridad/ética:
  - Inyección de prompts: mitigar con instrucciones firmes y validación de entradas (p.ej., límites de longitud, patrones no permitidos).
  - PII: evitar registrar datos sensibles; anonimizar campos como dirección y número de orden en logs de producción.
  - Uso de herramientas: el agente sólo llama herramientas permitidas; nunca escribe fuera de la app.
- Monitoreo y observabilidad:
  - Logging a nivel de agente y herramienta (verbose=True en AgentExecutor). Puede integrarse con soluciones como Streamlit logger, OpenTelemetry o un SIEM.
  - Alertas al detectar errores repetidos (p.ej., fallos de RAG o múltiples etiquetas no elegibles).
- Propuestas de mejora:
  - Agente CRM para actualizar datos del cliente, con autenticación y autorización.
  - Autenticación de usuarios en la UI (p.ej., streamlit-auth).
  - Persistencia del índice FAISS en disco y jobs de reindexación.
  - Re-ranking con modelos cross-encoder para mejorar precisión.

## Fase 4: Despliegue e Instrucciones
1) Requisitos e instalación:
   - Python 3.10+
   - pip install -r requirements.txt
2) Configuración:
   - Exporta GEMINI_API_KEY como variable de entorno o edita config.py.
3) Ejecución:
   - streamlit run app.py
4) Uso:
   - Consulta políticas e inventario con preguntas naturales.
   - Para devoluciones, proporciona numero_orden, dias_desde_compra y luego direccion_cliente.

## Notas
- La app sólo lee datos desde "Base de conocimientos" dentro del proyecto.
- No se consumen sistemas reales: las herramientas de devolución son simuladas para fines académicos.
