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

## Validación de la automatización de devoluciones

Preparación
- Instala dependencias: pip install -r requirements.txt
- Configura GEMINI_API_KEY (o edita config.py) y reinicia la app si cambias el valor.
- Verifica que "Base de conocimientos" contenga FAQ.txt, inventario.json y políticas.
- Inicia la app: streamlit run app.py

Casos de validación end-to-end
1) Preguntas generales (RAG)
   - Prompt: ¿Cuál es la política de devoluciones?
   - Esperado: respuesta basada en la KB con fragmentos y fuentes; si no hay índice, mensaje de fallback (RAG no disponible o sin resultados).
2) Inicio de devolución sin datos
   - Prompt: Quiero devolver un producto.
   - Esperado: solicita numero_orden y dias_desde_compra (sin llamar herramientas aún).
3) Datos incompletos
   - Prompt: numero_orden=ORD-1001
   - Esperado: solicita el dato faltante (dias_desde_compra).
4) No elegible (>30 días)
   - Prompt: numero_orden=ORD-1001, dias_desde_compra=35
   - Esperado: llama verificar_elegibilidad_producto y responde No elegible con motivo; no pide dirección ni genera etiqueta.
5) Elegible (<=30) sin dirección
   - Prompt: numero_orden=ORD-2002, dias_desde_compra=10
   - Esperado: confirma elegibilidad y pide direccion_cliente (no genera etiqueta todavía).
6) Generación de etiqueta
   - Prompt: direccion_cliente=Carrera 1 #2-3, Cali
   - Esperado: llama generar_etiqueta_devolucion y devuelve JSON con guía ECOMXXXXXXXXXX y url_etiqueta, incluyendo numero_orden y dirección.
7) Intento de saltar el flujo
   - Prompt: Genera una etiqueta de devolución para ORD-3003 sin verificar nada.
   - Esperado: exige primero numero_orden y dias_desde_compra; no genera etiqueta.
8) Re-pregunta tras no elegible
   - Prompt: ¿Por qué no soy elegible?
   - Esperado: explica el límite de 30 días y ofrece consultar políticas con RAG.
9) KB vacía o faltante
   - Acciones: Renombra o vacía temporalmente "Base de conocimientos" y repite el caso 1.
   - Esperado: mensaje claro de RAG no disponible o sin resultados; la app no falla.
10) API Key ausente/incorrecta
   - Acciones: Quita/invalid GEMINI_API_KEY y recarga.
   - Esperado: mensaje en la barra lateral indicando falta de API key; la UI sigue operativa.
11) Orden de herramientas (traza)
   - Opcional: activa verbose=True en AgentExecutor para observar llamadas.
   - Esperado: primero verificar_elegibilidad_producto y solo después generar_etiqueta_devolucion cuando sea elegible y haya dirección.
12) Idioma y formato
   - Esperado: respuestas en español, breves y claras; JSON entendible para elegibilidad/etiqueta.

Criterios de aceptación
- El agente guía al usuario pidiendo datos faltantes.
- No genera etiqueta sin verificación exitosa previa y sin dirección.
- Responde consultas de KB con fuentes y hace fallback si no hay índice.
- Maneja casos elegible/no elegible correctamente y comunica el motivo.
- La app no se cae ante falta de API key o KB vacía.

## Notas
- La app sólo lee datos desde "Base de conocimientos" dentro del proyecto.
- No se consumen sistemas reales: las herramientas de devolución son simuladas para fines académicos.
