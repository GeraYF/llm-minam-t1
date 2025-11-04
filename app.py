import os
import streamlit as st
import json
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# ========================
# CARGA DE VARIABLES .ENV
# ========================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ========================
# FUNCIÓN DE PROCESAMIENTO RAG
# ========================
@st.cache_resource
def load_and_process_documents():
    """Carga, divide y crea la base de datos vectorial."""
    try:
        loader = JSONLoader(
            file_path='./data.jsonl',
            jq_schema='.texto_completo',
            text_content=False,
            json_lines=True
        )
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store

    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo `data.jsonl`.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al procesar los documentos: {e}")
        return None

# ========================
# FUNCIÓN DE LIMPIEZA
# ========================
def clear_query():
    """Limpia el campo de texto de la pregunta después de la consulta."""
    st.session_state["user_query_input"] = ""

# ========================
# INTERFAZ Y LÓGICA PRINCIPAL
# ========================
def main():
    st.set_page_config(page_title="LLM MINAM RAG", layout="wide")

    # --- Estilos CSS: Ajustes agresivos para el sidebar y tema oscuro ---
    st.markdown("""
        <style>
        /* Ajuste general para el tema oscuro (asegura consistencia) */
        .stApp {
            background-color: #0E1117; 
            color: #FAFAFA;
        }
        
        /* 1. ELIMINACIÓN AGRESIVA DE MARGENES/PADDINGS EN EL SIDEBAR */
        /* Elimina el padding de la parte superior del sidebar */
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        /* Elimina el padding interno de los elementos del sidebar (afecta a los lados) */
        [data-testid="stSidebarContent"] {
            padding-left: 0rem !important; 
            padding-right: 0rem !important; 
            padding-bottom: 1rem !important; /* Mantener un poco de espacio abajo */
        }
        
        /* 2. BANNER DE IMAGEN: OCUPAR TODO EL ESPACIO SUPERIOR DEL SIDEBAR */
        .sidebar-banner-img-container {
            width: 100%; 
            margin: 0;
            padding: 0;
            overflow: hidden; 
            box-sizing: border-box;
        }
        .sidebar-banner-img-container img {
            display: block;
            width: 100%;
            /* Altura fija para el banner: puedes ajustar este valor (vh = % de la altura de la ventana) */
            height: 250px; 
            object-fit: cover; 
            margin: 0;
            padding: 0;
            border-radius: 0;
        }
        
        /* 3. ESTILOS FORMALES PARA TEMA OSCURO */
        .response-box {
            background-color: #212121; 
            color: #e0e0e0; 
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            font-size: 16px;
        }

        /* Títulos principales y subtítulos */
        .stMarkdown h1 { color: #BBDEFB !important; }
        .stMarkdown h3 { color: #90CAF9 !important; }
        .stMarkdown h4 { color: #A5D6A7 !important; }

        /* Contenedor del texto debajo de la imagen en el sidebar */
        .sidebar-text-content {
            padding: 1.5rem 1rem; /* Añadir padding solo al texto para que no se pegue al borde */
        }
        
        /* Asegurar consistencia de colores en el sidebar */
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown span {
            color: #FAFAFA !important;
        }

        </style>
    """, unsafe_allow_html=True)


    # --- Sidebar (Columna de Referencia Visual) ---
    with st.sidebar:
        # A. BANNER DE IMAGEN SUPERIOR (Sin Espacios)
        # Usando la URL del logo que se adapta a tu diseño.
        st.markdown(
            f"""
            <div class="sidebar-banner-img-container">
                <img 
                    src="{'https://noticias.rse.pe/respaldo/wp-content/uploads/2020/10/MINAM.jpg'}" 
                    alt="MINAM Banner"
                />
            </div>
            """, 
            unsafe_allow_html=True
        )

        # B. TÍTULO Y INFORMACIÓN (Con padding interior)
        st.markdown(
            """
            <div class="sidebar-text-content">
                <h3>MINISTERIO DEL AMBIENTE (MINAM)</h3>
                <p>Asistente RAG con IA Gemini 2.5 Flash</p>
                <hr>
                <p><b>Objetivo:</b> Consultar documentación oficial proporcionada (JSONL) sobre resoluciones, nombramientos y normativas de gestión de residuos y recursos.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --- Contenido Principal (Interacción) ---
    st.markdown("<h1>🏛️ Asistente de Consulta Normativa</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Base de Conocimiento: Resoluciones del MINAM</h4>", unsafe_allow_html=True)
    st.markdown("---")


    # --- LÓGICA RAG (sin cambios) ---
    vector_store = load_and_process_documents()
    if not vector_store:
        st.error("❌ No se pudo inicializar el motor RAG.")
        return

    if not GEMINI_API_KEY:
        st.warning("⚠️ Clave GEMINI_API_KEY no encontrada en el archivo .env. Por favor, revisa tu clave de Google AI Studio.")
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=GEMINI_API_KEY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 

    prompt_template = ChatPromptTemplate.from_template("""
        Eres un asistente experto y formal en normativa del MINAM. 
        Responde con precisión y profesionalismo basándote únicamente en el CONTEXTO proporcionado. 
        Si la información no está en el contexto, indica de forma clara que no puedes responder.

        CONTEXTO: {context}
        PREGUNTA: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain_rag = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )


    # --- INTERFAZ DE PREGUNTA Y BOTONES ---
    with st.form("consulta_form"):
        st.markdown("<h3>💬 Realiza tu consulta</h3>", unsafe_allow_html=True)
        user_query = st.text_input(
            "Escribe tu pregunta:",
            placeholder="Ej: ¿Quién fue designado Director General de Gestión de Residuos Sólidos a partir del 24 de octubre de 2025?",
            key="user_query_input",
            label_visibility="collapsed"
        )
        
        col_btn = st.columns([1, 1])
        submit = col_btn[0].form_submit_button("🔍 Consultar", type="primary", use_container_width=True)
        clear_q = col_btn[1].form_submit_button("🧹 Limpiar Pregunta", on_click=clear_query, use_container_width=True)
        
        if submit:
            st.session_state['query_to_process'] = user_query
        
        if 'query_to_process' not in st.session_state:
            st.session_state['query_to_process'] = None
    
    
    # --- PROCESAMIENTO Y RESULTADOS (Fuera del formulario) ---
    if st.session_state['query_to_process']:
        query_to_process = st.session_state['query_to_process']
        
        with st.spinner("🔎 Analizando documentos y generando respuesta..."):
            response = qa_chain_rag.invoke(query_to_process)
            source_documents = retriever.invoke(query_to_process)

        st.markdown("## ") # Espacio

        st.markdown("<h3>✅ Respuesta del Asistente</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)
        
        st.markdown("## ") # Espacio

        st.markdown("<h3>📚 Documentos Fuente Consultados</h3>", unsafe_allow_html=True)
        
        for i, doc in enumerate(source_documents):
            source_title = doc.metadata.get('titulo', 'Sin título disponible')
            source_type = doc.metadata.get('tipo', 'Documento')
            st.expander(
                f"**{source_type}** | Título: *{source_title}*", expanded=False
            ).code(doc.page_content, language='text')
            
        st.session_state['query_to_process'] = None
    
# ========================
# EJECUCIÓN
# ========================
if __name__ == "__main__":
    main()