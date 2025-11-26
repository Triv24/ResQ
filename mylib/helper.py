# ------------------- Importing required functionalities ------------------------------
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import tool
from langchain_chroma import Chroma
from docx import Document

# ---------------------------- Session Initialisation ---------------------------------
def init_session():
    if "llm" not in st.session_state:
        st.session_state["llm"] = get_llm()
    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = get_embedding_model()
    if "my_store" not in st.session_state:
        st.session_state["my_store"] = get_vector_store()
    if "string_parser" not in st.session_state:
        st.session_state["string_parser"] = get_string_output_parser()
    if "json_parser" not in st.session_state:
        st.session_state["json_parser"] = get_json_output_parser()

# --------------------------- Initialising Models --------------------------------------
def get_llm():
    llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash',
        google_api_key = st.secrets["GEMINI_API_KEY"]
    )
    return llm

def get_embedding_model():
    embedding_model = GoogleGenerativeAIEmbeddings(
        model = "gemini-embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    return embedding_model

def get_vector_store():
    chroma_client = Chroma(
        embedding_function=st.session_state.embedding_model,
        persist_directory="Textbooks",
        collection_name="textbook_data"
    ) 
    return chroma_client

def get_string_output_parser ():
    return StrOutputParser()

def get_json_output_parser ():
    return JsonOutputParser()

# ---------------------- Tools for the Agent -------------------------------------------
@tool
def get_model_question_paper(question_papers : list) :
    """
    This tool will create a model question paper from given previous year question papers and the answers to it from the textbook context in word format.
    """

@tool
def explain_concepts(textbook) :
    """
    This tool will explain the doubts regarding the concepts which students find difficult to understand. It will explain it from the context of knowledgebase vectoDB provided.
    """

@tool
def summarise_github_repo (url):
    """
    This tool will summarise the contents of the github repo from provided url of the repo.
    """

@tool
def solve_assignment(assignment) :
    """
    This tool will solve the assignments provided by the student by context retrieval from the textbook which is present in the knowledge base in a vectorDB
    """

# --------------------- Helper functions for the tools ---------------------------------

def generate_docx():
    pass

def create_knowledge_base(file):
    pass

# ----------------------------- PROMPTS --------------------------------------------------

model_question_prompt = PromptTemplate(
    template="""
    
    """,
    input_variables=[]
)



# ------------------------------- RAG essentials -----------------------------------------
def add_knowledge_base():
    pass

def add_question_papers():
    pass

