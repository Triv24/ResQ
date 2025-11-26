# ------------------- Importing required functionalities ------------------------------
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# --------------------------- Initialising Models --------------------------------------
def get_llm():
    llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash',
        google_api_key = st.secrets["GEMINI_API_KEY"]
    )
    return llm

def get_db():
    pass 

def get_output_parser ():
    return StrOutputParser()

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

# ----------------------------- PROMPTS --------------------------------------------------

# ------------------------------- RAG essentials -----------------------------------------
def add_knowledge_base():
    pass

def add_question_papers():
    pass

