import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from docx import Document
import tempfile

def init_session():
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            google_api_key = st.secrets["GEMINI_API_KEY"]
        )

def load_pdf(file):
    loader = PyPDFLoader(file)
    docs = loader.load()

    full_text = ""
    for doc in docs:
        full_text += doc.page_content

    return full_text

@tool
def extract_questions(qp):
    """
    This tool will take the content that is loaded from the pdf file of question paper which contains a lot of other things.
    This tool extracts only the questions from the loaded content
    """
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        google_api_key = st.secrets["GEMINI_API_KEY"]
    )

    response = llm.invoke(
        """
        please extract all the questions from the contents of a question paper loaded from a pdf file. 
        Output should be clear, concise and to the point.
        Avoid any greetings, just provide the questions.
        start directly from Q1
        ---
        contents : {qp}
        """
    )

    return response.content

@tool
def get_model_questions(question_sets : str) -> str :
    """
    This tool takes the extracted questions in form of string and then uses an llm to analyse the patterns of questions in the question papers and then predicts all the question for model question paper.
    """
    llm = get_llm()

    response = llm.invoke(
        f"""
    You are an expert exam paper analyst and academic question setter with deep experience in identifying patterns across previous examinations.
    ---
    INSTRUCTIONS
    Carefully read and analyze the full contents of the previous question papers provided.

    Identify recurring patterns, frequently tested concepts, question styles, difficulty levels, and the distribution of marks.

    Based on this analysis, estimate and construct a complete model question paper that closely reflects what is most likely to appear in the upcoming exam.

    Ensure the model paper:

    Follows the same structure and format as typical past exams

    Covers all major topics proportionately

    Uses clear, academic language

    Includes appropriate sectioning (Part A, Part B, etc., if relevant)
    ---
    CONTEXT

    Previous question paper content:
    {question_sets}

    ---
    OUTPUT

    A well-structured string, all questions for model question paper that represents the best possible estimation of upcoming exam questions.
    directly start from Q1
    """
    )

    return response.content

@tool
def get_docx(model_questions : str) :
    """
    This tool takes the predicted model questions as an arguement and generates a file path of a word document containing these question.
    """
    document = Document()

    document.add_heading("Model Question Paper", level=2)
    
    document.add_paragraph(model_questions)

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        file_path = tmp.name
        document.save(file_path)
    
    return file_path


def get_llm():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        google_api_key = st.secrets["GEMINI_API_KEY"]
    )

    return llm

