import streamlit as st
import tempfile
import os
from mylib.helper import init_session, get_llm, load_pdf, extract_questions, get_model_questions, get_docx
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

init_session()


st.markdown("""
# ResQ's here to Rescue! 🫡
            
## What can ResQ do for You :
            
    📜 Generate a predictive model Question Paper based on previous question papers
            
    🧑🏻‍🏫 Explain concepts from the context of your textbook
            
    🎯 Solve the assignments by learning from textbook
""")

st.write("\n\n")

model_qp, concepts, assignments = st.tabs(["Generate Model QP", "Explain Concepts", "Solve Assignments"])

with model_qp:
    st.write("How many Question Papers are there ? ")
    no_of_qp = st.number_input("Input number here..", value=1, min_value=1, max_value=5)

    st.session_state.qp_content = ""
    for i in range(no_of_qp):
        file = st.file_uploader(f"Upload the {i}{"st" if i == 1 else "th"} QP :")
        if file is not None:
            with st.spinner("Loading the contents...") :
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp :
                    tmp.write(file.getbuffer())
                    temp_pdf_path = tmp.name
                st.session_state.qp_content += load_pdf(temp_pdf_path)
            st.success("Loaded contents")

    

    loaded_qp_content = st.session_state.qp_content
    if loaded_qp_content :
        with st.spinner("Starting Prediction..."):

            llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        google_api_key = st.secrets["GEMINI_API_KEY"]
    )
            st.session_state.agent = create_agent(
                model=ChatGoogleGenerativeAI(
                    model = "gemini-2.5-flash",
                    google_api_key = st.secrets["GEMINI_API_KEY"]
                ),
                tools = [extract_questions, get_model_questions, get_docx],
                system_prompt="You are a helpful assistant for students"
            )

            agent = st.session_state.agent
            response = agent.invoke({
                "messages" : [{
                    "role" : "user",
                    "content" : f"Please predict the model question paper for the question papers content : {loaded_qp_content}. Ultimately i need the file path of the word document that contains the model question papers. Just give me the final file path in string format. Do not add unnecessary words in output, the output string should wholly represent the file path only."
                }]
            })
            
            file_path = response["messages"][-1].content[0]["text"]

            with open(file_path, "rb") as file:
                st.download_button(
                    label="Download Saved File",
                    data=file.read(), # Read the file's bytes into memory here
                    file_name="Model_Questions.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

