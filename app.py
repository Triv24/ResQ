import streamlit as st
import tempfile
from mylib.helper import load_pdf, extract_questions, get_model_questions, get_docx, get_chunks, store_in_db, get_relevant_chunks, explain_concept
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Introductory content
st.markdown("""
# ResQ's here to Rescue! 🫡
            
## What can ResQ do for You :
            
    📜 Generate a predictive model Question Paper based on previous question papers
            
    🧑🏻‍🏫 Explain concepts from the context of your textbook
            
    🎯 Solve the assignments by learning from textbook
""")
st.write("\n\n")

# Defining streamlit tabs
model_qp, concepts, assignments = st.tabs(["Generate Model QP", "Explain Concepts", "Solve Assignments"])

# ----------------------------------------------------------------------------------------------------------------------------
#  Model Question Paper Generator 
# ----------------------------------------------------------------------------------------------------------------------------

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
                    "content" : f"Please predict the model question paper for the question papers content : {loaded_qp_content}. first extract the clear questions from the content using a tool, then use that extracted questions for analysis and prediction for model question paper. Ultimately i need the file path of the word document that contains the model question papers. Just give me the final file path in string format. Do not add unnecessary words in output, the output string should wholly represent the file path only."
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

# ----------------------------------------------------------------------------------------------------------------------------
#  Concepts Explaining Assistant
# ----------------------------------------------------------------------------------------------------------------------------

with concepts:
    st.write("Please upload your Textbook : ")
    file = st.file_uploader("Upload here...", type=".pdf")
    if file is not None :

        if "file" not in st.session_state:
            with st.spinner("Loading contents will take time as the file is very large...") :
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp :
                    tmp.write(file.getbuffer())
                    temp_pdf_path = tmp.name
                st.session_state.text = load_pdf(temp_pdf_path)
            st.session_state.file = temp_pdf_path
            st.success("Loaded content successfully...")


            
            with st.spinner("Adding to knowledge base..."):
                text = st.session_state.text
                if text :
                    chunks = get_chunks(text)
                    store_in_db(chunks)
            st.success("Stored succesfully")
            st.session_state.stored = True

    
        if "llm" not in st.session_state:
            st.session_state.llm = ChatGoogleGenerativeAI(
                model = "gemini-2.5-flash",
                google_api_key = st.secrets["GEMINI_API_KEY"]
            )
        
        if "agent" not in st.session_state:
            st.session_state.agent = create_agent(
                model = st.session_state.llm,
                tools=[get_relevant_chunks, explain_concept],
                system_prompt="You are a helpful student assistant and a master explainer who gives suitable real world examples just by taking the contexts from textbook. Just ensure the final output is in markdown format. Make good use of markdown visual elements."
            )

        st.session_state.query = st.text_input("What concept would you like to learn ?")
        if st.session_state.query :
            with st.spinner("Generating response...") :
                agent = st.session_state.agent
                query = st.session_state.query
                
                prompt = f"You are a helpful study assistant. You are given a user query and you have to first collect the relevant chunks from the vector store and then based on the relevant chunks, you have to explain the concepts mentioned in the query. give the final output in string of markdown format. If you do not get any relevant chunks you can answer based on your own knowledge about the subject.\nquery : {query}"

                response = agent.invoke(
                    {
                        "messages" : [
                            {
                                "role" : "user",
                                "content" : prompt
                            }
                        ]
                    }
                )
            
            st.markdown(response["messages"][-1].content[0]['text'])

# ----------------------------------------------------------------------------------------------------------------------------
#  Assignments Helper
# ----------------------------------------------------------------------------------------------------------------------------

with assignments :
    st.write("\n")
    st.session_state.selected = st.selectbox("What is the file-type of assignments ?", ["PDF", "Plain Text"])

    if st.session_state.selected == "PDF" :
        file = st.file_uploader("Upload the PDF here...", type = ".pdf")
        if file is not None:
            if "file" not in st.session_state:
                with st.spinner("Loading contents will take time as the file is very large...") :
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp :
                        tmp.write(file.getbuffer())
                        temp_pdf_path = tmp.name
                        st.session_state.text = load_pdf(temp_pdf_path)
                st.session_state.file = temp_pdf_path
                st.success("Loaded content successfully...")

    elif (st.session_state.selected == "Plain Text") :
        st.session_state.text = st.text_area("Paste the text Here...")
        if "text" not in st.session_state or st.session_state.text.strip() == "":
            st.warning("Please Insert text")
    
    if "text" in st.session_state :
        st.session_state.words = st.slider("Please set the approximate number of words for the size of the answers", min_value=50, max_value=500, step=50,)

    if "words" in st.session_state :
        with st.spinner("Processing...") :
            if "llm" not in st.session_state:
                st.session_state.llm = ChatGoogleGenerativeAI(
                    model = "gemini-2.5-flash",
                    google_api_key = st.secrets["GEMINI_API_KEY"]
                )

            response = st.session_state.llm.invoke(
                    f"""
    You are a helpful study assistant. Please solve all the given questions in standard answer formats each in about {st.session_state.words} words.
    Directly start from 1st question.
    Do not repeat the qeustoins in response.
    Just give the sequence number of question and then immediately the answer.
                    Questions : {st.session_state.text}
                    """
                )

            st.markdown(response.content)