# ResQ
ResQ, as its name suggests is a "rescue" project for students pursuing Engineering. As we all know how we engineering students always play with the deadlines during our exams, assignment submissions, project submissions, etc. But ResQ is here to rescue us all from actually failing to meet the deadlines.  

## What does ResQ do ?
  - Helps students predict and generate a model question paper by analysing the previous years question papers
  - It provides a downloadable word file of the model question paper for the students to be able to print it
  - Explains concepts to students from the context of the textbook
  - Solves the assignments questions according to the student needs
  - searches for resources or students on the web

## Techstack :
### Backend :
- Python (logic)
- Langchain
### Frontend
- Streamlit

## Functionalities Used :

### Langchain :
- Document Loaders
- Text Splitters
- Google Genai Chat model
- Prompts
- Tools
- Agents
- LLM Model
- Tavily

### Streamlit :
- Streamlit basic UI components
- Streamlit markdown
- Streamlit Tabs
- File Uploaders
- Succes messages
- Warning messages
- Session States

## Tools used in ResQ :

### Built-in tools :
- `Tavily` for web search

### Custom tools :
- `Questions extracter` to extract pure question content from a noisy PDF loaded content
- `Model question paper generator` to predict the questions and also generate a word file for that
- `Relevant chunks extracter` to get relevant chunks from the knowledge base
- `Concepts explainer` to explain the concepts based on the extracted relevant chunks

## Setup codebase :
1. clone the repository
2. Install all the dependencies listed in `requirements.txt` file
3. When you want to run the app - type the following comand in terminal :
   ```bash
   streamlit run app.py
   ```
   - Ensure that you are in the cloned directory before executing the command
