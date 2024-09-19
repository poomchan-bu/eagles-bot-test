# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
st.set_page_config(page_title="Eagles", page_icon=":bird:")

from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import AIMessage
import pandas as pd

# 1. Create vectorized database from .csv data
@st.cache_resource
def create_vectorstore():
    loader = CSVLoader(file_path="courses.csv", encoding="utf-8-sig")
    documents = loader.load()
    return Chroma.from_documents(documents, OpenAIEmbeddings())

vectorstore = create_vectorstore()

# 2. Create the chain
retriever = vectorstore.as_retriever()

@st.cache_resource
def create_llm_chain():
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-3.5-turbo",
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    system_prompt = """
    You are a student advisor
    
    You will give the class information based on the question that a student asks.
    Be aware, students usually call the course as department+course_id such as CS546.

    Context:
    "{context}"
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    parser = StrOutputParser()

    rag_chain = prompt | llm | parser

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(rag_chain, 
                                                    get_session_history,
                                                    input_messages_key="input"
                                                    )
    
    return with_message_history

initial_message = AIMessage(content="Welcome to the academic advising session! I'm Buan. What's your name?")

@st.cache_resource
def create_reception_chain():
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4o",
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    system_prompt = """
    I am a student advisor and you are helping me just to get some information from the student
    
    You need to know the following information from the student
    1. The student's name. Make sure this is not null.
    2. The courses that the student has already taken. The possible courses are in the course catalog.
    3. The number of courses the student want to take in the next semester. Make sure this is not null.
    4. The one concentration that the student wants. The possible options are: 
    "data science", "web development", "cyber security", "artificial intelligence", and "software engineering"
    Make sure this is not null.
    
    Keep asking the student to get all the information above but in a kind and natural way.
    Do NOT proceed to the next step no matter what if you still don't get the required information.
    
    If you are confident that you have all the information,
    output it in the following format and not saying anything else:
    The number in the bracker matches the information number from above.
    student_name = [1]
    course_taken = [2]
    num_course = [3]
    concentration = [4]

    Course catalog:
    course_id, course_name, credit
    CS521,Information Structures with Python,4.0
    CS526,Data Structure and Algorithms,4.0
    CS622,Advanced Programming Techniques,4.0
    CS665,Software Design and Patterns,4.0
    CS673,Software Engineering,4.0
    CS682,Information Systems Analysis and Design,4.0
    CS601,Web Application Development,4.0
    CS602,Server-Side Web Development,4.0
    CS633,"Software Quality, Testing, and Security Management",4.0
    CS634,Agile Software Development,4.0
    CS664,Artificial Intelligence,4.0
    CS669,Database Design and Implementation for Business,4.0
    CS677,Data Science with Python,4.0
    CS683,Mobile Application Development with Android,4.0
    CS701,Rich Internet Application Development,4.0
    CS763,Secure Software Development,4.0
    CS767,Advance Machine Learning and Neural Networks,4.0
    CS101,Computers and Their Applications,4.0
    CS200,Introduction to Computer Information Systems,4.0
    CS201,Introduction to Programming,4.0
    CS231,Programming with C++,4.0
    CS232,Programming with Java,4.0
    CS248,Discrete Mathematics,4.0
    CS300,Introduction to Software Development,4.0
    CS341,Data Structure with C++,4.0
    CS342,Data Structures with Java,4.0
    CS382,Information Systems for Management,4.0
    CS401,Introduction to Web Application Development,4.0
    CS422,Advanced Programming Concepts,4.0
    CS425,Introduction to Business Data Communications and Networks,4.0
    CS432,Introduction to IT Project Management,4.0
    CS469,Introduction to Database Design and Implementation for Business,4.0
    CS472,Computer Architecture,4.0
    CS473,Introduction to Software Engineering,4.0
    CS495,Directed Study,4.0
    CS496,Directed Study,4.0
    CS506,Internship in Computer Science,
    CS520,Information Structures with Java,4.0
    CS532,Computer Graphics,4.0
    CS535,Computer Networks,4.0
    CS544,Foundations of Analytics and Data Visualizaion,4.0
    CS546,Introduction to Probability and Statistics,4.0
    CS550,Computational Mathematics for Machine Learning,4.0
    CS555,Foundation of Machine Learning,4.0
    CS561,Financial Analytics,4.0
    CS566,Analysis of Algorithms,4.0
    CS570,Biomedical Sciences and Health IT,4.0
    CS575,Operating Systems,4.0
    CS579,Database Management,4.0
    CS580,Health Infomatics,4.0
    CS581,Health Information Systems,4.0
    CS584,Ethical and Legal Issues in Healthcare Informatics,4.0
    CS593,Special Topics,4.0
    CS599,Biometrics,4.0
    CS625,Business Data Communication and Networks,4.0
    CS632,Information Technology Project Management,4.0
    CS662,Computer Language Theory,4.0
    CS674,Database Security,4.0
    CS684,Enterprise Cybersecurity Management,4.0
    CS685,Network Design and Management,4.0
    CS688,Web Mining and Graph Analytics,4.0
    CS689,Designing and Implementing a Data Warehouse,4.0
    CS690,Network Security,4.0
    CS693,Digital Forensics and Investigations,4.0
    CS694,Mobile Forensics and Security,4.0
    CS695,Cybersecurity,4.0
    CS697,Special Topics in Computer Science,4.0
    CS699,Data Mining,4.0
    CS775,Advanced Networking,4.0
    CS777,Big Data Analytics,4.0
    CS779,Advance Database Management,4.0
    CS781,Advanced Health Informatics,4.0
    CS782,IT Strategy and Management,4.0
    CS783,Enterprise Architecture,4.0
    CS789,Cryptography,4.0
    CS793,Special Topics in Computer Science,4.0
    CS795,Directed Study,4.0
    CS796,Directed Study,4.0
    CS799,Advance Cryptography,4.0
    CS766,Deep Reinforcement Learning,4.0
    CS787,Adversarial Machine Learning,4.0
    CS788,Generative AI,4.0
    CS790,Computer Vision in AI,4.0
    CS810,Master's Thesis in Computer Science,4.0
    CS811,Master's Thesis in Computer Science,4.0
    CS703,Network Forensics,4.0
    CS635,Network Media Technologies,4.0
    
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    parser = StrOutputParser()

    rag_chain = prompt | llm | parser

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            history = InMemoryChatMessageHistory()
            history.add_message(initial_message)
            store[session_id] = history
            # st.session_state.messages = [{"role": "assistant", "content": initial_message.content}]
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(rag_chain, 
                                                    get_session_history,
                                                    input_messages_key="input"
                                                    )
    
    return with_message_history

with_message_history = create_reception_chain()

# 4. Test the app with streamlit
def main():
    st.title("Eagles - an AI-Powered Academic Advisor :bird:")
    
    # Initialize session state to store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": initial_message.content}]

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input section
    if user_input := st.chat_input("Enter your message"):
        # Add user message to the session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call the retriever and LLM
        config = {"configurable": {"session_id": "001"}}
        context = retriever.invoke(user_input)
        response = with_message_history.invoke(
            {"input": user_input, "context": context},
            config=config,
        )

        # Add bot response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == '__main__':
    main()
