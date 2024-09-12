__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

with_message_history = create_llm_chain()

# 4. Test the app with streamlit
def main():
    st.title("Eagles - an AI-Powered Academic Advisor :bird:")
    
    # Initialize session state to store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
