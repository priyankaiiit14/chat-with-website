#all third party libraries are stored in langchain_community
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage # langchain schema for human and AI response
from langchain_community.document_loaders import WebBaseLoader #for website scraping, beatiful soup is used
from langchain.text_splitter import RecursiveCharacterTextSplitter # to split the documents into chunks
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv() #will make all the .env variables available in the environment


def get_vectorstore_from_url(url):
    #get the texts in documents format
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    #create a vector store for document chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_context_retriever_chain(vector_store):
    #get the relevant documents based on the entire user chat history context
    # from vector store
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input")
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    # using retrived documents return the answer to the user input

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])


    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    #create converstaion chain
    # in a production env, we can expose conversation_rag_chain as a service
    # and call it here instead of creating an instance every time
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })
    
    #response contains context, all the retrieved documents and answer
    return response['answer']
    


#App config
st.set_page_config(page_title="Chat with websites", page_icon="")
st.title("chat with websites")

#sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url =="":
    st.info("Please enter a website url")
else:
    #Persist chat history so that it does not reinitialize with each input
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot, How can I help you?",)
        ]
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
    
    

    #user input
    #https://docs.streamlit.io/library/api-reference/chat/st.chat_input
    #https://docs.streamlit.io/library/api-reference/chat/st.chat_message

    user_query = st.chat_input("Please enter your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        #debug get_context_retriever_chain
        # retreived_documents = retriever_chain.invoke({
        #     "chat_history": st.session_state.chat_history,
        #     "input": user_query
        # }
        # )
        # st.write(retreived_documents)

    #converation - format as per AI and Human instance
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):  # only display AI messages
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
