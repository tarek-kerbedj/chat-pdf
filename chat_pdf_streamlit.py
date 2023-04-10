import streamlit as st
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ChatVectorDBChain,ConversationalRetrievalChain
from langchain.document_loaders import PDFMinerLoader
from io import StringIO,BytesIO 
from streamlit_chat import message as st_message
from typing import List, Dict, Any
from langchain.docstore.document import Document
import itertools
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain import OpenAI, VectorDBQA
from utils import parse_pdf,text_to_docs


os.environ["OPENAI_API_KEY"]=st.secrets.OPENAI_API_KEY
#API=st.secrets.OPENAI_API_KEY

#os.environ["OPEN_API_Key"]=st.secrets["OPEN_API_Key"]
st.set_page_config(page_title='ChatPDF',page_icon="http://aidevlab.com/wp-content/uploads/2023/03/cropped-AI-Derivatives_FF-04.png")

## centering the title
col1,col_center,col3 = st.columns(3)
with col_center :
    st.title('Chat-PDF 🗎')

if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory=[]

## it supports multiple PDFs
files=st.file_uploader("Choose a file",accept_multiple_files=True,type=['pdf'])

def generate_answer():
    """ this function will handle parsing the document , turning it into embeddings and the query"""
  
    user_query = st.session_state.input_text
    if user_query.strip()=='':
        st.warning('user query cant be empty, Please type something in the text box')
        return
       

    
    with st.spinner("Indexing document... This may take a while⏳"):
        # parsing the uploaded files
        parsed_files=[parse_pdf(f) for f in files]

        # flattening the list of files         
        merged_docs = list(itertools.chain(*parsed_files))
        # turning the string into lang chain's Document format
        docs=text_to_docs(merged_docs)
        embeddings = OpenAIEmbeddings()
        # create embeddings
        vectordb = Chroma.from_documents(docs, embeddings, collection_name="collection")
       # pdfqa = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", vectorstore=vectordb)
        #pdfqa=a = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever())
        pdfqa=ConversationalRetrievalChain.from_llm(OpenAI(temperature=0),vectordb.as_retriever(),verbose = True)
        #pdfqa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.7, model_name="gpt-3.5-turbo"),
         #                           vectordb, return_source_documents=True)
        
    # generate the answer
    #answer=pdfqa.run(user_query)
 
 
    answer = pdfqa({"question": user_query, "chat_history":[]})

    # save the exchanged messages
   # st.session_state.memory.append((user_query,answer['answer']))

    st.session_state.history.append({"message": user_query, "is_user": True})
    st.session_state.history.append({"message": answer['answer'], "is_user": False})
   

if  files != []:
    # trigger if the user uploads a file
    for chat in st.session_state.history:
        # unpacking the messages stored 
        st_message(**chat) 
    
    # create some space when there is no chat to display
    if st.session_state.history==[]:

        for i in range(0,15):
            st.text(" ") 

    text_box=st.text_input("Talk to your pdf", key="input_text", on_change=generate_answer,placeholder="does this assignment require C++ knowledge ?")
    
    # centering the button
    col1, col2, col_center , col4, col5 = st.columns(5)
    with col_center :
        rerun = st.button('Re-run',help="this button will delete your chat history and prompt you to create a new one")
        if rerun:
            st.session_state.history=[]
            st.experimental_rerun()
          

    
    
   
 
       
    
