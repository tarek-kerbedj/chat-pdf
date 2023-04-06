import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import base64
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import PDFMinerLoader
from pypdf import PdfReader
from io import StringIO
from io import BytesIO
from streamlit_chat import message as st_message
from typing import List, Dict, Any
from langchain.docstore.document import Document
import re
import itertools
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain import OpenAI, VectorDBQA
from utils import parse_pdf,text_to_docs

os.environ["OPENAI_API_KEY"] = "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"
st.set_page_config(page_title='ChatPDF', page_icon="http://aidevlab.com/wp-content/uploads/2023/03/cropped-AI-Derivatives_FF-04.png")
st.title('Chat-PDF 🗎')

if "history" not in st.session_state:
    st.session_state.history = []

## it supports multiple PDFs
files=st.file_uploader("Choose a file",accept_multiple_files=True,type=['pdf'])

def generate_answer():
    """ this function will handle parsing the document , turning it into embeddings and the query"""
    user_query = st.session_state.input_text
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
        pdfqa = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", vectorstore=vectordb)
    # generate the answer
    answer=pdfqa.run(user_query)
    # save the exchanged messages
    st.session_state.history.append({"message": user_query, "is_user": True})
    st.session_state.history.append({"message": answer, "is_user": False})
    

if  files != []:
    # trigger if the user uploads a file
    for chat in st.session_state.history:
        # unpacking the messages stored 
        st_message(**chat) 
    text_box=st.text_input("Talk to your pdf", key="input_text", on_change=generate_answer,placeholder="does this assignment require C++ knowledge ?")
    
   
 
       
    
