from llama_index import SimpleDirectoryReader, LLMPredictor, Document , GPTSimpleVectorIndex
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from langchain import OpenAI
import streamlit as st
import requests
import pymongo
import cloudinary
import cloudinary.uploader
# My OpenAI Key
import os
from dotenv import load_dotenv
import json
from langchain.llms import OpenAIChat

 
os.environ['OPENAI_API_KEY'] = "YOUR API KEY HERE openai"


# Define the Streamlit app
def app():
    load_dotenv()
    # Set up MongoClient credentials
    client = pymongo.MongoClient("link to your mongo db")
    
    # Set up Cloudinary credentials
    cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
    )
    
    #Define the llm predictor
    llm_predictor = LLMPredictor(llm=OpenAIChat(temperature=0, model_name="gpt-3.5-turbo"))

    # Set the title and description of the web page
    st.markdown("All you need is to upload a PDF file !")
 
    # Create a file uploader widget to allow the user to upload an audio file
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
 
   
    # When the user clicks the "Submit" button, create the index
    if st.button("Submit") and uploaded_file is not None:
        # Read the contents of the uploaded file
        file_contents = uploaded_file.read()
        progress_bar = st.progress(10)


        ## You may need to Add a pdf reader here using llama index lib
        # you can test different indexing techniques
        doc=''
        # NOTE: can take a while! 
        progress_bar.progress(50)
        index = GPTKnowledgeGraphIndex(
            [uploaded_file], 
            chunk_size_limit=1200, 
            max_triplets_per_chunk=5,
            llm_predictor=llm_predictor,
            include_embeddings=True
        )
        progress_bar.progress(80)
 
        # Access a database and collection
        db = client.mydb
        collection = db.mycollection
 
        
        index_kg=index.save_to_dict()

        # save the index to mongodb
        result=collection.insert_one(index_kg)
        st.session_state['key']= result.inserted_id
 
        
        progress_bar.progress(100)
        st.write("Great! now you can go to the chatPDF page!")

 


        
