from llama_index import SimpleDirectoryReader, LLMPredictor, Document , GPTSimpleVectorIndex
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex

from langchain.llms import OpenAIChat
import streamlit as st

import pymongo
from bson import ObjectId
from dotenv import load_dotenv


import os
# My OpenAI Key
os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"

 
# Define the Streamlit app
def app():
    load_dotenv()
 
    # Set up MongoClient credentials
    client = pymongo.MongoClient("your link")
    # Define the llm predictor:
    llm_predictor = LLMPredictor(llm=OpenAIChat(temperature=0, model_name="gpt-3.5-turbo"))
    db = client.mydb
    collection = db.mycollection

    # Initialization
    if 'key' not in st.session_state:
      st.write("Please go to the first page and upload your audio file!")
    # get the file with a specific ID
    else:
      file_id = ObjectId(st.session_state['key'])
      file = collection.find_one({"_id": file_id})

      st.title("Enter Text Page")
      prompt = st.text_input("Ask your question")

      progress_bar = st.progress(0)
      index = GPTKnowledgeGraphIndex.load_from_dict(file, llm_predictor=llm_predictor)
      
      if st.button("Ask"):
          # Do something with the entered text
          progress_bar.progress(35)
          response = index.query(
              prompt, 
              response_mode="tree_summarize",
              include_text=True, 
              embedding_mode='embedding',
              similarity_top_k=1
          )
          print(response)
          progress_bar.progress(80)
          st.write("Answer:", response)
          progress_bar.progress(100)


        # You can integrate here a langchain agent with GPT index to make it look more conversational

