import streamlit as st
from dotenv import load_dotenv

import os

# Custom imports 
from multipage import MultiPage
from pages import chatPDF, data_upload # import your pages here

st.set_page_config(
    page_title="ChatPDF",
    page_icon= ' put link to ai dev lab logo',
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "HEOL made for you this app, I hope you enjoy it! https://www.linkedin.com/in/lmthsm/"
    }
)
# Create an instance of the app 
app = MultiPage()
 
 
# Title and the img of the main page
st.image('ai dev lab logo link',width=200)
st.header("ChatPDF is available for you!")

# Add all your applications (pages) here
app.add_page("Upload Data", data_upload.app)
app.add_page("chatPDF", chatPDF.app)



# The main app
app.run()