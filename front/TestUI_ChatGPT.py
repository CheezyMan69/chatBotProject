import streamlit as st
import src.llm_calls.rag_functions
import os
from src.processing_indexing.chunking import detect_dtype

st.set_page_config(page_title="Chatbot", page_icon="💬")

# function to persist images/videos/audio to disk
def save_to_disk(data):
    with open(os.path.join("data", data.name), "wb") as f:
        f.write(data.getbuffer())
    return os.path.join("data", data.name)

# File uploading widget
# taking in images, videos, audio and text
uploaded_file = st.file_uploader("Choose a file for the RAG system", ["image/jpeg", ".jpeg", "image/png", ".png",  "image/jpg", ".jpg", "mp4", "mp3", "txt"])    
if uploaded_file is not None:
    if uploaded_file.name not in st.session_state:
        st.session_state[uploaded_file.name] = 'uploaded'
        if uploaded_file.type != "text/plain":
            st.write("Filename: ", uploaded_file.type, "is currently being processed.")
            detect_dtype(uploaded_file, uploaded_file.type, save_to_disk(uploaded_file))
        else: 
            st.write("Filename: ", uploaded_file.type, "is currently being processed.")
            detect_dtype(uploaded_file, uploaded_file.type)
    

st.title("💬 Simple Chatbot")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# rag activator
rag_button = st.toggle("RAG")

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate a bot response
    if rag_button:  
        bot_response = src.llm_calls.rag_functions.gemini_call_rag(user_input)
    else:
        bot_response = src.llm_calls.rag_functions.gemini_call_normal(user_input)


    # Add bot message to history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display bot message
    with st.chat_message("assistant"):
        st.write(bot_response)

