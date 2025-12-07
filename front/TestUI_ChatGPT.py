import streamlit as st
import src.rag_functions
import os

st.set_page_config(page_title="Chatbot", page_icon="💬")

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
        bot_response = src.rag_functions.gemini_call_rag(user_input)
    else:
        bot_response = src.rag_functions.gemini_call_normal(user_input)


    # Add bot message to history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display bot message
    with st.chat_message("assistant"):
        st.write(bot_response)

