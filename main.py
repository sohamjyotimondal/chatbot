import streamlit as st
import base64
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(page_title="Waifu", page_icon="🤖")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = "2024-02-15-preview"

if not all([api_key, api_base, deployment_name]):
    st.error("Please set all required environment variables in your .env file")
    st.stop()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}",
)


def encode_image(image_file) -> str:
    """Encode image to base64 string"""
    return base64.b64encode(image_file.read()).decode("ascii")


def get_chatbot_response(text_input: str, image_file=None) -> str:
    """Get response from Azure OpenAI API using SDK"""
    # Prepare message content
    message_content = []

    # Add image if provided
    if image_file:
        image_data = encode_image(image_file)
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            }
        )

    # Add text input
    message_content.append({"type": "text", "text": text_input})

    # Prepare messages including history
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        }
    ]

    # Add chat history
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": message_content})

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=3000,
            temperature=0.9,
            top_p=0.95,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with Client: {str(e)}")
        return None


def clear_chat():
    st.session_state.messages = []


# Streamlit UI
st.title("DL God 🤖")


# File uploader for images
image_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

# Display uploaded image
if image_file:
    st.image(image_file, caption="Uploaded Image", use_container_width=True)

# Text input
text_input = st.text_input("Enter your message:", key="text_input")

# Create two columns for the buttons
col1, col2 = st.columns([1, 1])

# Send button
with col1:
    if st.button("Send", key="send"):
        if text_input or image_file:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": text_input})

            # Get bot response
            bot_response = get_chatbot_response(text_input, image_file)

            if bot_response:
                # Add bot response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_response}
                )
        else:
            st.warning("Please enter a message or upload an image.")

# Clear chat button
with col2:
    if st.button("Clear Chat", key="clear"):
        clear_chat()
        st.rerun()

# Display chat history
st.subheader("Chat History")
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.write("You: " + message["content"])
        else:
            st.write("Bot: " + message["content"])
        st.divider()
