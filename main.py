import streamlit as st
import base64
from typing import Optional
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="BOMBOLCAT🤖",
    page_icon="🤖",
    layout="centered",
)

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


def get_chatbot_response(
    text_input: str,
    image_file: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None,
) -> str:
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
        },
        {
            "role": "user",
            "content": "Give me the cpp code for this problem in the context of compiler design.Give something simple apt for a college student lab exam. Review the code to make sure it works ",
        },
    ]

    # Add chat history
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": text_input})

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
st.title("Bombolcat🤖")

# Chat message interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input container
image_file = None  # Initialize image_file
with st.container():
    user_input = st.chat_input("Type your message here...")
    toggle = st.toggle("Upload an image", False)
    if toggle:
        image_file = st.file_uploader(
            "Upload an image (optional)", type=["png", "jpg", "jpeg"]
        )
    else:
        image_file = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display the latest user input
        with st.chat_message("user"):
            st.write(user_input)

        # Get bot response
        with st.chat_message("assistant"):
            bot_response = get_chatbot_response(user_input, image_file)
            if bot_response:
                st.write(bot_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_response}
                )

# Clear chat button
if st.button("Clear Chat"):
    clear_chat()
    st.experimental_rerun()
