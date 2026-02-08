import streamlit as st
import os
import tempfile
from agent import LocalStrandAgent
from tools import read_cv

# --- Configuration ---
st.set_page_config(page_title="Single-File Strand Agent", layout="wide")

# --- Model Definitions (Display Name -> Ollama Tag) ---
# UPDATE THIS LIST based on what you have pulled in Ollama
MODEL_MAP = {
    "llama3.2:1b": "llama3.2:1b",
    "gpt-oss:20b": "gpt-oss:20b",
    "gemma3:4b": "gemma3:4b",
    "deepseek-r1:14b": "deepseek-r1:14b",
    "mistral:latest": "mistral:latest",
    "qwen3:14b": "qwen3:14b"
}

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    
    # 1. Model Selector
    model_display_name = st.selectbox("Select LLM Brain", list(MODEL_MAP.keys()))
    selected_model_tag = MODEL_MAP[model_display_name]
    
    # 2. File Upload
    st.divider()
    st.subheader("📄 Target CV")
    uploaded_file = st.file_uploader("Upload a Resume", type="pdf")
    
    # Handle File Save (Save to temp path)
    current_file_path = None
    if uploaded_file:
        # Save to a temporary file that persists during the session
        temp_dir = tempfile.mkdtemp()
        current_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(current_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Loaded: {uploaded_file.name}")

    # 3. Clear History
    st.divider()
    if st.button("Reset Conversation"):
        if "agent" in st.session_state:
            st.session_state.agent.clear_history()
        st.session_state.messages = []
        st.rerun()

# --- Initialize Agent ---
if "agent" not in st.session_state:
    # We initialize with a placeholder prompt; it updates when a file is uploaded
    st.session_state.agent = LocalStrandAgent(
        model=selected_model_tag,
        tools=[read_cv],
        system_prompt="You are a helpful assistant. Wait for a file to be uploaded."
    )

# --- Dynamic State Updates ---
# 1. Update Model if changed
if st.session_state.agent.model != selected_model_tag:
    st.session_state.agent.update_model(selected_model_tag)
    st.toast(f"Brain switched to: {selected_model_tag}")

# 2. Update System Prompt if File Exists
if current_file_path:
    # We inject the file path directly into the system prompt.
    # This is "Grounding" the agent to this specific file.
    new_system_prompt = f"""
    You are an expert HR Analyst.
    
    CONTEXT:
    The user has uploaded a CV file at this specific path:
    "{current_file_path}"
    
    INSTRUCTIONS:
    1. If the user asks about the candidate, use the `read_cv` tool.
    2. You MUST pass the exact path "{current_file_path}" to the tool.
    3. Do not ask the user for the file path; you already know it.
    4. Answer based strictly on the file content.
    """
    
    # Only update if prompt is different to avoid resetting history constantly
    if st.session_state.agent.system_prompt != new_system_prompt:
        st.session_state.agent.system_prompt = new_system_prompt
        # We also need to update the system message in the message history
        st.session_state.agent.messages[0]["content"] = new_system_prompt

# --- Chat Interface ---
st.title("🧬 Strand Agent")
st.caption(f"Running on Localhost • Model: {selected_model_tag}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Handler
if prompt := st.chat_input("Ask about the candidate..."):
    if not current_file_path:
        st.error("Please upload a PDF first!")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner(f"Reading file & Thinking ({selected_model_tag})..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Connection Error: {e}")