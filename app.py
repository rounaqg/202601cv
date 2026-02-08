import streamlit as st
import os
import json
import tempfile
from agent import LocalStrandAgent
from tools import read_cv

# --- Configuration ---
st.set_page_config(page_title="Agentic Career Coach", layout="wide")

# --- Helper: Load Models from JSON ---
def load_model_map(json_path="models.json"):
    default_map = {"Llama 3.1": "llama3.1"}
    if not os.path.exists(json_path):
        return default_map
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except:
        return default_map

MODEL_MAP = load_model_map()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selector
    model_display_name = st.selectbox("Select Model", list(MODEL_MAP.keys()))
    selected_model_tag = MODEL_MAP[model_display_name]
    
    st.divider()
    st.subheader("📄 Candidate CV")
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    
    # Save file to temp location
    current_file_path = None
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        current_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(current_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Loaded: {uploaded_file.name}")

    if st.button("Clear Conversation"):
        if "agent" in st.session_state:
            st.session_state.agent.clear_history()
        st.session_state.messages = []
        st.rerun()

# --- Initialize Agent ---
if "agent" not in st.session_state:
    st.session_state.agent = LocalStrandAgent(
        model=selected_model_tag,
        tools=[read_cv], # Only one tool needed now
        system_prompt="Waiting for CV..."
    )

# --- Dynamic State Updates ---
if st.session_state.agent.model != selected_model_tag:
    st.session_state.agent.update_model(selected_model_tag)

if current_file_path:
    # We update the prompt to handle "Pasted Text" logic
    system_prompt = f"""
    You are an expert Career Coach and HR Analyst.
    
    **YOUR CONTEXT:**
    The user has uploaded a CV at: "{current_file_path}"
    
    **YOUR TOOL:**
    - `read_cv(file_path)`: Use this to read the candidate's details. 
    
    **YOUR INSTRUCTIONS:**
    1. **General Questions:** If the user asks about the candidate (e.g., "skills?", "email?"), call `read_cv` and answer.
    2. **Job Description (JD) Analysis:** If the user pastes a Job Description (a large block of text describing a role):
       - First, call `read_cv` to get the candidate's info.
       - Then, compare the CV (Tool Output) vs the JD (User Message).
       - Provide a "Gap Analysis":
         - Match Score (0-100%)
         - Key Matching Skills
         - Missing Critical Skills
         - 3 Specific suggestions to tailor the CV for this JD.
    """
    
    if st.session_state.agent.system_prompt != system_prompt:
        st.session_state.agent.system_prompt = system_prompt
        # Reset history so the new prompt takes effect immediately
        st.session_state.agent.messages = [{"role": "system", "content": system_prompt}]

# --- Chat Interface ---
st.title("🧬 Agentic Career Coach")
st.caption(f"Powered by Localhost Ollama ({selected_model_tag})")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Handler
if prompt := st.chat_input("Paste Job Description text here OR ask a question..."):
    if not current_file_path:
        st.error("Please upload a CV first.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")