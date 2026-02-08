import streamlit as st
import os, json, tempfile
from agent import LocalStrandAgent
from tools import read_cv

st.set_page_config(page_title="Agentic Career Coach", layout="wide")

# --- Helper: Load Models ---
def load_models(provider):
    filename = "models_ollama.json" if provider == "Ollama" else "models_lmstudio.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception:
            return {"Default": "latest"}
    return {"Default": "latest"}

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.radio("Select Provider", ["Ollama", "LM Studio"])
    
    MODEL_MAP = load_models(provider)
    model_display = st.selectbox("Select Model", list(MODEL_MAP.keys()))
    selected_tag = MODEL_MAP[model_display]
    
    st.divider()
    st.subheader("📄 Candidate CV")
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    
    current_file_path = None
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        current_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(current_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Loaded: {uploaded_file.name}")

    if st.button("Clear Conversation"):
        if "agent" in st.session_state:
            st.session_state.agent.messages = [] # Reset agent history
        st.session_state.messages = []
        st.rerun()

# --- Initialize/Update Agent ---
if "agent" not in st.session_state:
    st.session_state.agent = LocalStrandAgent(
        provider=provider, 
        model=selected_tag, 
        tools=[read_cv], 
        system_prompt="Waiting for CV..."
    )

# Sync UI changes with Agent state
if st.session_state.agent.model != selected_tag or st.session_state.agent.provider != provider:
    st.session_state.agent.update_config(provider, selected_tag)

# --- Dynamic System Prompt ---
if current_file_path:
    system_prompt = f"""
    You are an expert Career Coach. The user has uploaded a CV at: "{current_file_path}"
    1. Use `read_cv` to answer questions about the candidate.
    2. If a Job Description is provided, perform a Gap Analysis vs the CV.
    """
    if st.session_state.agent.system_prompt != system_prompt:
        st.session_state.agent.system_prompt = system_prompt
        st.session_state.agent.messages = [{"role": "system", "content": system_prompt}]

# --- Chat Interface ---
st.title("🧬 Agentic Career Coach")
st.caption(f"Running via {provider} ({selected_tag})")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Chat Input (The Chat Box)
if prompt := st.chat_input("Paste Job Description or ask a question..."):
    if not current_file_path:
        st.error("Please upload a CV first.")
    else:
        # Display User Message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")