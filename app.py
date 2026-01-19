import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import datetime

st.set_page_config(page_title="AI Travel Planner", page_icon="🌍", layout="wide")
st.title("🌍 AI Travel Planner (CPU Mode) 🌍")

@st.cache_resource(show_spinner="Loading Model to RAM... This will take a long time.")
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Loading in float16 to save some RAM, but still using CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"System RAM exceeded: {str(e)}")
    st.stop()

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with st.form("travel_form"):
    destination = st.text_input("📍 Destination", "Goa, India")
    submit = st.form_submit_button("Generate Plan")

if submit:
    with st.spinner("Generating on CPU (Slow)..."):
        prompt = f"[INST] Create a short travel plan for {destination}. [/INST]"
        result = generate(prompt)
        st.markdown(result.split("[/INST]")[-1])
