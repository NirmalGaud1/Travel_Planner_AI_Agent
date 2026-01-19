import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
import datetime

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Travel Planner (Mistral-7B Local) 🌍")

@st.cache_resource(show_spinner="Loading Model... This may take a few minutes.")
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model

try:
    if not torch.cuda.is_available():
        st.error("CUDA not detected. This app requires an NVIDIA GPU.")
        st.stop()
    
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clean_markdown(text):
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    return text

with st.form("travel_form"):
    col1, col2 = st.columns(2)
    with col1:
        destination = st.text_input("📍 Destination", "Paris, France")
        days = st.number_input("🗓️ Days", 1, 30, 5)
    with col2:
        budget = st.text_input("💰 Budget", "€2000")
        interests = st.text_input("🎯 Interests", "History, Food")
    
    submit = st.form_submit_button("Generate Plan")

if submit:
    with st.spinner("Generating..."):
        prompt = f"""[INST] You are a travel expert. Create a {days}-day itinerary for {destination} with a budget of {budget}. 
        Interests: {interests}. 
        Format with Markdown headers, bullet points, and emojis. [/INST]"""
        
        result = generate(prompt)
        final_plan = clean_markdown(result)
        
        st.markdown(final_plan)
        
        st.download_button(
            label="Download Plan",
            data=final_plan,
            file_name=f"plan_{destination}.md",
            mime="text/markdown"
        )
