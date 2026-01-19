import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import datetime

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Travel Planner (CPU-Friendly Version) 🌍")

with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This app uses **Qwen2.5-1.5B-Instruct** (very lightweight LLM)  
    running fully on CPU for maximum compatibility.

    • Loads in 1–4 minutes on Streamlit Cloud  
    • Uses ~1.5–2.5 GB RAM  
    • Generation: 20–90 seconds  
    • No GPU required!

    Perfect for free cloud deployment.
    """)
    st.markdown("---")
    st.info("Built with ❤️ | January 2026")

@st.cache_resource(show_spinner="Loading lightweight Qwen2.5-1.5B model... (CPU only) ⏳")
def load_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",                # Force CPU
        torch_dtype=torch.float32,       # CPU safe dtype
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {str(e)}\n\nTry rebooting the app from dashboard.")
    st.stop()

def generate(prompt, max_new_tokens=1000, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clean_markdown(text):
    text = text.strip()
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[-1].strip() if "<|im_start|>" in text else text
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[-1].strip()
    for marker in ["```markdown", "```md", "```", "```text"]:
        if marker in text:
            parts = text.split(marker)
            for part in parts[1:]:
                cleaned = part.split("```")[0].strip()
                if len(cleaned) > 100:
                    return cleaned
    if "#" in text:
        return text[text.find("#"):].strip()
    return text if len(text) > 150 else "Could not extract proper plan format."

with st.form(key="travel_form"):
    st.subheader("Tell us about your trip ✈️")

    col1, col2 = st.columns(2)

    with col1:
        destination = st.text_input("📍 Destination (city/country)", value="Goa, India")
        days = st.number_input("🗓️ Number of days", min_value=1, max_value=30, value=5)
        budget = st.text_input("💰 Approximate total budget", value="₹80,000")

    with col2:
        dates = st.text_input("📅 When are you traveling? (month/year or flexible)", value="March 2026")
        interests = st.text_input("🎯 Main interests", value="beaches, food, relaxation")
        travelers = st.text_input("👥 Number of travelers", value="2 adults")

    special_requests = st.text_input("✨ Any special requests?", value="None (vegetarian food, budget stay, etc.)")

    submit_button = st.form_submit_button("✨ Generate My Travel Plan ✨", use_container_width=True)

if submit_button:
    with st.spinner("Creating your itinerary... (20–90 seconds on CPU)"):
        prompt = f"""<|im_start|>system
You are an expert travel planner for realistic 2026 trips.
Create a detailed, beautiful itinerary using markdown and emojis.<|im_end|>
<|im_start|>user
Destination: {destination}
Duration: {days} days
Budget: {budget}
Dates: {dates}
Travelers: {travelers}
Interests: {interests}
Special requests: {special_requests}

Structure exactly like this:

# Trip to {destination}

## Quick Overview
- Duration:
- Estimated cost:
- Weather notes:
- Vibe:

## Day-by-Day Itinerary

### Day 1: ...
- Morning:
- Afternoon:
- Evening:
- Stay:
- Cost:

(continue all days)

## Food & Experiences

## Budget Breakdown

## 2026 Tips

Only output markdown.<|im_end|>
<|im_start|>assistant"""

        raw_output = generate(prompt)
        cleaned_plan = clean_markdown(raw_output)

    st.success("Plan ready!")
    st.markdown("### ✨ Your Travel Itinerary ✨")
    st.markdown(cleaned_plan)

    safe_dest = re.sub(r'[^a-zA-Z0-9_-]', '_', destination)
    filename = f"Travel_Plan_{safe_dest}_{datetime.date.today()}.md"

    st.download_button(
        label="📥 Download as Markdown",
        data=cleaned_plan,
        file_name=filename,
        mime="text/markdown"
    )
