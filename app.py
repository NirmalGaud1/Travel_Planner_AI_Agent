import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
import datetime

# ─── Sidebar & Title ───────────────────────────────────────────────────
st.title("🌍 AI Travel Planner (Powered by Mistral-7B) 🌍")
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses Mistral-7B-Instruct-v0.3 to generate personalized travel itineraries.
- Realistic 2026 prices & tips
- Runs on GPU for speed (if available)
- Generation: 30-90 seconds

**Note:** Model loads once per session. Be patient!
""")

# ─── Model Loading (cached for speed) ──────────────────────────────────
@st.cache_resource
def load_model():
    with st.spinner("Loading Mistral-7B model... (first time may take 2-5 mins) ⏳"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        return tokenizer, model

tokenizer, model = load_model()

# ─── Generation Function ───────────────────────────────────────────────
def generate(prompt, max_new_tokens=1400, temperature=0.75, top_p=0.92):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ─── Clean Markdown ────────────────────────────────────────────────────
def clean_markdown(text):
    text = text.strip()
    if "[INST]" in text:
        text = text.split("[/INST]")[1].strip() if "[/INST]" in text else text
    markers = ["```markdown", "```md", "```", "```text"]
    for marker in markers:
        if marker in text:
            parts = text.split(marker)
            for part in parts[1:]:
                if len(part.strip()) > 100:
                    return part.split("```")[0].strip()
    if "#" in text:
        start = text.find("#")
        return text[start:].strip()
    return text if len(text) > 200 else "No plan generated. Try again."

# ─── Main App Form ─────────────────────────────────────────────────────
with st.form(key="travel_form"):
    st.subheader("Plan Your Trip!")
    
    dest = st.text_input("📍 Destination (city/country)", value="Goa, India")
    days = st.number_input("🗓️ Number of days", min_value=1, max_value=30, value=5)
    budget = st.text_input("💰 Total budget (approx in INR or currency)", value="₹80,000")
    dates = st.text_input("📅 Travel month/year (or flexible)", value="March 2026")
    interests = st.text_input("🎯 Interests (beaches, culture, food...)", value="beaches, food, relaxation")
    people = st.text_input("👥 Travelers (e.g. 2 adults)", value="2 adults")
    special = st.text_input("✨ Special requests (vegetarian, budget stay...)", value="None")
    
    submit = st.form_submit_button("Generate My Travel Plan! ✨")

if submit:
    with st.spinner("Creating your personalized itinerary... (30-90 seconds) ⏳"):
        prompt = f"""[INST] You are an expert travel planner, specializing in realistic 2026 trips with current prices, weather, and trends.

Create a detailed, exciting travel itinerary for:

Destination: {dest}
Duration: {days} days
Budget: {budget} total
Dates: {dates}
Travelers: {people}
Interests: {interests}
Special requests: {special}

Use this exact markdown structure with emojis:

# Trip to {dest}

## Quick Overview
- Duration: 
- Estimated cost range (mid-range): 
- Weather & best time notes: 
- Overall vibe: 

## Day-by-Day Itinerary

### Day 1: Arrival & Exploration
- Morning: 
- Afternoon: 
- Evening: 
- Suggested accommodation: 
- Approx. daily cost: 

### Day 2: ...
(continue for all {days} days)

## Must-Try Food & Experiences
## Budget Breakdown (in INR or specified currency)
- Flights/Transport: 
- Accommodation: 
- Food: 
- Activities: 
- Local travel/Misc: 
- Total estimate: 

## Practical Tips for 2026
- Transport options
- Safety & apps
- Sustainable ideas

Output ONLY the clean markdown plan. No extra text. [/INST]"""
        
        raw_response = generate(prompt)
        plan = clean_markdown(raw_response)
    
    st.markdown("### ✨ Your Personalized Travel Itinerary ✨")
    st.markdown(plan)
    
    # Download button after displaying
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', dest)
    filename = f"Travel_Plan_{safe_name}_{datetime.date.today()}.md"
    st.download_button(
        label="📥 Download Plan as Markdown",
        data=plan,
        file_name=filename,
        mime="text/markdown"
    )

# ─── Footer ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ by Nirmal Gaud | January 2026")
