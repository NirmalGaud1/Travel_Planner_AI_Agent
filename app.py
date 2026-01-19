import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
import datetime

st.set_page_config(
    page_title="AI Travel Planner - Mistral 7B",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Travel Planner (Powered by Mistral-7B) 🌍")

with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This application uses Mistral-7B-Instruct-v0.2 to create personalized travel itineraries.
    Features:
    • Realistic 2026 prices & recommendations
    • GPU acceleration (when available)
    • Generation time: 30–90 seconds
    • Model loads only once per session
    Note: First run may take 2–5 minutes to download & load the model.
    """)
    st.markdown("---")
    st.info("Built with ❤️ | January 2026")

@st.cache_resource(show_spinner="Loading Mistral-7B model... Please wait ⏳")
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {str(e)}\nTry restarting the app or check your GPU memory.")
    st.stop()

def generate(prompt, max_new_tokens=1400, temperature=0.75, top_p=0.92):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
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
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[-1].strip()
    for marker in ["```markdown", "```md", "```", "```text"]:
        if marker in text:
            parts = text.split(marker)
            for part in parts[1:]:
                cleaned = part.split("```")[0].strip()
                if len(cleaned) > 150:
                    return cleaned
    if "#" in text:
        start_idx = text.find("#")
        return text[start_idx:].strip()
    return text if len(text) > 200 else "Could not extract proper plan format."

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
    with st.spinner("Creating your personalized travel itinerary... (30–90 seconds)"):
        prompt = f"""[INST] You are an expert travel planner specialized in realistic 2026 travel planning with current prices, weather patterns, and trends.

Create a detailed, beautiful, and practical travel itinerary for:

Destination: {destination}
Duration: {days} days
Total budget: {budget}
Travel period: {dates}
Number of travelers: {travelers}
Main interests: {interests}
Special requests: {special_requests}

Follow **exactly** this markdown structure. Use emojis. Be exciting but realistic.

# Trip to {destination}

## Quick Overview
- Duration:
- Estimated cost range (mid-range):
- Weather & best time notes:
- Overall vibe:

## Day-by-Day Itinerary

### Day 1: Arrival & First Exploration
- Morning:
- Afternoon:
- Evening:
- Suggested accommodation:
- Approx. daily cost:

### Day 2: ...
(continue for all {days} days)

## Must-Try Food & Local Experiences

## Realistic Budget Breakdown
- Flights/Transport:
- Accommodation:
- Food:
- Activities/Sightseeing:
- Local transport/Misc:
- Total estimate:

## Practical Tips for 2026
- Best transport options
- Safety notes
- Useful apps
- Sustainable travel ideas

Output **ONLY** the clean markdown itinerary. No extra explanations or text outside the markdown. [/INST]"""

        raw_output = generate(prompt)
        cleaned_plan = clean_markdown(raw_output)

    st.success("Your travel plan is ready!")
    st.markdown("### ✨ Your Personalized Travel Itinerary ✨")
    st.markdown(cleaned_plan)

    safe_dest = re.sub(r'[^a-zA-Z0-9_-]', '_', destination)
    filename = f"Travel_Plan_{safe_dest}_{datetime.date.today()}.md"

    st.download_button(
        label="📥 Download Plan as Markdown",
        data=cleaned_plan,
        file_name=filename,
        mime="text/markdown",
        key="download_btn"
    )
