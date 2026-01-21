import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Multi-Task Tool",
    layout="wide"
)
st.title("🤖 Multifunctional AI Tool (Hugging Face)")

# ---------------- MODEL CACHE ----------------
@st.cache_resource
def load_nlp_pipeline(task, model):
    """Load NLP model from Hugging Face and cache it"""
    return pipeline(task=task, model=model)

@st.cache_resource
def load_sd_pipeline():
    """Load Stable Diffusion (public model, no token required)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using a public model that doesn't require a token
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # public model
        torch_dtype=torch.float32
    )
    pipe.to(device)
    return pipe

# ---------------- TASK SELECTION ----------------
task = st.sidebar.selectbox(
    "Select AI Task",
    (
        "Sentiment Analysis",
        "Text Summarization",
        "Next Word Prediction",
        "Story Generation",
        "Chatbot",
        "Question Answering",
        "Image Generation"
    )
)

# ---------------- SENTIMENT ANALYSIS ----------------
if task == "Sentiment Analysis":
    st.subheader("😊 Sentiment Analysis")
    text = st.text_area("Enter text")
    if st.button("Analyze"):
        pipe = load_nlp_pipeline(
            "sentiment-analysis",
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
        result = pipe(text)
        st.success(result)

# ---------------- TEXT SUMMARIZATION ----------------
elif task == "Text Summarization":
    st.subheader("📄 Text Summarization")
    text = st.text_area("Enter long text")
    if st.button("Summarize"):
        pipe = load_nlp_pipeline(
            "summarization",
            "facebook/bart-large-cnn"
        )
        summary = pipe(text, max_length=130, min_length=30)
        st.success(summary[0]["summary_text"])

# ---------------- NEXT WORD PREDICTION ----------------
elif task == "Next Word Prediction":
    st.subheader("➡️ Next Word Prediction")
    text = st.text_input("Enter text")
    if st.button("Predict"):
        pipe = load_nlp_pipeline(
            "text-generation",
            "gpt2"
        )
        output = pipe(text, max_new_tokens=20)
        st.success(output[0]["generated_text"])

# ---------------- STORY GENERATION ----------------
elif task == "Story Generation":
    st.subheader("📖 Story Generation")
    prompt = st.text_area("Story prompt")
    if st.button("Generate Story"):
        pipe = load_nlp_pipeline(
            "text-generation",
            "EleutherAI/gpt-neo-125M"
        )
        story = pipe(prompt, max_new_tokens=200)
        st.success(story[0]["generated_text"])

# ---------------- CHATBOT ----------------
elif task == "Chatbot":
    st.subheader("💬 Chatbot")
    user_input = st.text_input("You")
    if st.button("Send"):
        pipe = load_nlp_pipeline(
            "text-generation",
            "microsoft/DialoGPT-medium"
        )
        response = pipe(user_input, max_new_tokens=100)
        st.success(response[0]["generated_text"])

# ---------------- QUESTION ANSWERING ----------------
elif task == "Question Answering":
    st.subheader("❓ Question Answering")
    context = st.text_area("Context")
    question = st.text_input("Question")
    if st.button("Get Answer"):
        pipe = load_nlp_pipeline(
            "question-answering",
            "deepset/roberta-base-squad2"
        )
        answer = pipe(
            question=question,
            context=context
        )
        st.success(answer["answer"])

# ---------------- IMAGE GENERATION ----------------
elif task == "Image Generation":
    st.subheader("🎨 Image Generation (Stable Diffusion)")
    prompt = st.text_input("Enter image prompt")
    num_images = st.slider("Number of images", min_value=1, max_value=3, value=1)
    steps = st.slider("Inference steps (quality)", min_value=10, max_value=50, value=25)
    
    if st.button("Generate Image"):
        pipe = load_sd_pipeline()
        images = pipe([prompt]*num_images, num_inference_steps=steps).images
        
        for i, img in enumerate(images):
            st.image(img, caption=f"Generated Image {i+1}", use_column_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using **Streamlit + Hugging Face Transformers / Diffusers**")
