# Multifunctional-NLP-and-Image-Generation-Tool-using-Hugging-Face-Models
# Multifunctional NLP and Image Generation Tool using Hugging Face Models

A **multifunctional AI tool** that allows users to select and utilize different pretrained models from [Hugging Face](https://huggingface.co/) for various NLP and image generation tasks. The tool provides a **user-friendly interface** to select a task and input the required text or image for processing.

---
## Features
This tool supports the following tasks:

- **Text Summarization** – Condense long text into concise summaries.
- **Next Word Prediction** – Predict the next word in a sequence.
- **Story Prediction** – Generate creative story continuations.
- **Chatbot** – Interact with a conversational AI model.
- **Sentiment Analysis** – Detect sentiment (positive, negative, neutral) in text.
- **Question Answering** – Provide answers based on a given context.
- **Image Generation** – Create images from textual prompts using AI models.

All models are pretrained and sourced from Hugging Face.

---
## Developer Guide

### 1. Tools Required

- **Python**  
- **VS Code** (or any code editor)  
- **Command Prompt / Terminal**  

---
### 2. Python Libraries

#### a. Dashboard Creation
- `streamlit`  

#### b. Deep Learning
- `torch`  
- `transformers`  
- `diffusers`  

---
### 3. Python Modules to Import

#### a. Hugging Face Models
```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from diffusers import StableDiffusionPipeline

b. Dashboard Libraries
import streamlit as st


4. Implementation Process

Set up environment and install necessary libraries:

pip install streamlit torch transformers diffusers


Implement front-end interface for task selection and input using Streamlit.

Load and integrate pretrained models from Hugging Face for all supported tasks:

Text Summarization

Next Word Prediction

Story Prediction

Chatbot

Sentiment Analysis

Question Answering

Image Generation

Implement backend logic to process user input and generate outputs.

Test the application with various inputs and refine the interface and backend.

5. Running the Application Locally

Open VS Code and create app.py with all your Streamlit code.

Open Command Prompt / Terminal and navigate to your project directory:

cd path\to\your\project


Run the Streamlit app:

streamlit run app.py


The app will launch in your browser at http://localhost:8501 (or another available port).
