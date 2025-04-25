import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "Fake_news_bert.model"  # Your saved model file
MODEL_NAME = "bert-base-uncased"  # Same as the pretrained model used

def load_model():
    try:
        # First create the model architecture
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        
        # Load the state dictionary
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Load the weights into the model
        model.load_state_dict(state_dict)
        
        # Set the model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource  # Cache the model loading
def initialize_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = load_model()
    return tokenizer, model

def predict_fake_news(text, model, tokenizer):
    if model is None:
        return "Error", 0.0
        
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()
        return "Fake News" if prediction == 0 else "Real News", confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

# Initialize the app
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Load model and tokenizer
tokenizer, model = initialize_model()

# UI Elements
st.title("📰 Fake News Detector")
st.markdown("Detect whether a news article is real or fake using AI-powered BERT model.")

# User Input
news_text = st.text_area("Enter news content:", height=150)

if st.button("Detect News"):
    if news_text.strip():
        with st.spinner("Analyzing..."):
            label, confidence = predict_fake_news(news_text, model, tokenizer)
            
            if label != "Error":
                confidence_percentage = confidence * 100
                st.success(f"Prediction: **{label}**")
                st.progress(confidence)
                st.info(f"Confidence: **{confidence_percentage:.2f}%**")
    else:
        st.warning("Please enter some news text to analyze.")

# Footer
st.markdown("---")
st.markdown("💡 *Built using Streamlit & BERT for NLP classification.*")