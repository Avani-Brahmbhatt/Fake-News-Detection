
# 📰 Fake News Detection using BERT

This project leverages **BERT (Bidirectional Encoder Representations from Transformers)** to detect whether a news article is **real** or **fake**. A pre-trained BERT model is fine-tuned on a labeled dataset and served using a **Streamlit** web app.

## 🚀 Features

- Fine-tuned BERT model for binary classification (Fake vs Real).
- Streamlit-based interactive web interface.
- Predicts label along with confidence score.
- User-friendly design with real-time feedback.

## 🧠 Model Overview

- **Model:** `bert-base-uncased` from HuggingFace Transformers.
- **Task:** Binary text classification.
- **Frameworks Used:** PyTorch, Transformers, Streamlit.


## 🛠️ Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Avani-Brahmbhatt/Fake-News-Detection.git
   cd Fake-News-Detection
   ```

2. **Install Requirements**

   It's recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```


3. **Run the App**

   ```bash
   streamlit run app.py
   ```

## 📊 How It Works

- Input a news article or passage into the text box.
- The app uses a fine-tuned BERT model to classify the input as either **Fake News** or **Real News**.
- A confidence score is displayed as a progress bar.

## 📌 Example Usage

1. Paste the article in the text area.
2. Click **"Detect News"**.
3. View the classification and confidence score.

## 📦 Dependencies

- `torch`
- `transformers`
- `streamlit`



## 📚 Dataset

*Dataset used for training is included and processed within the `fake-news-detection.ipynb` notebook. Common datasets include [Fake and real news dataset on Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection).*


---

**Built with ❤️ using BERT and Streamlit**
```

---
