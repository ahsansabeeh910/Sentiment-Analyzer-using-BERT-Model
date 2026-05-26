# BERT Sentiment Analyzer


## Overview

BERT Sentiment Analyzer is a Machine learning based web application that performs sentiment analysis on user-provided text using a fine-tuned BERT transformer model.

The application predicts whether the sentiment of the input text is:

* Positive
* Negative
* Neutral

The project is built using Streamlit for the frontend and HuggingFace Transformers with PyTorch for the backend learning model.

---

## Features

* Fine-tuned BERT sentiment classification
* Real-time text sentiment prediction
* Confidence score visualization
* Interactive probability charts using Plotly
* Prediction history tracking
* Modern responsive UI
* Sidebar navigation with example inputs
* Streamlit-based web application

---

## Technologies Used

* Python
* Streamlit
* PyTorch
* HuggingFace Transformers
* Plotly
* Pandas
* Scikit-learn

---

## Project Structure

```bash
BERT_Sentiment_Analyzer/
│
├── app.py
├── model.ipynb
├── requirements.txt
├── README.md
├── train.csv
│
├── bert_sentiment_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/BERT_Sentiment_Analyzer.git
```

### Move into Project Folder

```bash
cd BERT_Sentiment_Analyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Model Training

The BERT model was trained using Google Colab with GPU acceleration.

Training notebook:

* `model.ipynb`

---

## Screenshots

Sentiment Analysis
<img width="1859" height="982" alt="Screenshot 2026-05-26 182131" src="https://github.com/user-attachments/assets/ed096d1f-66f3-49ae-9e14-d1f827a85331" />

Prediction Distribution
<img width="1436" height="914" alt="Screenshot 2026-05-26 182142" src="https://github.com/user-attachments/assets/f3099c2e-56f8-4ee2-9998-9e701c5dfc55" />


---

## Author

### Sabeeh Ahsan

Jaypee Institute of Information Technology

---

## License

This project is developed for educational and portfolio purposes.
