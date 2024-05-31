# 🎬 IMDB Sentiment Analysis Project

Welcome to the **IMDB Sentiment Analysis Project**! This project explores two different models for sentiment analysis on movie reviews: a basic sentiment analysis (SA) model and an aspect-based sentiment analysis (ABSA) model.

## 🚀 Project Overview

### Basic Sentiment Analysis (SA) Model
- Utilizes a simple neural network to classify movie reviews as positive or negative.
- Incorporates text vectorization and word embeddings.

### Aspect-Based Sentiment Analysis (ABSA) Model
- Extends sentiment analysis by considering specific aspects (e.g., plot, acting) of the reviews.
- Uses a more complex model that combines LSTM layers with dense layers for aspect analysis.

## 📂 Project Structure

```plaintext
IMDB_Sentiment_Analysis/
│
├── aclImdb/                   # Dataset folder (downloaded and extracted)
├── README.md                  # Project readme file
├── sentiment_analysis.py      # Main script for sentiment analysis
└── requirements.txt           # Dependencies


📥 Installation and Setup
**
Clone the repository:
**
git clone https://github.com/yourusername/IMDB_Sentiment_Analysis.git
cd IMDB_Sentiment_Analysis

**
Create a virtual environment:
**
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

**
Install dependencies:
**
pip install -r requirements.txt

**
Download and extract the IMDB dataset:
**
python sentiment_analysis.py --setup


⚙️ Usage
To run the sentiment analysis models, execute the main script:
python sentiment_analysis.py


This script will:
Download and preprocess the IMDB dataset.
Train and evaluate the Basic Sentiment Analysis model.
Train and evaluate the Aspect-Based Sentiment Analysis model.
Display results including accuracy and sample predictions.

📊 Results
Basic Sentiment Analysis (SA)
Test Accuracy: 87%
Aspect-Based Sentiment Analysis (ABSA)
Test Accuracy: 85%

📈 Visualizations
Positive Words Word Cloud
Negative Words Word Cloud
Model Accuracy
Model Loss

🎓 Sample Predictions
Basic SA Sample Review:
The movie was fantastic! The story was gripping and the characters were well-developed.
Sentiment: Positive

ABSA Sample Review:
The movie had great acting and a captivating storyline, but the music was terrible.
Sentiment: Positive (based on majority aspects being positive)

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

<p align="center">
  <img src="https://img.shields.io/badge/Made_with-❤️-red.svg" alt="Made with Love">
  <img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="Python">
</p>
