
# News Classification using NLP and Machine Learning

This project focuses on classifying news articles in English and Hindi using machine learning and natural language processing (NLP). Models are trained on datasets obtained from Kaggle and saved for reuse. The application also includes a Streamlit interface for user interaction.

# Features 

Language Support: English and Hindi news classification.
Machine Learning Models:
English: Multinomial Naive Bayes and Tokenization
Hindi: Random Forest Classifier and Tokenization
Streamlit Application: Interactive interface for classification.
Datasets: Preprocessed English and Hindi datasets sourced from Kaggle.


# Directory Structure


MLNClassifier/
├── EngNewsClassify.ipynb        # Jupyter notebook for training and saving English models
├── HinNewsClassify.ipynb        # Jupyter notebook for training and saving Hindi models
├── NewsNLP.py                   # Python script for loading models and running the classifier

├── Multinomial.pkl          # English Multinomial Naive Bayes model
├── tokenization.pkl         # English Tokenizer
├── hindi_Rf.pkl             # Hindi Random Forest model
└── hindiTokenization.pkl    # Hindi Tokenizer
├── english_news.csv         # English dataset (from Kaggle)
├── hindi_news.csv           # Hindi dataset (from Kaggle)
└── README.md                # Project documentation


# Setup Instructions

Prerequisites

Python 3.7 or higher
Libraries: numpy, pandas, sklearn, nltk, streamlit


Installation
Clone the repository:

git clone https://github.com/AdityaC784/MLNClassifier.git
cd MLNClassifier


Install dependencies:

pip install -r requirements.txt


# How to Run

1. Train and Save Models

Open EngNewsClassify.ipynb and HinNewsClassify.ipynb in Jupyter Notebook.
Run the cells to train the models and save them as .pkl files in the directory.


2. Streamlit Application
Run the Streamlit app:
    streamlit run NewsNLP.py


3. Classify News

Use the Streamlit app to input news texts.
The app will classify the news as relevant categories based on the language.

File Details

English Classification Models:
Multinomial.pkl: Trained Multinomial Naive Bayes classifier.
tokenization.pkl: Tokenizer for preprocessing English text.

Hindi Classification Models:
hindi_Rf.pkl: Trained Random Forest classifier.
hindiTokenization.pkl: Tokenizer for preprocessing Hindi text.

Datasets

English Dataset: [Kaggle Link](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)   
for english both test.csv and train.csv used.

Hindi Dataset: [Kaggle Link](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

Download and place them in the directory.

Future Work

Extend support to more languages.
Experiment with deep learning models.
Add more categories for classification.

