
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
├── EngNewsClassify.ipynb        # Jupyter notebook for training and saving English models <br>
├── HinNewsClassify.ipynb        # Jupyter notebook for training and saving Hindi models<br>
├── NewsNLP.py                   # Python script for loading models and running the classifier<br>

├── Multinomial.pkl          # English Multinomial Naive Bayes model<br>
├── tokenization.pkl         # English Tokenizer<br>
├── hindi_Rf.pkl             # Hindi Random Forest model<br>
└── hindiTokenization.pkl    # Hindi Tokenizer<br>
├── english_news.csv         # English dataset (from Kaggle)<br>
├── hindi_news.csv           # Hindi dataset (from Kaggle)<br>
└── README.md                # Project documentation<br>


# Setup Instructions

<h3>Prerequisites</h3>

Python 3.7 or higher
Libraries: numpy, pandas, sklearn, nltk, streamlit
 
 
<h3>Installation</h3>

Clone the repository:

       git clone https://github.com/AdityaC784/MLNClassifier.git
       cd MLNClassifier


Install dependencies:

       pip install -r requirements.txt


# How to Run

1. <h2>Train and Save Models</h2>

Open EngNewsClassify.ipynb and HinNewsClassify.ipynb in Jupyter Notebook. <br>
Run the cells to train the models and save them as .pkl files in the directory.


2. <h2> Streamlit Application</h2>

Run the Streamlit app:   streamlit run NewsNLP.py  <br>

   
3. <h2>Classify News</h2>

Use the Streamlit app to input news texts.
The app will classify the news as relevant categories based on the language.


# File Details

<h4>English Classification Models:</h4>

Multinomial.pkl: Trained Multinomial Naive Bayes classifier.
tokenization.pkl: Tokenizer for preprocessing English text.

<h4>Hindi Classification Models:</h4>
hindi_Rf.pkl: Trained Random Forest classifier.
hindiTokenization.pkl: Tokenizer for preprocessing Hindi text.

# Datasets

<h4>English Dataset:</h4> [Kaggle Link](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)   
for english both test.csv and train.csv used.

<h4>Hindi Dataset:</h4>   [Kaggle Link](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

Download and place them in the directory.


# Future Work

Extend support to more languages.
Experiment with deep learning models.
Add more categories for classification.

