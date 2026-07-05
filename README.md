# AI Based Essay Grading System

## About the Project

This project is an AI Based Essay Grading System developed using Python and Natural Language Processing (NLP). It automatically evaluates essays and predicts a score based on the content of the essay.

The project uses text preprocessing, TF-IDF feature extraction, and a Random Forest Machine Learning model for essay score prediction.

## Features

- Essay Score Prediction
- Text Preprocessing
- TF-IDF Feature Extraction
- Grammar and Vocabulary Analysis
- Essay Statistics
- Web Interface using Flask

## Technologies Used

- Python
- Flask
- Scikit-learn
- NLTK
- spaCy
- Pandas
- NumPy
- Random Forest

## Project Structure

```
AI-Based-Essay-Grading-System/
│
├── app.py
├── train_model.py
├── predict.py
├── preprocess.py
├── feature_extraction.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── models/
│   ├── essay_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── dataset/
    └── ASAP2_train_sourcetexts.csv
```

## How to Run

1. Clone the repository.

```
git clone <https://github.com/SiddheshPardale/AI-Based-Essay-Grading-System>
```

2. Install the required libraries.

```
-r requirements.txt
```

3. Download the spaCy model.

```
python -m spacy download en_core_web_sm
```

4. Run the project.

```
python app.py
```

5. Open your browser and visit:

```
http://127.0.0.1:5000
```

## Sample Output

The application predicts:

- Essay Score
- Word Count
- Sentence Count
- Vocabulary Richness
- Grammar Feedback
- Suggestions for Improvement


## Dataset

The project uses the ASAP 2.0 Essay Dataset for training and testing.

## Author

**Siddhesh Pardale**

