import re
import string
import nltk
import spacy

from nltk.corpus import stopwords

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download NLTK resources (only first time)
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Cleans the essay text by:
    - Lowercasing
    - Removing punctuation
    - Removing numbers
    - Removing stopwords
    - Lemmatizing
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatization
    doc = nlp(text)

    words = []

    for token in doc:

        if token.text not in stop_words and not token.is_space:

            words.append(token.lemma_)

    return " ".join(words)