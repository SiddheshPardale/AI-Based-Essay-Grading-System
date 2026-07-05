import joblib


from preprocess import clean_text

# Load model
model = joblib.load("models/essay_model.pkl")

# Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")




def predict_score(essay):

    cleaned = clean_text(essay)

    tfidf = vectorizer.transform([cleaned])

    score = float(model.predict(tfidf)[0])

    words = essay.split()

    word_count = len(words)

    sentence_count = essay.count(".") + essay.count("?") + essay.count("!")

    if word_count == 0:
        avg_word_length = 0
    else:
        avg_word_length = sum(len(word) for word in words) / word_count

    vocabulary = len(set(words))

    if word_count == 0:
        vocab_richness = 0
    else:
        vocab_richness = vocabulary / word_count

    grammar_errors = 0

    feedback = []

    if grammar_errors < 5:
        feedback.append("Good grammar.")
    else:
        feedback.append("Grammar needs improvement.")

    if word_count > 150:
        feedback.append("Essay length is good.")
    else:
        feedback.append("Essay is too short.")

    if vocab_richness > 0.60:
        feedback.append("Excellent vocabulary.")
    else:
        feedback.append("Use more diverse vocabulary.")

    return {
        "score": round(score,2),
        "grammar": grammar_errors,
        "words": word_count,
        "sentences": sentence_count,
        "avg_length": round(avg_word_length,2),
        "vocabulary": round(vocab_richness,2),
        "feedback": feedback
    }


if __name__ == "__main__":

    essay = input("Enter Essay:\n")

    result = predict_score(essay)

    print(result)