from flask import Flask, render_template, request
from predict import predict_score

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    essay = request.form["essay"]

    result = predict_score(essay)

    return render_template(
        "index.html",
        essay=essay,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)