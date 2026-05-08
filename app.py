from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/kmeans")
def kmeans():
    return render_template("kmeans.html")

@app.route("/problematica")
def problematica():
    return render_template("problematica.html")

if __name__ == "__main__":
    app.run(debug=True)