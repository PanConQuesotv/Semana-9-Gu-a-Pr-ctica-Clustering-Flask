from flask import Flask, render_template

app = Flask(__name__)

# Página principal (explicación del proyecto)
@app.route("/")
def home():
    return render_template("index.html")

# Página para ver el dataset
@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

# Página para la implementación de K-Means
@app.route("/kmeans")
def kmeans():
    return render_template("kmeans.html")

if __name__ == "__main__":
    app.run(debug=True)