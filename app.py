from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("sp.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    salary = None

    if request.method == "POST":
        exp = float(request.form["experiance"])

        # Correct prediction
        salary = model.predict([[exp]])[0]
        salary = round(salary, 2)

    return render_template("index.html", salary=salary)

if __name__ == "__main__":
    app.run(debug=True)
