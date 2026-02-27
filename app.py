from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
  
# Load Models
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")
kidney_model = joblib.load("kidney_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

# ================= DIABETES =================
@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    result = None
    if request.method == "POST":
        data = [
            int(request.form["preg"]),
            int(request.form["glucose"]),
            int(request.form["bp"]),
            int(request.form["skin"]),
            int(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            int(request.form["age"])
        ]
        prediction = diabetes_model.predict([data])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render_template("diabetes.html", result=result)

# ================= HEART =================
@app.route("/heart", methods=["GET", "POST"])
def heart():
    result = None
    if request.method == "POST":
        data = [
            int(request.form["age"]),
            int(request.form["sex"]),
            int(request.form["cp"]),
            int(request.form["trestbps"]),
            int(request.form["chol"]),
            int(request.form["fbs"]),
            int(request.form["restecg"]),
            int(request.form["thalach"]),
            int(request.form["exang"]),
            float(request.form["oldpeak"]),
            int(request.form["slope"]),
            int(request.form["ca"]),
            int(request.form["thal"])
        ]
        prediction = heart_model.predict([data])
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    return render_template("heart.html", result=result)

# ================= KIDNEY =================
@app.route("/kidney", methods=["GET", "POST"])
def kidney():
    result = None
    if request.method == "POST":
        data = [
            int(request.form["age"]),
            int(request.form["bp"]),
            float(request.form["sg"]),
            int(request.form["al"]),
            int(request.form["su"]),
            int(request.form["bgr"]),
            int(request.form["bu"]),
            float(request.form["sc"]),
            float(request.form["hemo"]),
            int(request.form["pcv"]),
            int(request.form["wc"]),
            float(request.form["rc"])
        ]
        prediction = kidney_model.predict([data])
        result = "Kidney Disease Detected" if prediction[0] == 1 else "No Kidney Disease"

    return render_template("kidney.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
