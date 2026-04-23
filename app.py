import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load model
model = pickle.load(open("collabera_hr_model.pkl", "rb"))

# Simple HTML template (clean + modern)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>HR Prediction App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff;
            text-align: center;
            padding: 50px;
        }
        .container {
            background: #ffffff;
            color: #333;
            padding: 30px;
            border-radius: 12px;
            width: 350px;
            margin: auto;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }
        input {
            width: 90%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        button {
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        button:hover {
            background: #5a67d8;
        }
        h1 {
            margin-bottom: 20px;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
        }
    </style>
</head>
<body>
    <h1>💼 HR Prediction App</h1>
    <div class="container">
        <form method="POST" action="/predict">
            <input type="number" step="any" name="feature1" placeholder="Feature 1" required>
            <input type="number" step="any" name="feature2" placeholder="Feature 2" required>
            <input type="number" step="any" name="feature3" placeholder="Feature 3" required>
            <button type="submit">Predict</button>
        </form>
        {% if prediction %}
            <div class="result">Prediction: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["feature1"]),
            float(request.form["feature2"]),
            float(request.form["feature3"])
        ]

        final_features = np.array([features])
        prediction = model.predict(final_features)[0]

        return render_template_string(HTML_TEMPLATE, prediction=prediction)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
