from flask import Flask, render_template, request
import pickle
import numpy as np
from keras.models import load_model
loaded_model= load_model('parkinson_neural_model.h5')

app = Flask(__name__)

# Load saved StandardScaler, PCA, and SVM Model
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("pca.pkl", "rb") as file:
    pca = pickle.load(file)

with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")  # Show input form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from form
        input_data = [
            float(request.form[key]) for key in [
                "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
                "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
                "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
                "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", 
                "spread1", "spread2", "D2", "PPE"
            ]
        ]

        # Convert to NumPy array
        input_array = np.array(input_data).reshape(1, -1)

        # Apply StandardScaler and PCA
        input_scaled = scaler.transform(input_array)
        input_pca = pca.transform(input_scaled)

        # Make prediction
        out = loaded_model.predict(input_pca)[0]
        if out>0.5:
            prediction=1
        else:
            prediction=0

        # Return result page
        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
