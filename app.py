from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('model_pickle/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('model_pickle/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_pickle/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['POST', 'GET'])
def predict_heart_failure():
    if request.method == 'POST':

        """
        This is taking the values from the form in index.html.
        """
        features = {
            "Age": request.form['Age'],
            "Sex": request.form['Sex'],
            "ChestPainType": request.form['ChestPainType'],
            "RestingBP": request.form['RestingBP'],
            "Cholesterol": request.form['Cholesterol'], 
            "FastingBS": request.form['FastingBS'],
            "RestingECG": request.form['RestingECG'],
            "MaxHR": request.form['MaxHR'],
            "ExerciseAngina": request.form['ExerciseAngina'],
            "Oldpeak": request.form['Oldpeak'],
            "ST_Slope": request.form['ST_Slope']
        }

        """
        We are onehotencoding the categorical features.
        """
        categorical_features = [
            features['Sex'],
            features['ChestPainType'],
            features['RestingECG'],
            features['ExerciseAngina'],
            features['ST_Slope']
        ]
        encoded_cols = encoder.transform([categorical_features])

        """
        We are converting the numerical features to float values.
        """
        numerical_features = [
            float(features['Age']),
            float(features['RestingBP']),
            float(features['Cholesterol']),
            float(features['FastingBS']),
            float(features['MaxHR']),
            float(features['Oldpeak'])
        ]

        """
        We are joining the catagorical and numercial features.
        """
        combined_features = np.concatenate([numerical_features, encoded_cols[0]])

        """
        scaling features using standard scaler
        """
        scaled_features = scaler.transform([combined_features])

        """
        Predicting the output using LogisticRegression
        """
        prediction = model.predict(scaled_features)

        """
        Sending the result to result.html for printing.
        """
        return render_template('result.html', prediction=prediction[0])

    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)