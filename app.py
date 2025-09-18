from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar modelo y encoders
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        data = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'bmi': float(request.form['bmi']),
            'alcohol_consumption': request.form['alcohol_consumption'],
            'smoking_status': request.form['smoking_status'],
            'hepatitis_b': int(request.form['hepatitis_b']),
            'hepatitis_c': int(request.form['hepatitis_c']),
            'liver_function_score': float(request.form['liver_function_score']),
            'alpha_fetoprotein_level': float(request.form['alpha_fetoprotein_level']),
            'cirrhosis_history': int(request.form['cirrhosis_history']),
            'family_history_cancer': int(request.form['family_history_cancer']),
            'physical_activity_level': request.form['physical_activity_level'],
            'diabetes': int(request.form['diabetes'])
        }
        
        # Crear DataFrame
        df = pd.DataFrame([data])
        
        # Codificar variables categóricas
        categorical_cols = ['gender', 'alcohol_consumption', 'smoking_status', 'physical_activity_level']
        for col in categorical_cols:
            df[col] = encoders[col].transform([data[col]])[0]
        
        # Realizar predicción
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        # Preparar resultado
        result = {
            'prediction': int(prediction),
            'probability_no_cancer': round(probability[0] * 100, 2),
            'probability_cancer': round(probability[1] * 100, 2)
        }
        
        return render_template('result.html', **result)
        
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)