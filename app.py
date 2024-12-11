from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model and feature columns
model = joblib.load('rf_model.joblib')
feature_columns = joblib.load('feature_columns.joblib')

# Print feature columns for debugging
print("Feature Columns Loaded:", feature_columns)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Get user input from the form
            model_year = float(request.form.get("model_year"))
            milage = float(request.form.get("milage"))
            brand = request.form.get("brand")  # Get the brand
            accident = int(request.form.get("accident"))  # Get accident status
            clean_title = int(request.form.get("clean_title"))  # Get clean title status
            
            # Create input array with all required features
            input_data = pd.DataFrame(columns=feature_columns)  # Initialize with all columns
            input_data.loc[0] = 0  # Fill with zeros initially
            
            # Set only the values we have
            input_data.loc[0, 'milage'] = milage
            input_data.loc[0, 'model_year'] = model_year
            input_data.loc[0, brand] = 1  # Set the selected brand to 1 (one-hot encoding)
            input_data.loc[0, 'accident_yes'] = accident  # Set accident status
            input_data.loc[0, 'clean_title_yes'] = clean_title  # Set clean title status

            # Check the input_data to ensure it has the correct feature names
            print("Input Data for Prediction:", input_data)

            # Make prediction
            prediction = model.predict(input_data)
            
            # Convert prediction to price
            predicted_price = np.exp(prediction[0])  # Since we used log transformation
            
            return render_template("result.html", 
                                 prediction=f"${predicted_price:,.2f}",
                                 milage=f"{milage:,.0f}",
                                 year=f"{model_year:.0f}")
        except Exception as e:
            return render_template("result.html", 
                                 error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True) 