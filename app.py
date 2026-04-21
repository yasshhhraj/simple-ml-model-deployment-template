from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
# Ensure 'model.joblib' exists from previous steps where the model was saved
try:
    model = joblib.load('model.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: model.joblib not found. Please ensure the model is trained and saved.")
    model = None # Set model to None to prevent errors if not found

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json(force=True)

        # Expecting input like: {'age': 45, 'annual_salary': 60000}
        age = data.get('age')
        annual_salary = data.get('annual_salary')

        if age is None or annual_salary is None:
            return jsonify({'error': 'Please provide both age and annual_salary in the request body.'}), 400

        # Create a DataFrame for prediction
        # Ensure column names match the training data features ('age', 'annual Salary')
        input_data = pd.DataFrame([[age, annual_salary]], columns=['age', 'annual Salary'])

        prediction = model.predict(input_data)[0]

        return jsonify({'car_purchase_amount_prediction': prediction.item()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# To run the Flask app
# For deployment, consider using a production-ready WSGI server like Gunicorn or uWSGI
# Running on port 80 requires root privileges (sudo) or proper port forwarding setup.
# For local development, port 5000 is common: app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # Attempt to run on port 80 as requested
    # In Colab, you might need ngrok or similar for public access, or run on a higher port
    try:
        app.run(host='0.0.0.0', port=80)
    except PermissionError:
        print("\nPermission denied for port 80. Try running on a higher port, e.g., 5000:")
        print("app.run(host='0.0.0.0', port=5000)")
    except Exception as e:
        print(f"An error occurred: {e}")