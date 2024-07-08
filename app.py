from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the trained model
model = joblib.load('random_forest_regressor_model.pkl')

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    opening = float(request.form['opening'])
    high = float(request.form['high'])
    low = float(request.form['low'])
    volume = float(request.form['volume'])

    input_data = np.array([[opening, high, low, volume]])
    predicted_close = model.predict(input_data)

    return render_template('result.html', opening=opening, high=high, low=low, volume=volume,
                           predicted_close=predicted_close[0])

if __name__ == '__main__':
    app.run(debug=True)
