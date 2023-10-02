from flask import Flask, jsonify, request, render_template 
import torch 
import numpy as np
from take_home import Arima_module

app = Flask(__name__)

# Load the saved prediction
values = np.load('predicted_values.npy').tolist()
 
# Load the trained model 
model = Arima_module()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    month_number = int(request.form['month_number'])
    prediction = values[month_number - 1]
    return render_template('results.html', prediction = prediction, month_number = month_number)

if __name__ == '__main__':
    app.run(debug=True)