from flask import Flask, jsonify, request, render_template 
import torch 
import numpy as np
from take_home import Arima_module, load_data, plot_df, PlotAcf, Plot_Pafc

app = Flask(__name__)

# Load the saved prediction
values = np.load('predicted_values.npy').tolist()
 
# Load the trained model 
model = Arima_module()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

#load data
df_full, _, _ = load_data('data_daily.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    month_number = int(request.form['month_number'])
    prediction = values[month_number - 1]
    
    plot_2021 = plot_df(df_full['Month'].values, df_full['Receipt_Count'].values, title='Monthly receipts count in 2021')
    plot_2022 = plot_df(df_full['Month'].values, values, title='Monthly receipts count in 2022')
    data_series = df_full['Receipt_Count']
    train_after_diff = data_series.diff().dropna().values
    acf=PlotAcf(train_after_diff)
    pacf = Plot_Pafc(train_after_diff)
    return render_template('results.html', prediction = prediction, month_number = month_number, plot_2021=plot_2021, plot_2022=plot_2022, acf = acf, pacf = pacf)

if __name__ == '__main__':
    app.run(debug=True)