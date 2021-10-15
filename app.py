import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import *
app = Flask(__name__)
lr_clf = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    location = int_features[0]
    sqft = int_features[1]
    bath = int_features[2]
    bhk = int_features[3]

    prediction = predict_price(location,sqft,bath,bhk)

    output = round(prediction, 2)

    return render_template('index.html', prediction_text='House Price should be  {} Lakhs'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = predict_price([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)