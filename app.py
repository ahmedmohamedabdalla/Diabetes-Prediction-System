import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, static_url_path='/static')


model = pickle.load(open('Random_Forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Page.html')


@app.route('/predict',methods=['POST','GET'])

def predict():
    x_1=request.form['x_1']
    x_2=request.form['x_2']
    x_3=request.form['x_3']
    x_4=request.form['x_4']
    x_5=request.form['x_5']
    x_6=request.form['x_6']
    x_7=request.form['x_7']

    arr = [x_1,x_2,x_3,x_4,x_5,x_6,x_7]
    row = np.array(arr).reshape(1,-1)
    features = scaler.transform(row.reshape(1,-1))

    prediction = model.predict(features)

    if (prediction == [1]):
        return render_template("Page.html", prediction_text = "The Patient Has Diabetes")
    else:
         return render_template("Page.html", prediction_text="The Patient Does't Have Diabetes")
  


if __name__ == "__main__":
    app.run(debug=True)