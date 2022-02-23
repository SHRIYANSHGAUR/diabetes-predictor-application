#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import PolynomialFeatures
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('land.html')

@app.route('/software')
def software():
    return render_template('home.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    # this fetches data from the client side interface!
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # we need an array for preidcitions
    # we load the array of values receved from user into our model

    poly_reg = PolynomialFeatures(degree=2)

    prediction = model.predict(poly_reg.fit_transform(final_features))


    if (prediction > 0.8):
        output=" DIABETIC- PLAESE TAKE CARE OF YOUR ROUTINE AND MAKE LIFESTYLE CHANGES . Probability  of being diabetic is HIGH !"
    else:
        output=" NON-DIABETIC- Congratulations . Your Lifestyle is Brilliant. Stay healthy stay safe! "

    return render_template('index.html', prediction_text='PRIDICTIONS => :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)