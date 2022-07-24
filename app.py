# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 09:59:32 2022

@author: Dell
"""

import numpy as np
import os
from keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='sample1.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)
    preds = model.predict(x)
    a=preds[0]
    m=a[0]
    z=0
    for i in range(1,len(a)):
        if a[i]>m:
            m=a[i]
            z=i
    if z==0:
        preds="Brown Spot. A simple cure is spray 1g of ediphenphos or 2g mancozeb or 2.25g Zineb in 1liter of water."
    else:
        preds="Leaf Blight. Cures for this disease are Terramycin 17, Brestanol, Agrimycin 500 and a combination of Agrimycin 100 + Fytolan."
    
        
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
