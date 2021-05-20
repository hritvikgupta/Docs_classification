#!/usr/bin/env python
# coding: utf-8

# In[9]:
from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()

import re
from keras.preprocessing.sequence import pad_sequences
import nltk
import string
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
from PIL import Image
import pytesseract
from pytesseract import image_to_string
from keras.models import model_from_json
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import request
from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import pickle#Initialize the flask App
app = Flask(__name__, template_folder = 'templates')

@app.route('/')
def home():
    return render_template('in2.html')
def upload():
    return render_template('in2.html')

@app.route('/', methods=['POST', 'GET'])
def upload():
    return render_template("in2.html")

json_file = open('/Users/hritvikgupta/Desktop/model_20_may1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/Users/hritvikgupta/Desktop/model_20_may-1.h5")

def clean_text(text):
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split('\W+', text)
        text = " ".join([lemmer.lemmatize(word) for word in tokens if word not in stopwords])
        return text
        
def img_to_features(img_path, tokenizer):
    Img = Image.open(img_path)
    a = pytesseract.image_to_string(Img)
    test = ' '.join(s for s in a.split() if not any(c.isdigit() for c in s))
    max_len = len(tokenizer.word_index)+1
    test = clean_text(test)
    test_ = tokenizer.texts_to_sequences([test])
    test_ = pad_sequences(test_, maxlen = max_len)
    return test_
@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':  
        image = request.files['file']  
    with open('/Users/hritvikgupta/Desktop/tokenizer_20_may1.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    features  = img_to_features(image,tokenizer)
    preds = loaded_model.predict(features)
    indexes = np.where(preds > 0.8)
    classes = ['business', 'entertainment', 'food', 'graphics', 'historical',
       'medical', 'politics', 'space', 'sport', 'technologie']
    indexes = indexes[1]
    ans = []
    for i in indexes:
        ans.append((classes[i], preds[0][i]))
    return render_template('in2.html', prediction_text = "classes that image belong to is {}".format(ans))

if __name__ == "__main__":
    app.run(debug=True)


# In[12]:





# In[ ]:




