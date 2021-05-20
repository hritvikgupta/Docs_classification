#!/usr/bin/env python
# coding: utf-8

# In[9]:
from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()
#Importing required libraries
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
import re
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
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    #print(set(sent1.split(' ') + sent2.split(' ')))
    #sent1 = [w.lower() for w in sent1]
    #sent2 = [w.lower() for w in sent2]
    #print(sent1)
    all_words = []
    c= set(sent1.split(' ') + sent2.split(' '))
    for i in c:
        all_words.append(i)
   # print(all_words)
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
   # print(sent1)
    for w in sent1.split(' '):
        if w in stopwords:
            continue
       # print(w)
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2.split(' '):
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            #print(sentences[idx1])
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(sent,original, top_n=5):
    nltk.download('stopwords')
    nltk.download('punkt')
    
    stop_words = nltk.corpus.stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences = sent

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(original)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    
    
    ans = []
    for i in range(top_n):
         ans.append(ranked_sentence[i][1])
    return ans     

# let's begin 
def img_to_features(img_path, tokenizer):
    Img = Image.open(img_path)
    a = pytesseract.image_to_string(Img)
    test = ' '.join(s for s in a.split() if not any(c.isdigit() for c in s))
    max_len = len(tokenizer.word_index)+1
    test_ = clean_text(test)
    test_ = tokenizer.texts_to_sequences([test_])
    test_ = pad_sequences(test_, maxlen = max_len)
    return test_, test
@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':  
        image = request.files['file']  
    with open('/Users/hritvikgupta/Desktop/tokenizer_20_may1.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    features , text0 = img_to_features(image,tokenizer)
    text1= [ clean_text(i) for i in nltk.sent_tokenize(text0)]
    text2 = [i for i in nltk.sent_tokenize(text0) ]
    summary = generate_summary(text1, text2, 3)
    preds = loaded_model.predict(features)
    indexes = np.where(preds > 0.8)
    classes = ['business', 'entertainment', 'food', 'graphics', 'historical',
       'medical', 'politics', 'space', 'sport', 'technologie']
    indexes = indexes[1]
    ans = []
    for i in indexes:
        ans.append((classes[i], preds[0][i]))
    return render_template('in2.html', prediction_text = "classes that image belong to is {} and summary of the text is {}".format(ans, summary))

if __name__ == "__main__":
    app.run(debug=True)


# In[12]:





# In[ ]:




