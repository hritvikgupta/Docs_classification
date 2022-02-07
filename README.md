## Document Classification
Document Classification is a NLP application which trains Neural Network to recognize different documents based on text and images present in them. Therefore Natural Langauge processing plays a vital role in converting the text to feature present in images alongside converting images into feature array using OCR. PyTesseract is used to convert images into image array and extracting the text in them. And Further classifying them based on trained model. This Is NLP base Text Document identifier. For Example if some paper is about politics it will identify it as a political paper, similarly with other types of documents

#### Dataset: 
             * BBC new dataset consist of the categories Tech, Bussiness, Sports, Politics and Entertainment. 
             * Train Dataset: 1490 records
             * Test Dataset:  736
             
https://www.kaggle.com/c/learn-ai-bbc/data?select=BBC+News+Train.csv


* Steps involved:-

      * 1)  Convert the image into the Text using OCR
      * 2)  Cleaning text.
      * 3)  Making the text embedding
      * 4)  spliting the train test using tensorflow dataset API
      * 5)  Feeding the data to model - This Uses Reccurent Neural Network Which is Long short Term memory (LSTM) model for training
      * 6)  Creating Custom CallBacks
      * 7)  After training getting the prediction
      * 8)  Converting the prediction into the classifcaiton label such as politics, medical etc.
