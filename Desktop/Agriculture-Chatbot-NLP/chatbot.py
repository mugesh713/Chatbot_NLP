import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow

def trainIntentModel():
    dataset = pd.read_csv('datasets/intent.csv', names=["Query", "Intent"])
    X = dataset["Query"]
    y = dataset["Intent"]

    unique_intent_list = list(set(y))
    print("Intent Dataset successfully loaded!")
    
    queryCorpus = []
    ps = PorterStemmer()

    for query in X:
        query = re.sub('[^a-zA-Z]', ' ', query)
        query = query.split(' ')
        tokenized_query = [ps.stem(word.lower()) for word in query]
        tokenized_query = ' '.join(tokenized_query)
        queryCorpus.append(tokenized_query)
        
    print(queryCorpus)
    print("Corpus created")
    
    countVectorizer= CountVectorizer(max_features=800)
    corpus = countVectorizer.fit_transform(queryCorpus).toarray()
    print(corpus.shape)
    print("Bag of words created!")
    
    pk.dump(countVectorizer, open("saved_state/IntentCountVectorizer.sav", 'wb'))
    print("Intent CountVectorizer saved!")
    
    labelencoder_intent = LabelEncoder()
    y = labelencoder_intent.fit_transform(y)
    y = to_categorical(y)
    print("Encoded the intent classes!")
    print(y)
    
    res = {}
    for cl in labelencoder_intent.classes_:
        res.update({cl:labelencoder_intent.transform([cl])[0]})

    intent_label_map = res
    print(intent_label_map)
    print("Intent Label mapping obtained!")
    
    classifier = Sequential()
    classifier.add(Dense(units = 96, kernel_initializer = 'uniform', activation = 'relu', input_dim = 133))
    classifier.add(Dense(units = 96, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(corpus, y, batch_size = 10, epochs = 500)
    
    return classifier, intent_label_map

intent_model, intent_label_map = trainIntentModel()
intent_model.save('saved_state/intent_model.h5')
print("Intent model saved!")

def trainEntityModel():
    dataset = pd.read_csv('datasets/data-tags.csv', names=["word", "label"])
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    print(X)
    print("Entity Dataset successfully loaded!")

    entityCorpus=[]
    ps = PorterStemmer()

    for word in X.astype(str):
        word = [ps.stem(word[0])]
        entityCorpus.append(word)
    
    print(entityCorpus)
    X = entityCorpus
    from numpy import array
    X = array(X)
    X = X.reshape(len(X),)
    
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(X).toarray()
    print("Entity Bag of words created!")
    
    pk.dump(cv, open('saved_state/EntityCountVectorizer.sav', 'wb'))
    print("Entity CountVectorizer state saved!")
    
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y.astype(str))
    print("Encoded the entity classes!")
    
    res = {}
    for cl in labelencoder_y.classes_:
        res.update({cl:labelencoder_y.transform([cl])[0]})
    entity_label_map = res
    print("Entity Label mapping obtained!")
    
    classifier = GaussianNB()
    classifier.fit(X, y)
    print("Entity Model trained successfully!")
    
    pk.dump(classifier, open('saved_state/entity_model.sav', 'wb'))
    print("Trained entity model saved!")
    
    return entity_label_map

entity_label_map = trainEntityModel()
loadedEntityCV = pk.load(open('saved_state/EntityCountVectorizer.sav', 'rb'))
loadedEntityClassifier = pk.load(open('saved_state/entity_model.sav', 'rb'))

def getEntities(query):
    query = loadedEntityCV.transform(query).toarray()
    response_tags = loadedEntityClassifier.predict(query)
    entity_list=[]
    for tag in response_tags:
        if tag in entity_label_map.values():
            entity_list.append(list(entity_label_map.keys())[list(entity_label_map.values()).index(tag)])
    return entity_list

import json
import random

with open('datasets/intents.json') as json_data:
    intents = json.load(json_data)

loadedIntentClassifier = load_model('saved_state/intent_model.h5')
loaded_intent_CV = pk.load(open('saved_state/IntentCountVectorizer.sav', 'rb'))    

USER_INTENT = ""

while True:
    user_query = input()
    query = re.sub('[^a-zA-Z]', ' ', user_query)
    query = query.split(' ')
    ps = PorterStemmer()
    tokenized_query = [ps.stem(word.lower()) for word in query]
    processed_text = ' '.join(tokenized_query)
    processed_text = loaded_intent_CV.transform([processed_text]).toarray()
    predicted_Intent = loadedIntentClassifier.predict(processed_text)
    result = np.argmax(predicted_Intent, axis=1)
    
    for key, value in intent_label_map.items():
        if value==result[0]:
            USER_INTENT = key
            break
        
    for i in intents['intents']:
        if i['tag'] == USER_INTENT:
            print(random.choice(i['responses']))

    entities = getEntities(tokenized_query)
    token_entity_map = dict(zip(entities, tokenized_query))


def trainIntentModel():
    dataset = pd.read_csv('datasets/intent.csv', names=["Query", "Intent"])
    X = dataset["Query"]
    y = dataset["Intent"]

    unique_intent_list = list(set(y))
    print("Intent Dataset successfully loaded!")
    
    queryCorpus = []
    ps = PorterStemmer()

    for query in X:
        query = re.sub('[^a-zA-Z]', ' ', query)
        query = query.split(' ')
        tokenized_query = [ps.stem(word.lower()) for word in query]
        tokenized_query = ' '.join(tokenized_query)
        queryCorpus.append(tokenized_query)
        
    print(queryCorpus)
    print("Corpus created")
    
    countVectorizer= CountVectorizer(max_features=800)
    corpus = countVectorizer.fit_transform(queryCorpus).toarray()
    print(corpus.shape)
    print("Bag of words created!")
    
    pk.dump(countVectorizer, open("saved_state/IntentCountVectorizer.sav", 'wb'))
    print("Intent CountVectorizer saved!")
    
    labelencoder_intent = LabelEncoder()
    y = labelencoder_intent.fit_transform(y)
    y = to_categorical(y)
    print("Encoded the intent classes!")
    print(y)
    
    res = {}
    for cl in labelencoder_intent.classes_:
        res.update({cl:labelencoder_intent.transform([cl])[0]})

    intent_label_map = res
    print(intent_label_map)
    print("Intent Label mapping obtained!")
    
    classifier = Sequential()
    classifier.add(Dense(units = 96, kernel_initializer = 'uniform', activation = 'relu', input_dim = 133))
    classifier.add(Dense(units = 96, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(corpus, y, batch_size = 10, epochs = 500)
    
    return classifier, intent_label_map

intent_model, intent_label_map = trainIntentModel()
intent_model.save('saved_state/intent_model.h5')
print("Intent model saved!")

def trainEntityModel():
    dataset = pd.read_csv('datasets/data-tags.csv', names=["word", "label"])
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    print(X)
    print("Entity Dataset successfully loaded!")

    entityCorpus=[]
    ps = PorterStemmer()

    for word in X.astype(str):
        word = [ps.stem(word[0])]
        entityCorpus.append(word)
    
    print(entityCorpus)
    X = entityCorpus
    from numpy import array
    X = array(X)
    X = X.reshape(len(X),)
    
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(X).toarray()
    print("Entity Bag of words created!")
    
    pk.dump(cv, open('saved_state/EntityCountVectorizer.sav', 'wb'))
    print("Entity CountVectorizer state saved!")
    
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y.astype(str))
    print("Encoded the entity classes!")
    
    res = {}
    for cl in labelencoder_y.classes_:
        res.update({cl:labelencoder_y.transform([cl])[0]})
    entity_label_map = res
    print("Entity Label mapping obtained!")
    
    classifier = GaussianNB()
    classifier.fit(X, y)
    print("Entity Model trained successfully!")
    
    pk.dump(classifier, open('saved_state/entity_model.sav', 'wb'))
    print("Trained entity model saved!")
    
    return entity_label_map

entity_label_map = trainEntityModel()
loadedEntityCV = pk.load(open('saved_state/EntityCountVectorizer.sav', 'rb'))
loadedEntityClassifier = pk.load(open('saved_state/entity_model.sav', 'rb'))

def getEntities(query):
    query = loadedEntityCV.transform(query).toarray()
    response_tags = loadedEntityClassifier.predict(query)
    entity_list=[]
    for tag in response_tags:
        if tag in entity_label_map.values():
            entity_list.append(list(entity_label_map.keys())[list(entity_label_map.values()).index(tag)])
    return entity_list

import json
import random

with open('datasets/intents.json') as json_data:
    intents = json.load(json_data)

loadedIntentClassifier = load_model('saved_state/intent_model.h5')
loaded_intent_CV = pk.load(open('saved_state/IntentCountVectorizer.sav', 'rb'))    

USER_INTENT = ""

while True:
    user_query = input()
    query = re.sub('[^a-zA-Z]', ' ', user_query)
    query = query.split(' ')
    ps = PorterStemmer()
    tokenized_query = [ps.stem(word.lower()) for word in query]
    processed_text = ' '.join(tokenized_query)
    processed_text = loaded_intent_CV.transform([processed_text]).toarray()
    predicted_Intent = loadedIntentClassifier.predict(processed_text)
    result = np.argmax(predicted_Intent, axis=1)
    
    for key, value in intent_label_map.items():
        if value==result[0]:
            USER_INTENT = key
            break
        
    for i in intents['intents']:
        if i['tag'] == USER_INTENT:
            print(random.choice(i['responses']))

    entities = getEntities(tokenized_query)
    token_entity_map = dict(zip(entities, tokenized_query))
