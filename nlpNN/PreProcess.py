import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
import multiprocessing
from collections import Counter
lemmatizer = WordNetLemmatizer()
import csv




def create_lexicon(sentance_data):
    
    lexicon = list()
    paragraph = str()

    for sent in sentance_data:
        paragraph += " " + sent.lower()
    lexicon = word_tokenize(paragraph)
    w_counts = Counter(lexicon)
    l2 = set()
    for w in w_counts:
        l2.add(w)

    with open("./supportfiles/lexicon.pickle",'wb') as f:
        pickle.dump(lexicon, f)
       
    return list(l2)

def sample_handling(sentance_data, label_data, lexicon):
    featureset= list() 
    print("Progress [", end= " ")
    j = 0.1
    for i in range(0,len(sentance_data)):
        if i == int(len(sentance_data)*j):
            print(".", end = " ")
            j += .1
        current_words = word_tokenize(sentance_data[i])
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))
        for word in current_words:
            if word.lower() in lexicon:
                idxVal = lexicon.index(word)
                features[idxVal] = 1
        features = list(features)
        featureset.append([features,label_data[i]])
    print("]")
    with open("./supportfiles/featureset.pickle",'wb') as f:
        pickle.dump(features, f)

    return featureset


def create_feature_set_and_labels(file):
    sentance_data = list()
    label_data = list()

    with open('t_data.csv') as myfile:
        reader = csv.reader(myfile, delimiter = '*')
        for row in list(reader):
            sentance_data.append(row[0].lower())
            label_data.append(row[1].lower())
    
    HashSet = list(set(label_data))
    
    label_data = [HashSet.index(label.lower()) for label in label_data]
    labels = list()
    for x in label_data:
        sets = [0 for i in range(0,len(HashSet))]
        sets[x] = 1
        labels.append(sets)
    label_data = labels
    
    print("DONE Reading Data")

    lexicon = create_lexicon(sentance_data)
   
    print("DONE Building Lexicon")
    
    features = sample_handling(sentance_data,label_data,lexicon)

    print("DONE building featureset")
    

    train_x = [x[0] for x in features]
    train_y = [x[1] for x in features]

  
    
    return train_x,train_y,len(HashSet),HashSet

def fromFeatureSet(inpt):
    with open("./supportfiles/featureset.pickle",'rb') as f:
        lexicon = pickle.load(f)
    cur_word = word_tokenize(inpt)
    cur_word = [lemmatizer.lemmatize(i) for i in cur_word]
    features = np.zeros(len(lexicon))
    for word in cur_word:
        if word.lower() in lexicon:
            idxVal = lexicon.index(word)
            features[idxVal] = 1
    features = list(features)
    return features
    

