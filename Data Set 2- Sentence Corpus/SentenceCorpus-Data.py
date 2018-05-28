##Importing Libraries
import os
import json
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from textblob.classifiers import NLTKClassifier
import time
import nltk.classify
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
import re

import nltk
nltk.download('punkt')

##Classifier classes
class SVMClassifier(NLTKClassifier):
    nltk_class = nltk.classify.SklearnClassifier(LinearSVC())

class RocchioClassifier(NLTKClassifier):
    nltk_class = nltk.classify.SklearnClassifier(NearestCentroid())

class KNNClassifier(NLTKClassifier): 
    nltk_class = nltk.classify.SklearnClassifier(KNeighborsClassifier())

##################
##directory
def files_from_directory(directory): #Lists all file paths from given directory
    ret_val_direc = []
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            ret_val_direc.append(str(directory) + "/" + str(file))
    return ret_val_direc

#########################
##Defintions to read files
def read_file(path):         #read linewise all files
    f = open(path, 'r')      #open files from path in read mode
    read = f.readlines()     #readlines
    ret_val_lines = []       #linedata
    for line in read:
        if line.startswith("#"):      #ommiting #edlines
            pass
        else:
            ret_val_lines.append(line)     #append line data in ret_val_lines
    return ret_val_lines                   #return ret_val_lines


#######################
#Processiong read lines- category and sentence
def process_lines(line):                       #Returns sentence category and sentence of given line

    if "\t" in line:                               ##if category and sentence seperated with \t
        splits = line.split("\t")                  #split
        sen_category = splits[0]                   #first split in category
        sentence = splits[1].lower()               #next split in sentence
        
        for sw in stopwords:                        #stop words removal
            sentence = sentence.replace(sw, "")     #replcae  
        pattern = re.compile("[^\w']")              #recomipling sentences
        sentence = pattern.sub(' ', sentence)       #remove spaces 
        sentence = re.sub(' +', ' ', sentence)      #remove  + before sentence
        return sen_category, sentence               #return recompiled sentence
    
    else:                                          ##if category and sentence split with space
        splits = line.split(" ")                    #split first
        sen_category = splits[0]                      #first space split in category
        sentence = line[len(sen_category)+1:].lower()      #len of category+1 is put in sentence- lower case
       
        for sw in stopwords:                        #stop words removal
            sentence = sentence.replace(sw, "")
        pattern = re.compile("[^\w']")              #recomplie sentence
        sentence = pattern.sub(' ', sentence)       #remove spcae in sentence
        sentence = re.sub(' +', ' ', sentence)      #remove + and spcae before sentence
        return sen_category, sentence               #return compiled sentence

###############################
##Writing training data in json file
def create_json_file(input_folder, destination_file):
    tr_folder = files_from_directory(input_folder)
    all_in_json = []     
    
    for file in tr_folder:                  #accessing input training files
        lines = read_file(file)             #reading files
        for line in lines:                     
            c, s = process_lines(line)       #getting line category and sentence
            if s.endswith('\n'):            #if sentence ends with \n 
                s = s[:-1]                  #sentence before \n
            json_data = {                   #json data
                'text': s,                  #text as s
                'label': c                  #label as c
            }
            all_in_json.append(json_data)    #append data

    with open(destination_file, "w") as outfile:    #write output file
        json.dump(all_in_json, outfile)


##Mapping each sentence to its category
def prepare_test_data(input_folder):
    """Maps each sentence to it's category"""

    test_folder = files_from_directory(input_folder)
    t_sentences = []                                #test sentences
    t_categories = []                               #test categories
    for file in test_folder:                        
        lines = read_file(file)
        for line in lines:
            c, s = process_lines(line)
            if s.endswith('\n'):
                s = s[:-1]
            t_sentences.append(s)
            t_categories.append(c)
    return t_categories, t_sentences                #return test categories and test sentences

##Main

#######################
##Loading stopword file
input_stopwords = read_file("C:/Users/HP/Desktop/word_lists/stopwords.txt")
stopwords = []
for word in input_stopwords:
    if word.endswith('\n'):             #extracting stopwords
        word = word[:-1]
        stopwords.append(word)

        
#########################
##Training and test data

create_json_file("training_set", "training.json")
categories, sentences = prepare_test_data("test_set")


###################################
##Naive Bayes Classifier
print("######### Naive Bayes######")

##Training Naive Bayes
print("Training :: Naive Bayes Classifier...")
start_nbc = time.time()                                #start time

with open('training.json', 'r') as training:
    nbc = NaiveBayesClassifier(training, format = "json")        #pass training data to NaiveBayesClassifier
    
stop_nbc = time.time()
print("Training of Naive Bayes Classifier is completed...")

elapsed_time = stop_nbc - start_nbc
print("Recorded Training time (in seconds) = " + str(elapsed_time))

##Testing Naive Bayes
print("Testing :: Naive Bayes Classifier...")                    
correct_predict = 0
start_nbc = time.time()

for i in range(0, len(sentences)):
    category = str(nbc.classify(sentences[i])).lower()
    expected = str(categories[i]).lower()
    if category == expected:
        correct_predict += 1
        
stop_nbc = time.time()
elapsed_time = stop_nbc - start_nbc

print("Number of tests performed = " + str(len(sentences)))
print("Correct labels identified = " + str(correct_predict))

accuracy = correct_predict / len(sentences)                      #checking accuracy
print("Naive Bayes Classifier's accuracy = " + str(accuracy))
print("Recorded Testing time (in seconds) = " + str(elapsed_time))


#########################################
##Rocchio
print("######### Rocchio ########")
print("Training :: Rocchio Classifier...")
start_rocchio = time.time()

##Training Rocchio
with open('training.json', 'r') as training:
    rocchio = RocchioClassifier(training, format = "json")
stop_rocchio = time.time()

print("Training Rocchio Classifier completed...")
elapsed_time_r = stop_rocchio - start_rocchio
print("Recorded Training time (in seconds) = " + str(elapsed_time_r))

##Testing Rocchio
print("Testing :: Rocchio Classifier...")
correct_pred_r = 0
start_rocchio = time.time()

for i in range(0, len(sentences)):
    category = str(rocchio.classify(sentences[i])).lower()
    expected = str(categories[i]).lower()
    if category == expected:
        correct_pred_r += 1
        
stop_rocchio = time.time()
elapsed_time_r = stop_rocchio - start_rocchio

print("Number of tests performed = " + str(len(sentences)))
print("Correct labels identified = " + str(correct_pred_r))

accuracy_r = correct_pred_r / len(sentences)
print("Rocchio Classifier accuracy = " + str(accuracy_r))
print("Recorded Testing time (in seconds) = " + str(elapsed_time_r))


############################################
##KNN
print("######### KNN ########")

##Training KNN
print("Training :: K-Nearest Neighbor Classifier...")

start_knn = time.time()
with open('training.json', 'r') as training:
    knn = KNNClassifier(training, format = "json")
stop_knn = time.time()
print("Training K-Nearest Neighbor Classifier completed...")

elapsed_time_k = stop_knn - start_knn
print("Recorded Training time (in seconds) = " + str(elapsed_time_k))

##Testing KNN
print("Testing :: K-Nearest Neighbor Classifier...")
correct_pred_K = 0
start_knn = time.time()

for i in range(0, len(sentences)):
    category = str(nbc.classify(sentences[i])).lower()
    expected = str(categories[i]).lower()
    if category == expected:
        correct_pred_K += 1
stop_knn = time.time()

elapsed_time_k = stop_knn - start_knn
print("Number of tests performed = " + str(len(sentences)))
print("Correct labels identified = " + str(correct_pred_K))

accuracy = correct_pred_K / len(sentences)
print("K-Nearest Neighbor Classifier accuracy = " + str(accuracy))
print(" Recorded Testing time (in seconds): " + str(elapsed_time_k))