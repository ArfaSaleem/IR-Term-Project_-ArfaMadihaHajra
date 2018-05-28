# IR-Term-Project_-ArfaMadihaHajra

Three data sets were chosen to run three classifiers i.e. Navie Bayes, Rocchio and KNN.

Used Libraries/Intsall first:
1) Numpy  2)NLTK  3)textblob 4)Spacy 5)sklearn 6)pandas

#############
First DataSet:
Download folder. Open First DataSet folder ".. " and run "Data Set 1- dbWorld.ipynb" by changing the dataset paths in the code.

First data set was taken from online source: https://archive.ics.uci.edu/ml/datasets/DBWorld+e-mails
The data was first coverted into csv file from weka.
Train test variables are defined on the data with test size = 0.3.
Sklearn libraries are used to classify the data.
Results: KNN has 0.76 precision, Rocchio has 0.85 and Naive Bayes has 0.75. Rocchio provides the best results in this case.

#############
Second DataSet:
Download folder. Open Second DataSet folder "Data Set 2- Sentence Corpus" and run "DataSet 2- Sentence Corpus.ipynb".

Second data set was taken from online source: https://archive.ics.uci.edu/ml/datasets/Sentence+Classification
It is recommended to install the textblob libraries first.
The .txt files as read first and then pre-processed are used to create training and test data which is then classified.
Results: Acuuracy of Naive Bayes = 0.74, Rocchio = 0.68 and KNN= 0.74. So, Naive bayes and KNN works well in this case.

##############
Third DataSet:
Download folder. Open Third DataSet folder "Data Set 3- Health Tweets" and run "Pre-Processing Files.ipynb" to pre-process data.
Run "Classifier-Data3.ipynb" to run classifiers.

Third data set was taken from online source: https://archive.ics.uci.edu/ml/datasets/Sentence+Classification
Download libraries.
The .txt files are read first and pre-processed. The pre-processed files are then clustered as no labels are provided.
The labeled result is stored as "health_tweets_labeled.csv" in dataset folder.
The .csv file is used and classifiers are applied on it.
Results: Acuuracy of Naive Bayes = 0.49 aprox 0.5, Rocchio = 0.63 and KNN= 0.74. So, Naive bayes and KNN works well in this case.



