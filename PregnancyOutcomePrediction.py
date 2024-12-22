import os
import math
import random
import csv
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import statistics as stats
from tqdm import tqdm

from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import BertTokenClassifier
from bert_sklearn import load_model

# Import Section
import csv
import codecs
import sys
import io
import numpy as np
import pandas as pd


# For HashtagSegmentation
# pip install ekphrasis


# For Classifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Python script for confusion matrix creation.
from sklearn.metrics import *

#Global Initialization Section
# seg = Segmenter(corpus="twitter")

counter = 1
tweets = []
label = []
csv.field_size_limit(2000 * 2024 * 2024)
counterP = 0
counterN = 0
lineCounter = 0
with open('/home/mbiswas2/ondemand/test-site-venv-3.9.9-no-sys-pack-compute-node/research/Data Set/GPT Augmented Data/TrainingSet_GPT_3_6_7_1_2_5_4_Augmented.txt', 'r') as f:
    next(f) # skip headings
    #reader=csv.reader(f, dialect="excel-tab")
    Lines = f.readlines()
    for eline in Lines:
        line = eline.split("\t")
        #print(line[0])
        #print(type(line[1]))
        #preProcessedTweetText= preProcessingModule(line[0])
        #print(preProcessedTweetText)
        tweets.append(line[0])
        #print(line[0])
        #print(line[1])
        # print(tweets)
        lineCounter = lineCounter+1
        #print(lineCounter)
        if(line[1].strip()== '1'):
          label.append("Positive")
          counterP = counterP+1
        else:
          label.append("Negative")
          counterN = counterN+1
    #print(label)

X_train = np.array(tweets)
Y_train = np.array(label)
print("The value of Positive in Training Set "+ str(counterP))
print("The value of Negative in Training Set "+ str(counterN))

## Test Set Prediction Module
testTweets = []
y_test = []
csv.field_size_limit(500 * 1024 * 1024)
counterP = 0
counterN = 0
with open('/home/mbiswas2/ondemand/test-site-venv-3.9.9-no-sys-pack-compute-node/research/Data Set/ValidationSet.txt', 'r') as f:
    next(f) # skip headings
    #reader=csv.reader(f, dialect="excel-tab")
    Lines = f.readlines()
    for eline in Lines:
        line = eline.split("\t")
        #print(line[0])
        #print(line[1])
        #preProcessedTweetText= preProcessingModule(line[0])
        #print(preProcessedTweetText)
        testTweets.append(line[0])
        if(line[1]=="1"):
          y_test.append("Positive")
          counterP=counterP+1
        else:

          y_test.append("Negative")
          counterN=counterN+1
print("The value of Positive in Validation Set "+ str(counterP))
print("The value of Negative in Validation Set "+ str(counterN))
X_test = np.array(testTweets)


model = BertClassifier(bert_config_json=None, bert_model='bert-base-multilingual-cased',
               bert_vocab=None, do_lower_case=None, epochs=3, eval_batch_size=8,
               fp16=False, from_tf=False, gradient_accumulation_steps=1,
               ignore_label=None, label_list=None, learning_rate=2e-05,
               local_rank=-1, logfile='bert_sklearn.log', loss_scale=0,
               max_seq_length=64, num_mlp_hiddens=500, num_mlp_layers=0,
               random_state=42, restore_file=None, train_batch_size=16,
               use_cuda=True, validation_fraction=0.1, warmup_proportion=0.1)
model = model.fit(X_train, Y_train)
# score model
accy = model.score(X_test, y_test)

# make class probability predictions
y_prob = model.predict_proba(X_test)
#print("class prob estimates:\n", y_prob)

# make predictions
y_pred = model.predict(X_test)
print("Accuracy: %0.2f%%"%(metrics.accuracy_score(y_pred,  y_test) * 100))

target_names = ['negative', 'positive']
print(classification_report(y_test, y_pred, target_names=target_names))

evaluation_report = classification_report(y_test, y_pred, output_dict=True)

df_training_data = pd.DataFrame({'Tweets': tweets, 'Label': label})
df_tweets = pd.DataFrame({'Tweets': testTweets, 'Predicted_Label': y_pred, 'Actual_Label': y_test})
df_evaluation_report = pd.DataFrame(evaluation_report).transpose()
# name = fileName.split("/")
FilePath = "/home/mbiswas2/ondemand/test-site-venv-3.9.9-no-sys-pack-compute-node/research/Result/"+"TrainingSet_GPT_3_6_7_1_2_5_4_Augmented_bert-bert-base-multilingual-cased"+".xlsx"
  # Save to Excel file
# with pd.ExcelWriter("/content/drive/MyDrive/Research/Result/Balance DataSet Result/Result_Balance_DataSet.xlsx") as writer:
with pd.ExcelWriter(FilePath) as writer:
      df_training_data.to_excel(writer, sheet_name='Training_Data', index=False)
      df_tweets.to_excel(writer, sheet_name='Tweets', index=False)
      df_evaluation_report.to_excel(writer, sheet_name='Evaluation_Report')


# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Regression)')

# Save the plot to a file
plt.savefig('true_vs_predicted_plot.png')  # You can specify the file format (.png, .jpg, .pdf, etc.)
plt.show()