
from glob import glob
# import
import re
import json
import torch
import os
import math
import random
import csv
import sys
import re

import random
import pandas as pd
from tqdm import tqdm
from simplet5 import SimpleT5
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Import Section
import os
import math
import random
import csv
import sys
import re

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

counter = 1
tweets = []
label = []
csv.field_size_limit(2000 * 2024 * 2024)
counterP = 0
counterN = 0
lineCounter = 0

testTweets = []
y_test = []
# Data load function
def load_sentiment_dataset(random_seed = 1, file_path="/scratch/mbiswas2/Dataset/TrainingSet_GPT_3_6_7_1_2_5_4_Augmented.txt"):

    with open(file_path, 'r') as f:
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
            #print(lineCounter)
            if(line[1].strip()== '1'):
              label.append("Positive")
             # counterP = counterP+1
            else:
              label.append("Negative")
             # counterN = counterN+1
        #print(label)

    X_train = np.array(tweets)
    Y_train = np.array(label)
   # print("The value of Positive in Training Set "+ str(counterP))
   # print("The value of Negative in Training Set "+ str(counterN))

    ## Test Set Prediction Module
    testTweets = []
    y_test = []
    csv.field_size_limit(500 * 1024 * 1024)
    counterP = 0
    counterN = 0
    with open('/scratch/mbiswas2/Dataset/ValidationSet.txt', 'r') as f:
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
              #counterP=counterP+1
            else:

              y_test.append("Negative")
              #counterN=counterN+1
   # print("The value of Positive in Validation Set "+ str(counterP))
   # print("The value of Negative in Validation Set "+ str(counterN))
    X_test = np.array(testTweets)

    # transform to pandas dataframe
    train_data = pd.DataFrame({'source_text': X_train, 'target_text': Y_train})    
    test_data = pd.DataFrame({'source_text': X_test, 'target_text': y_test})    

    # return
    return train_data, test_data



for trial_no in range(3):
    # create data
    train_df, test_df = load_sentiment_dataset(trial_no)    
    # load model
    model = SimpleT5()
    model.from_pretrained(model_type="t5", model_name="t5-base")
    # train model
    model.train(train_df=train_df,
                eval_df=test_df, 
                source_max_token_len=300, 
                target_max_token_len=200, 
                batch_size=8, 
                max_epochs=2, 
                outputdir = "outputs",
                use_gpu=True
               )
    # fetch the path to last model
    last_epoch_model = None 
    for file in glob("./outputs/*"):
        if 'epoch-1' in file:
            last_epoch_model = file
    # load the last model
    model.load_model("t5", last_epoch_model, use_gpu=True)
    # test and save
    # for each test data perform prediction
    predictions = []
    for index, row in test_df.iterrows():
        prediction = model.predict(row['source_text'])[0]
        predictions.append(prediction)
    df = test_df.copy()
    df['predicted'] = predictions
    df['original'] = df['target_text']
    print(f1_score(df['original'], df['predicted'], average='macro'))
    df.to_csv(f"result_run_{trial_no}.csv", index=False)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(df['original'], df['predicted'])

    # Print confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Print classification report
    target_names = ['negative', 'positive']
    print(classification_report(df['original'], df['predicted'], target_names=target_names))

    # Save results to an Excel file
    df_training_data = pd.DataFrame({'Tweets': tweets, 'Label': label})
    df_tweets = pd.DataFrame({'Tweets': testTweets, 'Predicted_Label': predictions, 'Actual_Label': y_test})
    df_evaluation_report = classification_report(df['original'], df['predicted'], target_names=target_names, output_dict=True)

    FilePath = "/home/mbiswas2/ondemand/test-site-venv-3.9.9-no-sys-pack-compute-node/research/Result/"+"TrainingSet_GPT_3_6_7_1_2_5_4_Augmentedn"+".xlsx"

    with pd.ExcelWriter(FilePath) as writer:
        df_training_data.to_excel(writer, sheet_name='Training_Data', index=False)
        df_tweets.to_excel(writer, sheet_name='Tweets', index=False)
        pd.DataFrame(df_evaluation_report).transpose().to_excel(writer, sheet_name='Evaluation_Report')
        pd.DataFrame(conf_matrix).to_excel(writer, sheet_name='Confusion_Matrix')