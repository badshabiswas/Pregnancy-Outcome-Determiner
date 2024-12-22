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


def is_tweet_by_someone_else(tweet):
    # Reason 1: Reference to another person (e.g., "My [Name]")
    reference_to_another_person_pattern = r'\bMy\s+\w+\b'

    # Reason 2: Expression of excitement (e.g., "excited", "can't wait")
    excitement_pattern = r'\b(?:excited|can\'t wait)\b'

    # Reason 3: No direct indication of maternity (e.g., "I'm", "my baby")
    indication_of_maternity_pattern = r'\b(?:I\'m|I am|my\s+baby)\b'

    # Check if any of the patterns are found in the tweet
    if (re.search(reference_to_another_person_pattern, tweet) or
        re.search(excitement_pattern, tweet) or
        re.search(indication_of_maternity_pattern, tweet)):
        return False  # Tweet may be given by someone else
    else:
        return True  # Tweet likely given by the mom

# Define function to extract rule-based features
def extract_rule_based_features(tweet):
    features = []

    # # Rule 1: Pregnancy reached term (Positive)
    # if "full term" in tweet.lower() or "term" in tweet.lower():
    #     features.append(1)
    # else:
    #     features.append(0)

    # # Rule 2: Baby born at normal weight (Positive)
    # if re.search(r'\bYou(?:\'re)?\s+due\b', tweet):
    #     features.append(10)  # Assign a positive weight
    #     print("It's working for 'You are due'")
    #     print(tweet)
    # else:
    #     features.append(0)


    # # Rule 3: No adverse pregnancy outcome (Positive)
    # if "pain" in tweet.lower() or "horrific" in tweet.lower() or "adverse" in tweet.lower() or "issues" in tweet.lower():
    #     features.append(100)
    # else:
    #     features.append(0)

    # # Rule 4: Pregnancy not reached term (Negative)
    # if is_tweet_by_someone_else(tweet):
    #     features.append(100)
    #     print("It's working for 'someone_else'")
    #     print(tweet)
    # else:
    #     features.append(0)

    if re.search(r"^(?!.*\bour\s)(?!.*\bmy\s+(?:son|daughter)\b)(?=.*(?:my\s+god\s+(?:daughter|son|child)|brother|frankie)).*$", tweet.lower()):
        #  if re.search(r'\bI\'m\s+\d+\s+weeks?\b', tweet):
        features.append(0)  # Assign a positive weight
        print("It's working for 'my god daughter/god son/baby brother/frankie' and I am 38 week")
        print(tweet)
        #   else:
        # features.append(100)  # Assign a positive weight
        # print("It's working for 'my god daughter/god son/baby brother/frankie'")
        # print(tweet)
    else:
        features.append(0)
    # Rule 6: Adverse pregnancy outcome (Negative)
    if re.search(r'^(?!.*\b(?:I|my)\b\s+due\b.*\b(?:proud\s?aunt(?:y|ie)?|uncle)\b)(?=.*\b(?:proud\s?aunt(?:y|ie)?|uncle)\b|\b(?:my\s+)?(?:niece|nephew)\b)', tweet.lower()):
        # if re.search(r'\bI\s+due\b', tweet):
        #    features.append(0)
           print("It's working for 'ProudAuntie and Due'")
           print(tweet)
           features.append(100)
        # else:
        #    features.append(100)
        #    print("It's working for 'ProudAuntie'")
        #    print(tweet)

    else:
        features.append(0)

    # # Rule 7: Refers to someone else's pregnancy (Negative)
    if re.search(r'^(?!.*(?:\b(?:\d{2} weeks\b|due\b)))(?=.*\b(?:congrats|congratulations)\b).*$', tweet, re.IGNORECASE):
        # Check if the tweet contains a name after "congrats" or "congratulations"
        # if re.search(r'congratulations?\s+\b\w+[- ]?\w*\b', tweet, re.IGNORECASE):
            features.append(100)  # Assign a negative weight
            print("It's working for 'Congrats' with name")
            print(tweet)
        # else:
        #     if re.search(r'\b(?:[1-9]\d*|1\d) weeks\b|due\b', tweet, re.IGNORECASE):
        #       features.append(0)  # Default negative weight
        #     #   print("It's working for 'Congrats' and 38 weeks")
        #     #   print(tweet)
        #     else:
        #       features.append(100)  # Default negative weight
        #       print("It's working for 'Congrats'")
        #       print(features)
        #       print(tweet)
    else:
        features.append(0)

    return features

# Extract rule-based features for training data
rule_based_features_train = [extract_rule_based_features(tweet) for tweet in tweets]
#print(rule_based_features_train)
rule_based_features_train = np.array(rule_based_features_train)

# Extract rule-based features for testing data
rule_based_features_test = [extract_rule_based_features(tweet) for tweet in testTweets]
rule_based_features_test = np.array(rule_based_features_test)

# Reshape X_train to have a single column
# Concatenate rule-based features with existing features for train data
X_train_with_rules = np.hstack((X_train.reshape(-1, 1), rule_based_features_train))

# Concatenate rule-based features with existing features for test data
X_test_with_rules = np.hstack((X_test.reshape(-1, 1), rule_based_features_test))


# # # Train model with rule-based features
model = BertClassifier(bert_config_json=None, bert_model='bert-base-uncased',
               bert_vocab=None, do_lower_case=None, epochs=3, eval_batch_size=8,
               fp16=False, from_tf=False, gradient_accumulation_steps=1,
               ignore_label=None, label_list=None, learning_rate=2e-05,
               local_rank=-1, logfile='bert_sklearn.log', loss_scale=0,
               max_seq_length=64, num_mlp_hiddens=500, num_mlp_layers=0,
               random_state=42, restore_file=None, train_batch_size=16,
               use_cuda=True, validation_fraction=0.1, warmup_proportion=0.1)

model = model.fit(X_train_with_rules, Y_train)

# Extract feature importances (for rule-based features, assess their impact)
# For BERT models, we might not have direct access to feature importances,
# so we'll analyze the impact of rule-based features
# You can also use techniques like permutation importance for other models
# For rule-based features, compare model predictions with and without the features
# and assess how much they influence predictions
predictions_with_rules = model.predict(X_test_with_rules)
predictions_without_rules = model.predict(X_test)

# Analyze the difference in predictions due to rule-based features
rule_based_feature_impact = np.mean(predictions_with_rules != predictions_without_rules, axis=0)

# Print or visualize feature importances
print("Impact of Rule-Based Features on Predictions:")
print(rule_based_feature_impact)

# Test model with rule-based features
accy = model.score(X_test_with_rules, y_test)

# make class probability predictions
y_prob = model.predict_proba(X_test)
#print("class prob estimates:\n", y_prob)

# make predictions
y_pred = model.predict(X_test)
print(y_pred)
print("Accuracy: %0.2f%%"%(metrics.accuracy_score(y_pred,  y_test) * 100))

target_names = ['negative', 'positive']
print(classification_report(y_test, y_pred, target_names=target_names))

evaluation_report = classification_report(y_test, y_pred, output_dict=True)

df_training_data = pd.DataFrame({'Tweets': tweets, 'Label': label})
df_tweets = pd.DataFrame({'Tweets': testTweets, 'Predicted_Label': y_pred, 'Actual_Label': y_test})
df_evaluation_report = pd.DataFrame(evaluation_report).transpose()
# name = fileName.split("/")
FilePath = "/home/mbiswas2/ondemand/test-site-venv-3.9.9-no-sys-pack-compute-node/research/Result/"+"TrainingSet_GPT_3_6_7_1_2_5_4_Augmentedn"+".xlsx"
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
