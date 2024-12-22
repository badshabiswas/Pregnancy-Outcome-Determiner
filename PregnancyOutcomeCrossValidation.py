# Import Section
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score



from tqdm import tqdm

from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import BertTokenClassifier
from bert_sklearn import load_model




# Define the classifier pipeline
classifier = BertClassifier(bert_config_json=None, bert_model='bert-base-uncased',
               bert_vocab=None, do_lower_case=None, epochs=3, eval_batch_size=8,
               fp16=False, from_tf=False, gradient_accumulation_steps=1,
               ignore_label=None, label_list=None, learning_rate=2e-05,
               local_rank=-1, logfile='bert_sklearn.log', loss_scale=0,
               max_seq_length=64, num_mlp_hiddens=500, num_mlp_layers=0,
               random_state=42, restore_file=None, train_batch_size=16,
               use_cuda=True, validation_fraction=0.1, warmup_proportion=0.1)
# classifier = Pipeline([
#       ('count_vectorizer', CountVectorizer(ngram_range=(1, 3))),
#       ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)),
#       ('clf', OneVsRestClassifier(LinearSVC(C=10.0, class_weight=None, dual=True, fit_intercept=True,
#         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#         multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#         verbose=0)))])



# Load your data
DataFiles = [
    "/home/mbiswas2/ondemand/test-site-venv-3.9.9-no-sys-pack-compute-node/research/Data Set/PreprocessedTweets_Mixed_Class.txt"
    # Add other file paths as needed
]

def main():
    counterP = 0
    counterN = 0
    for fileName in DataFiles:
        tweets = []
        labels = []
        with open(fileName, 'r') as f:
            next(f)  # Skip headings
            for line in f:
                lineVal = line.split("\t")

                tweets.append(lineVal[0])

                if(lineVal[1].strip()== '1'):
                  labels.append("Positive")
                  counterP = counterP+1
                else:
                  labels.append("Negative")
                  counterN = counterN+1

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state=42)

        # Train the classifier
        classifier.fit(X_train, y_train)
        

        # Test the classifier
        y_pred = classifier.predict(X_test)

        # Print evaluation metrics
        print("Evaluation Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("The value of Positive in Training Set "+ str(counterP))
        print("The value of Negative in Training Set "+ str(counterN))
        scores = cross_val_score(classifier, tweets, labels, cv=12, scoring='f1_macro')
        print(scores)
        # scores = cross_val_score(classifier, tweets, labels, cv=12, scoring='f1_micro')
        # print(scores)
        # scores = cross_val_score(classifier, tweets, labels, cv=12, scoring='f1')
        # print(scores)
        scores = cross_val_score(classifier, tweets, labels, cv=12, scoring='f1_weighted')
        print(scores)

if __name__ == '__main__':
    main()
