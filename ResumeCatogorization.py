import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


#IMPORT CLEAN DATASET 
#clean_df = pd.read_csv("clean_resume_data.csv")
clean_df = pd.read_csv("dataset/clean_resume_data.csv")
print(clean_df.head())
print (clean_df.shape)

#EXPLORATORY DATA ANALYSIS 
print(clean_df['Category'].value_counts())
plt.figure(figsize=(15,5))
sns.countplot(clean_df['Category'])
plt.show()

#-----pie chart count-----#
counts = clean_df['Category'].value_counts()
labels = clean_df['Category'].unique()
plt.figure(figsize=(15,10))
plt.pie(counts,labels=labels,autopct='%1.1f%%',shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
plt.show()

#BALANCE DATASET
from sklearn.utils import resample
import pandas as pd

# Define the maximum count among all categories
max_count = clean_df['Category'].value_counts().max()

# Resample each category to match the maximum count
balanced_data = []

for category in clean_df['Category'].unique():
    category_data = clean_df[clean_df['Category'] == category]
    
    if len(category_data) < max_count:
        # Oversample categories with fewer samples
        balanced_category_data = resample(
            category_data,
            replace=True,
            n_samples=max_count,
            random_state=42
        )
    else:
        # Undersample categories with more samples
        balanced_category_data = resample(
            category_data,
            replace=False,
            n_samples=max_count,
            random_state=42
        )

    balanced_data.append(balanced_category_data)

# Concatenate the balanced data
balanced_df = pd.concat(balanced_data)

#Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(balanced_df['Category'].value_counts())

#TRAIN-TEST SPLIT 
balanced_df.isnull().sum
balanced_df.dropna(inplace=True)

X = balanced_df['Feature']
Y = balanced_df['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)

#ENCODING (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#TRAIN RANDOM FOREST CLASSIFIER 
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_tfidf , Y_train)
#ACCURACY EVALUATION 
y_pred = rf_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test, y_pred)
print("ACCURACY : " , accuracy)
print(classification_report(Y_test,y_pred))


#PLOTTING CONFUSION MATRIX 
cm = confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(12, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_classifier.classes_,
            yticklabels=rf_classifier.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#PREDICTIVE SYSTEM 

#EXAMPLE USAGE 
#resume_file="---hr assi"


