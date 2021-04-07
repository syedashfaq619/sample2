import numpy as np   #multi-dimensional arrays and matrices
import pandas as pd   #data manipulation and analysis
import matplotlib.pyplot as plt  #graph plot
from sklearn import svm  #
import warnings  
warnings.filterwarnings('ignore')

#To read the dataset
dataset = pd.read_csv('Reviews4.csv', encoding='latin1')

#to take the reviews/text from dataset
tweet = [ i for i in dataset['Comments']]    
#print(tweet)

d1 = {'positive':1,'neutral':0,'negative':-1}

#stores accuracy of all the models
l=[] 

#CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words
#(Tokenization is the act of breaking up a sequence of strings into pieces such as words,keywords, phrases, symbols and other elements called tokens)
#CountVectorizer converts a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer  
cv = CountVectorizer(max_features = 10)

#initial fitting of parameters on the training set
#toarray() can be used to populate a numpy array 
X = cv.fit_transform(tweet).toarray()

#creating rows and columns 
y = dataset.iloc[:, 1].values

#split data into training and testing set
from sklearn.model_selection import train_test_split   #split data into training and testing set
X_train,X_test,Y_train,Y_test= train_test_split(X, y, train_size=0.7, random_state=42)

#Defining a funtion print_score
#To print accuracy
from sklearn.metrics import accuracy_score  
def print_score(clf, X_train, Y_train, X_test, Y_test,train=True):   
    if train:  
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_train,clf.predict(X_train))))
        #{0:.4f} used for printing 4 values after decimal point

#Naive-Bayes algorithm
#Gaussian Naive Bayes (GaussianNB)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#fit the Gaussian model with data
model.fit(X_train,Y_train)

print("\n\n\n-------Naive Bayes algorithm----------\n")

#Scores for training data
#Function call 
print_score(model,X_train,Y_train,X_test, Y_test,train=True)   



#Total count of all sentiments

print('\n\n\n---------Total count of all sentiments------')

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Reviews4.csv',encoding='latin1')
      
df = pd.DataFrame(data, columns= ['Sentiment'])

print('\nTotal count of all sentiments:\n\n')

print(df['Sentiment'].value_counts())


print('\n\n--------Printing the 1s present in all three drugs-----\n')

import pandas as pd

d = pd.read_csv('Reviews4.csv',encoding='latin1')
df1= d[d['Sentiment'] != 0]
print(df1)

columns=['Sentiment']
for i in d:
    if columns==['Sentiment']:
        d = pd.DataFrame(df1, columns= ['Sentiment'])
print(df1['DrugName'].value_counts())


print('\n-------Printing the Topmost drug-----------\n')


#create a dictionary and save in csv file

top = {"DrugName":"Sentiment","Traxemic acid":263 , "Lupron":35, 'Ulipristal':2}

with open('top.csv', 'w') as f:
    for key in top.keys():
        f.write("%s,%s\n"%(key,top[key]))

#to read top college(first row)
import pandas as pd
data = pd.read_csv("top.csv", nrows=1)
print("The best drug is:\n\n", data)


print('-----Plotting the Pie chart----------')

import matplotlib.pyplot as plt

labels = 'Traxemic acid', 'Lupron', 'Ulipristal'
sizes = [263,35,2]
colors = ['green', 'yellow','red']
plt.pie(sizes,labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Pie-Chart of all 3 drugs')
plt.show()

