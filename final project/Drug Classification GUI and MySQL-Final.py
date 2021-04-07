#GUI and MySQL
from tkinter import *
import tkinter.messagebox
import mysql.connector


def MAIN():
  R1=Tk()
  R1.geometry('700x500')
  R1.title('WELCOME-1')

  l=Label(R1, text="WELCOME TO DRUG CLASSIFICATION PORTAL", font=('algerain',14,'bold'), fg="orange")
  l.place(x=100, y=50)

  b1=Button(R1, text="Register",width=10,height=2,font=('algerain',14), bg="lightblue", fg="red", command=m1)
  b1.place(x=200, y=200)
  
  b2=Button(R1, text="Login",width=10,height=2, font=('algerain',14), bg="lightblue", fg="red", command=m3)
  b2.place(x=200, y=300)
  
  R1.mainloop()


def m1():
  def m2():
    username=e1.get()
    password=e2.get()
    email=e3.get()
    phoneno=e4.get()

    a=mysql.connector.connect(host='localhost', port=3307, user="root", passwd="root", database="drug")
    b=a.cursor()
    b.execute("INSERT INTO t1 VALUES(%s,%s,%s,%s)",(username,password,email,phoneno))
    a.commit()

    if e1.get()=="" or e2.get=="":
      tkinter.messagebox.showinfo("SORRY!, PLEASE COMPLETE THE REQUIRED INFORMATION")
    else:
      tkinter.messagebox.showinfo("WELCOME %s" %username, "Lets Login")
      m3()

    
  R2=Tk()
  R2.geometry('600x500')
  R2.title('Register and Login')

  l=Label(R2, text="Login & Register", font=('algerain',14,'bold'), fg="orange")
  l.place(x=200, y=50)

  l1=Label(R2, text="Username", font=('algerain',14), fg="black")
  l1.place(x=100, y=200)
  l2=Label(R2, text="Password", font=('algerain',14), fg="black")
  l2.place(x=100, y=250)
  l3=Label(R2, text="Email", font=('algerain',14), fg="black")
  l3.place(x=100, y=300)
  l4=Label(R2, text="Phoneno", font=('algerain',14), fg="black")
  l4.place(x=100, y=350)
  
  e1=Entry(R2, font=14)
  e1.place(x=200, y=205)
  e2=Entry(R2, font=14, show="**")
  e2.place(x=200, y=255)
  e3=Entry(R2, font=14)
  e3.place(x=200, y=305)
  e4=Entry(R2, font=14)
  e4.place(x=200, y=355)

  b1=Button(R2, text="Signup",width=8,height=1, font=('algerain',14), bg="lightblue", fg="red", command=m2)
  b1.place(x=250, y=400)
      
  R2.mainloop()


def m3():
    def m4():
        a=mysql.connector.connect(host='localhost', port=3307, user="root", passwd="root", database="drug")
        b=a.cursor()
        username=e1.get()
        password=e2.get()

        if (e1.get()=="" or e2.get()==""):
            tkinter.messagebox.showinfo("SORRY!, PLEASE COMPLETE THE REQUIRED INFORMATION")
        else:
            b.execute("SELECT * FROM t1 WHERE username=%s AND password=%s",(username,password))

            if b.fetchall():
                tkinter.messagebox.showinfo("WELCOME %s" % username, "Logged in successfully")
                m5()#from function def m5() Function call for Fraud Detection
                
            else:
                tkinter.messagebox .showinfo("Sorry", "Wrong Password")
            
        
    R3=Tk()
    R3.geometry('600x500')
    R3.title('Login')

    l=Label(R3, text="Login", font=('algerain',14,'bold'), fg="orange")
    l.place(x=200, y=50)

    l1=Label(R3, text="Username", font=('algerain',14), fg="black")
    l1.place(x=100, y=200)
    l2=Label(R3, text="Password", font=('algerain',14), fg="black")
    l2.place(x=100, y=250)
      
    e1=Entry(R3, font=14)
    e1.place(x=200, y=205)
    e2=Entry(R3, font=14, show="**")
    e2.place(x=200, y=255)

    b1=Button(R3, text="Login",width=8,height=1, font=('algerain',14), bg="lightblue", fg="red", command=m4)
    b1.place(x=250, y=400)

    R3.mainloop()


def m5():
  R1=Tk()
  R1.geometry('700x600')
  R1.title('WELCOME-2')

  l=Label(R1, text="Algorithm and UI Selection", font=('algerain',14,'bold'), fg="orange")
  l.place(x=150, y=50)

  b1=Button(R1, text="Algorithm Selection",width=18,height=2,font=('algerain',14), bg="lightblue", fg="red", command=algorithm_selection)
  b1.place(x=150, y=250)
  
  b2=Button(R1, text="UI",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=UI)
  b2.place(x=400, y=250)

  R1.mainloop()
  

def algorithm_selection():
  R1=Tk()
  R1.geometry('1200x700')
  R1.title('WELCOME-3')

  l=Label(R1, text="Algorithms Selection", font=('algerain',20,'bold'), fg="orange")
  l.place(x=350, y=50)

  b1=Button(R1, text="SVM",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=SVM)
  b1.place(x=200, y=250)

  b2=Button(R1, text="KNN",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=KNN)
  b2.place(x=400, y=250)

  b3=Button(R1, text="Naive Bayes",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=Naive_Bayes)
  b3.place(x=600, y=250)

  b4=Button(R1, text="Pie-Chart",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=pie_chart)
  b4.place(x=800, y=250)
  
  R1.mainloop()

#SVM algorithm
def SVM():
  print('\n\n--------SVM algorithm----------')
  import numpy as np   #multi-dimensional arrays and matrices
  import pandas as pd   #data manipulation and analysis
  import matplotlib.pyplot as plt  #graph plot
  import warnings  
  warnings.filterwarnings('ignore')

  #To read the dataset
  dataset = pd.read_csv('Reviews4.csv', encoding='latin1')

  #to take the reviews/text from dataset
  reviews = [ i for i in dataset['Comments']]    
  #print(tweet)

  #classification of positive and negative sentiments
  d1 = {'positive':1,'negative':0}

  #stores accuracy of all the models
  l=[] 

  #(Tokenization is the act of breaking up a sequence of strings into pieces such as words,keywords, phrases, symbols and other elements called tokens)
  #CountVectorizer converts a collection of text documents to a matrix of token counts
  from sklearn.feature_extraction.text import CountVectorizer  
  cv = CountVectorizer(max_features = 10)

  #initial fitting of parameters on the training set
  #toarray() can be used to populate a numpy array 
  X = cv.fit_transform(reviews).toarray()

  #creating rows and columns 
  y = dataset.iloc[:, 1].values

  #split data into training and testing set
  from sklearn.model_selection import train_test_split  
  X_train,X_test,Y_train,Y_test= train_test_split(X, y, train_size=0.7, random_state=42)

  #To print accuracy
  from sklearn.metrics import accuracy_score  
  def print_score(clf, X_train, Y_train, X_test, Y_test,train=True):
      if train:  
          print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_train,clf.predict(X_train))))
          #{0:.4f} used for printing 4 values after decimal point

  #SVM algorithm
  from sklearn.svm import SVC
  classifier = SVC(kernel = 'linear', random_state = 0)
  classifier.fit(X_train, Y_train)     

  #function call- def print_score
  #Scores for training data
  print_score(classifier,X_train,Y_train,X_test, Y_test,train=True)



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


#KNN algorithm    
def KNN():
  print('\n---------KNN algorithm------------')
  import numpy as np   #multi-dimensional arrays and matrices
  import pandas as pd   #data manipulation and analysis
  import matplotlib.pyplot as plt  #graph plot
  import warnings  
  warnings.filterwarnings('ignore')

  #To read the dataset
  dataset = pd.read_csv('Reviews4.csv', encoding='latin1')

  #to take the reviews/text from dataset
  tweet = [ i for i in dataset['Comments']]    
  #print(tweet)

  #classification of positive and negative sentiments
  d1 = {'positive':1,'negative':0}

  #stores accuracy of all the models
  l=[] 

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
  from sklearn.model_selection import train_test_split  
  X_train,X_test,Y_train,Y_test= train_test_split(X, y, train_size=0.7, random_state=42)

  #To print accuracy
  from sklearn.metrics import accuracy_score  
  def print_score(clf, X_train, Y_train, X_test, Y_test,train=True):
      if train:  
          print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_train,clf.predict(X_train))))
  #{0:.4f} used for printing 4 values after decimal point

  #KNN algorithm
  #KNeighborsClassifier is a Classifier implementing the k-nearest neighbors vote.
  from sklearn.neighbors import KNeighborsClassifier  
  knn= KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
  # fit the knn model with data
  knn.fit(X_train,Y_train)
      
  #Scores for training data
  print_score(knn,X_train,Y_train,X_test, Y_test,train=True)



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


#Naive Bayes algorithm  
def Naive_Bayes():
  print('\n\n---------Naive Bayes algorithm---------------')
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


#pie-chart
def pie_chart():

  import matplotlib.pyplot as plt

  labels = 'Traxemic acid', 'Lupron', 'Ulipristal'
  sizes = [263,35,2]
  colors = ['green', 'yellow','red']
  plt.pie(sizes,labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
  plt.axis('equal')
  plt.title('Pie-Chart of all 3 drugs')
  plt.show()


#UI
def UI():

  R1 = Tk()
  R1.title('DRUG CLASSIFICATION')
  R1.geometry('600x400')

  w2 = Label(R1, justify=LEFT, text="Drug Classification using Machine Learning", fg="RED")
  w2.config(font=("Elephant", 15))
  w2.place(x=80,y=30)

  oc = StringVar(R1)
  oc.set("----Select tablet----")


  def function2():
      def function1(x):
          if (x == "Traxemic acid"):
              R3 = Tk()
              R3.title('Traxemic acid')
              R3.geometry('500x400')

              w2 = Label(R3, justify=LEFT, text="Drug Classification using Machine Learning-Traxemic acid", fg="Green")
              w2.config(font=("Elephant", 10))
              w2.place(x=80,y=18)

              import pandas as pd 
              d1 = pd.read_csv('Reviews4.csv',encoding='latin1',index_col ="DrugName")
              first = d1["Sentiment"]

              a = first['Traxemic acid'].value_counts(normalize=True).mul(100).round(1).astype(str)+'%'
              a1 = a.to_string()
              L1=Label(R3, text=str(a1), font=('Times',12,'bold'),fg="orange")
              L1.place(x=280, y=200)

              L2=Label(R3,text="Positive   =",font=('Times',12,'bold'))
              L2.place(x=210, y=200)

              L3=Label(R3,text="Negative = ",font=('Times',12,'bold'))
              L3.place(x=210, y=220)

          elif (x=="Lupron"):
              R4 = Tk()
              R4.title('Lupron')
              R4.geometry('500x400')

              w2 = Label(R4, justify=LEFT, text="Drug Classification using Machine Learning-Lupron", fg="Green")
              w2.config(font=("Elephant", 10))
              w2.place(x=80,y=18)

              import pandas as pd            
              d1 = pd.read_csv('Reviews4.csv',encoding='latin1',index_col ="DrugName")
              first = d1["Sentiment"] 
              
              b = first['Lupron'].value_counts(normalize=True).mul(100).round(1).astype(str)+'%'
              b1 = b.to_string()
              L1=Label(R4, text=str(b1), font=('Times',12,'bold'),fg="orange")
              L1.place(x=280, y=200)

              L2=Label(R4,text="Positive   =",font=('Times',12,'bold'))
              L2.place(x=210, y=200)

              L3=Label(R4,text="Negative = ",font=('Times',12,'bold'))
              L3.place(x=210, y=220)
              
          else:
              R5 = Tk()
              R5.title('Ulipristal')
              R5.geometry('500x400')

              w2 = Label(R5, justify=LEFT, text="Drug Classification using Machine Learning-Ulipristal", fg="Green")
              w2.config(font=("Elephant", 10))
              w2.place(x=80,y=18)

              import pandas as pd
              d1 = pd.read_csv('Reviews4.csv',encoding='latin1',index_col ="DrugName")
              first = d1["Sentiment"] 
          
              c = first['Ulipristal'].value_counts(normalize=True).mul(100).round(1).astype(str)+'%'
              c1 = c.to_string()
              L1=Label(R5, text=str(c1), font=('Times',12,'bold'),fg="orange")
              L1.place(x=280, y=200)

              L2=Label(R5,text="Positive   =",font=('Times',12,'bold'))
              L2.place(x=210, y=200)

              L3=Label(R5,text="Negative = ",font=('Times',12,'bold'))
              L3.place(x=210, y=220)     

      OM = OptionMenu(R1, oc,"Traxemic acid","Lupron","Ulipristal", command=function1)
      OM.place(x=240, y=185)

  function2()

  R1.mainloop()
  
MAIN()

