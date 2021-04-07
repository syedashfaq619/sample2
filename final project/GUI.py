from tkinter import *

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



