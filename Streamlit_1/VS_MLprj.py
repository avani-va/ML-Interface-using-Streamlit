import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

st.title('Machine Learning Analysis on Datasets')
st.markdown('---')
st.subheader('Main tools used to analyse the datasets are:')
st.checkbox('Support Vector Machine',value = 'True')
st.checkbox('Random Forest Classifier', value = 'True')
st.markdown('---')
st.subheader('A brief explanation on:')
st.markdown('**👉   SUPPORT VECTOR MACHINE (SVM) :**')
st.text('⁜  Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It is most commonly used for classification, where it separates data points into different classes by finding the best possible boundary')
st.text('⁜  Here is a visual representation:')
st.image('SVM.jpg')
st.markdown('---')
st.markdown('**👉   RANDOM FOREST CLASSIFIER :**')
st.text('⁜  A Random Forest Classifier is a supervised machine learning algorithm used for classification problems.'
'It is an ensemble learning method that combines the predictions of multiple decision trees to produce a more accurate and robust result.')
st.text('⁜  Here is a visual representation fro you:')
st.image('RFC.jpg')
st.markdown('---')
st.markdown('---')
st.subheader('We mainly use scikit-learn library for Machine Learning')
st.markdown('[You can look at some of their datasets here](https://scikit-learn.org/stable/datasets.html#datasets)')
st.subheader('WE WILL LOOK INTO IT IN MORE DETAIL 😉')
st.markdown('---')

st.text('We will provide you with 2 datasets from scikit-learn library')
st.text('👉     Digits.data')
st.text('👉     Iris.data')
choice_1 = st.selectbox('Now Please select the model you want to work with first:',options = ('Digits.data','Iris.data'))
st.markdown('---')
st.text('We will provide you with 2 ML Tools to perform on these dtatasets')
st.text('👉     SVM')
st.text('👉     Random Forest Classifier')
choice_2 = st.selectbox('Now Please select the tool you want to work with this dataset:',options = ('svm','rfc'))
st.markdown('---')

    
def iris_with_rfc():
    iris = load_iris()

    df = pd.DataFrame(iris.data,columns = iris.feature_names)
    df['target'] = iris.target
    df['flower_type'] = df.target.apply(lambda x: iris.target_names[x])
    st.write('Iris Dataset contains:')
    st.dataframe(df.head())

    st.markdown('---')
    x = df.drop(columns = ['target','flower_type'])
    y = df['target']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    trn_len = len(x_train)
    tst_len = len(x_test)
    st.write('Length of the training data',trn_len)
    st.write('Length of the testing data',tst_len)
    
    model = RandomForestClassifier()
    model.fit(x_train,y_train)

    st.markdown('---')
    accuracy = model.score(x_test,y_test)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)

    st.markdown('---')
    fig1, ax1 = plt.subplots()

    sn.heatmap(cm,annot = True,cmap = 'Greens',ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual Values')
    ax1.set_title('Confusion Matrix')
    st.pyplot(fig1)

    st.markdown('---')

    fig, ax = plt.subplots()

    df0 = df[:50]
    df1 = df[50:100]
    df2 = df[100:]

    ax.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color = 'pink',marker = '^',label = 'setosa')
    ax.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color = 'cyan',marker = '*',label = 'versicolor')
    ax.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color = 'orange',marker = 'o',label = 'virginica')
    ax.legend()
    st.pyplot(fig)

def digits_with_rfc():
    digit = load_digits()

    df = pd.DataFrame(digit.data,columns = digit.feature_names)
    df['target'] = digit.target
    st.write('Digit Dataset contains:')
    st.dataframe(df.head())

    st.markdown('---')
    x = df.drop(columns = ['target'])
    y = df['target']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    trn_len = len(x_train)
    tst_len = len(x_test)
    st.write('Length of the training data',trn_len)
    st.write('Length of the testing data',tst_len)
    
    model = RandomForestClassifier()
    model.fit(x_train,y_train)

    st.markdown('---')
    accuracy = model.score(x_test,y_test)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    
    fig1, ax1 = plt.subplots()

    sn.heatmap(cm,annot = True,cmap = 'coolwarm',ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual Values')
    ax1.set_title('Confusion Matrix')
    st.pyplot(fig1)

def digits_with_svc():
    digit = load_digits()

    df = pd.DataFrame(digit.data,columns = digit.feature_names)
    df['target'] = digit.target
    st.write('Digit Dataset contains:')
    st.dataframe(df.head())

    st.markdown('---')
    x = df.drop(columns = ['target'])
    y = df['target']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    trn_len = len(x_train)
    tst_len = len(x_test)
    st.write('Length of the training data',trn_len)
    st.write('Length of the testing data',tst_len)

    st.markdown('---')
    model = SVC()
    model.fit(x_train,y_train)
    accuracy = model.score(x_test,y_test)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    
    fig1, ax1 = plt.subplots()

    sn.heatmap(cm,annot = True,cmap = 'viridis',ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual Values')
    ax1.set_title('Confusion Matrix')
    st.pyplot(fig1)

def iris_with_svc():
    iris = load_iris()

    df = pd.DataFrame(iris.data,columns = iris.feature_names)
    df['target'] = iris.target
    df['flower_type'] = df.target.apply(lambda x: iris.target_names[x])
    st.write('Iris Dataset contains:')
    st.dataframe(df.head())

    st.markdown('---')
    x = df.drop(columns = ['target','flower_type'])
    y = df['target']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    trn_len = len(x_train)
    tst_len = len(x_test)
    st.write('Length of the training data',trn_len)
    st.write('Length of the testing data',tst_len)

    st.markdown('---')
    model = SVC()
    model.fit(x_train,y_train)
    accuracy = model.score(x_test,y_test)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    
    fig1, ax1 = plt.subplots()

    sn.heatmap(cm,annot = True,cmap = 'Blues',ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual Values')
    ax1.set_title('Confusion Matrix')
    st.pyplot(fig1)

# Creating a dictionary
model = {('Digits.data','svm'): digits_with_svc, 
         ('Iris.data','svm'):iris_with_svc,
         ('Digits.data','rfc'):digits_with_rfc,
         ('Iris.data','rfc'):iris_with_rfc
        }

state = st.button('Click to Run Model')
if state:
    func = model[(choice_1,choice_2)]
    func()