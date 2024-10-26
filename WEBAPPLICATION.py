# import the important packages

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings("ignore")

################## main ##################### 

st.subheader("Model APP Using streamlit")
st.text("Only for Preprocessed Data")

uploaded_file = st.file_uploader("Choose a file")

################# sidebar #####################

st.sidebar.header("Classfication Model")
st_checkbox = st.sidebar.checkbox("Display Data", value = False)
st.sidebar.subheader("Choose classifier")
st_option = st.sidebar.selectbox("Classifier", ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'KNN'])


###############            Metrices             ######################
dict1 = {}
def metrices(y_test, y_prediction, st_option, fpr, tpr):
    ac = accuracy_score(y_test,y_prediction)
    ps = precision_score(y_test, y_prediction)
    rs = recall_score(y_test, y_prediction)
    fs = f1_score(y_test, y_prediction)
    dict1[st_option] = [ac, ps, rs, fs]
    new_df = pd.DataFrame(dict1, index = ['Accuracy_score','Precision_Score','Recall_Score','F1_score'])
    st.write(new_df)

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot confusion matrix
    cmt = confusion_matrix(y_test, y_prediction)
    ConfusionMatrixDisplay(cmt).plot(ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].grid(False)

    # Plot ROC curve
    axes[1].plot(fpr, tpr)
    axes[1].set_title("ROC Curve")

    # Display the plot
    st.pyplot(fig)

################          file uploaded            #################

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    selected_columns = st.sidebar.multiselect("Select Target Column", df.columns.tolist())
    
    if selected_columns:
        X = df.drop(selected_columns, axis = 1)
        y = df[selected_columns]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)
        if st_option == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            LR = LogisticRegression()
            LR.fit(X_train, y_train)    
            y_pred = LR.predict(X_test)
            y_lr_prob = LR.predict_proba(X_test)[:,1]
            fpr,tpr,threshold = roc_curve(y_test,y_lr_prob)
            metrices(y_test,y_pred, st_option, fpr, tpr)
        elif st_option == 'Decision Tree':
            from sklearn.tree import DecisionTreeClassifier
            dtree = DecisionTreeClassifier()
            dtree.fit(X_train, y_train)
            y_pred = dtree.predict(X_test)
            y_dt_prob = dtree.predict_proba(X_test)[:,1]
            fpr,tpr,threshold = roc_curve(y_test,y_dt_prob)
            metrices(y_test, y_pred, st_option, fpr, tpr)
        elif st_option == 'Naive Bayes':
            from sklearn.naive_bayes import GaussianNB
            NBtree = GaussianNB()
            NBtree.fit(X_train,y_train)
            y_pred = NBtree.predict(X_test)
            y_nb_prob = NBtree.predict_proba(X_test)[:,1]
            fpr,tpr,threshold = roc_curve(y_test,y_nb_prob)
            metrices(y_test, y_pred, st_option, fpr, tpr)
        elif st_option == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            KNNtree = KNeighborsClassifier()
            KNNtree.fit(X_train, y_train)
            y_pred = KNNtree.predict(X_test)
            y_kn_prob = KNNtree.predict_proba(X_test)[:,1]
            fpr,tpr,threshold = roc_curve(y_test,y_kn_prob)
            metrices(y_test, y_pred, st_option, fpr, tpr)


          


################ display data function ##################

if st_checkbox == True and uploaded_file is not None:
    st.write(df.head())







