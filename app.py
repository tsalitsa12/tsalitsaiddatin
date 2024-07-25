import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

st.title('Restaurant Menu Optimization')

# Memuat dataset
@st.cache
def load_data():
    data = pd.read_csv("restaurant_menu_optimization_data.csv")
    return data

data = load_data()
st.write("## Dataset")
data

    # Descriptive statistics
    st.subheader('Descriptive Statistics')
    st.write(data.describe())
    
    st.subheader('Data Info')
    buffer = []
    data.info(buf=buffer)
    s = buffer[0]
    st.text(s)

    # Visualizations
    st.subheader('Category Distribution')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='MenuCategory', data=data)
    plt.title('Distribusi Kategori Menu')
    st.pyplot(plt)

    st.subheader('Price vs Profitability')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Profitability', y='Price', data=data)
    plt.title('Harga vs Profitabilitas')
    st.pyplot(plt)

    # Stage 2: Data preprocessing
    st.header('Stage 2: Data Preprocessing')
    st.write(data.isnull().sum())
    
    label_encoder_menu = LabelEncoder()
    label_encoder_profit = LabelEncoder()
    data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
    data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])
    
    st.write(data.head())
    
    menu_category_mapping = dict(zip(label_encoder_menu.classes_, label_encoder_menu.transform(label_encoder_menu.classes_)))
    profitability_mapping = dict(zip(label_encoder_profit.classes_, label_encoder_profit.transform(label_encoder_profit.classes_)))
    
    st.write("MenuCategory Mapping:", menu_category_mapping)
    st.write("Profitability Mapping:", profitability_mapping)
    
    scaler = StandardScaler()
    data[['Price']] = scaler.fit_transform(data[['Price']])
    
    st.write(data.head())

    # Stage 3: Model Training and Evaluation
    st.header('Stage 3: Model Training and Evaluation')
    X = data[['Price']]
    y = data['Profitability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC()
    }

    for name, model in models.items():
        st.subheader(f'{name}')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        st.write(f'Accuracy: {accuracy_score(y_test, preds)}')
        st.write(f'Precision: {precision_score(y_test, preds, average="weighted")}')
        st.write(f'Recall: {recall_score(y_test, preds, average="weighted")}')
        st.write(f'F1 Score: {f1_score(y_test, preds, average="weighted")}')

    # Hyperparameter tuning with GridSearchCV
    st.header('Hyperparameter Tuning with GridSearchCV')
    
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly']
        }
    }
    
    for name, model in models.items():
        st.subheader(f'{name} Hyperparameter Tuning')
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        preds_best = best_model.predict(X_test)
        st.write(f'Best Params: {grid_search.best_params_}')
        st.write(f'Best Accuracy: {accuracy_score(y_test, preds_best)}')
        st.write(f'Precision: {precision_score(y_test, preds_best, average="weighted")}')
        st.write(f'Recall: {recall_score(y_test, preds_best, average="weighted")}')
        st.write(f'F1 Score: {f1_score(y_test, preds_best, average="weighted")}')
        
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
        st.write(f'Cross-validation Accuracy: {cv_scores.mean()}')

# requirements.txt
st.header('requirements.txt')
st.text('streamlit\npandas\nseaborn\nscikit-learn\nmatplotlib')
