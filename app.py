import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import json

# Fungsi untuk memuat data
@st.cache
def load_data():
    data = pd.read_csv("restaurant_menu_optimization_data.csv")
    return data

# Fungsi untuk menampilkan informasi data
def display_data_info(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# Fungsi untuk menampilkan statistik deskriptif
def display_data_statistics(data):
    st.write(data.describe())

# Fungsi untuk menampilkan visualisasi distribusi kategori menu
def display_category_distribution(data):
    fig, ax = plt.subplots()
    sns.countplot(x='MenuCategory', data=data, ax=ax)
    st.pyplot(fig)

# Fungsi untuk menampilkan visualisasi hubungan antara harga dan profitabilitas
def display_price_vs_profitability(data):
    fig, ax = plt.subplots()
    sns.boxplot(x='Profitability', y='Price', data=data, ax=ax)
    st.pyplot(fig)

# Fungsi untuk pra-pemrosesan data
def preprocess_data(data):
    label_encoder_menu = LabelEncoder()
    label_encoder_profit = LabelEncoder()
    data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
    data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])
    scaler = StandardScaler()
    data[['Price']] = scaler.fit_transform(data[['Price']])
    return data, label_encoder_menu, label_encoder_profit

# Fungsi untuk menampilkan peta nilai
def display_mapping(label_encoder_menu, label_encoder_profit):
    menu_category_mapping = dict(zip(label_encoder_menu.classes_, label_encoder_menu.transform(label_encoder_menu.classes_)))
    profitability_mapping = dict(zip(label_encoder_profit.classes_, label_encoder_profit.transform(label_encoder_profit.classes_)))
    st.write("## Mapping Kategori Menu")
    st.json(menu_category_mapping, expanded=True)
    st.write("## Mapping Profitabilitas")
    st.json(profitability_mapping, expanded=True)

# Fungsi untuk membagi data menjadi fitur dan target
def split_data(data):
    X = data[['Price']]
    y = data['Profitability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Fungsi untuk evaluasi model
def evaluate_models(X_train, X_test, y_train, y_test):
    # Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, log_preds)
    log_prec = precision_score(y_test, log_preds, average='weighted')
    log_rec = recall_score(y_test, log_preds, average='weighted')
    log_f1 = f1_score(y_test, log_preds, average='weighted')

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_prec = precision_score(y_test, dt_preds, average='weighted')
    dt_rec = recall_score(y_test, dt_preds, average='weighted')
    dt_f1 = f1_score(y_test, dt_preds, average='weighted')

    # SVM
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_prec = precision_score(y_test, svm_preds, average='weighted')
    svm_rec = recall_score(y_test, svm_preds, average='weighted')
    svm_f1 = f1_score(y_test, svm_preds, average='weighted')

    # Menampilkan hasil
    model_performance = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'SVM'],
        'Accuracy': [log_acc, dt_acc, svm_acc],
        'Precision': [log_prec, dt_prec, svm_prec],
        'Recall': [log_rec, dt_rec, svm_rec],
        'F1 Score': [log_f1, dt_f1, svm_f1]
    })

    return model_performance

# Fungsi untuk menampilkan hasil evaluasi model
def display_model_performance(model_performance):
    st.write(model_performance)

    # Visualisasi hasil
    st.write("## Model Performance Comparison")
    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='Accuracy', data=model_performance, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='Precision', data=model_performance, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='Recall', data=model_performance, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='F1 Score', data=model_performance, ax=ax)
    st.pyplot(fig)

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    # Menambahkan CSS untuk latar belakang
    st.markdown(
        """
       <style>
        .reportview-container {
            background-color: #f0f0f0; /* Warna latar belakang utama */
        }
        .sidebar .sidebar-content {
            background-color: #ffffff; /* Warna latar belakang sidebar */
        }
        .main .block-container {
            background-color: #ffffff; /* Warna latar belakang konten utama */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Pilih Menu:", ["Deskripsi Data", "Stage 1", "Stage 2", "Stage 3"])

    data = load_data()

    if menu == "Deskripsi Data":
        st.title("Deskripsi Data")
        st.write("Silakan masukkan deskripsi data Anda di sini.")
        st.write(data)
        st.write("## Data Info")
        display_data_info(data)
        st.write("## Descriptive Statistics")
        display_data_statistics(data)

    elif menu == "Stage 1":
        st.title("Stage 1: Explorasi Data")
        st.write("## Distribusi Kategori Menu")
        display_category_distribution(data)
        st.write("## Harga vs Profitabilitas")
        display_price_vs_profitability(data)

    elif menu == "Stage 2":
        st.title("Stage 2: Pra-pemrosesan Data")
        data, label_encoder_menu, label_encoder_profit = preprocess_data(data)
        st.write("## Data setelah transformasi")
        st.write(data)
        display_mapping(label_encoder_menu, label_encoder_profit)

    elif menu == "Stage 3":
        st.title("Stage 3: Evaluasi Model")
        X_train, X_test, y_train, y_test = split_data(data)
        model_performance = evaluate_models(X_train, X_test, y_train, y_test)
        display_model_performance(model_performance)

# Menjalankan aplikasi utama
if __name__ == "__main__":
    main()
