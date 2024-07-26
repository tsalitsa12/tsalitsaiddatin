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

# Judul aplikasi
st.title('Restaurant Menu Optimization')

# Membuat sidebar untuk navigasi
menu = st.sidebar.selectbox("Menu", ["Data Description", "Data Info and Statistics", "Stage 1", "Stage 2 and 3"])

# Fungsi untuk memuat dataset
@st.cache
def load_data():
    data = pd.read_csv("restaurant_menu_optimization_data.csv")
    return data

data = load_data()

# Menu pertama: Deskripsi Data
if menu == "Data Description":
    st.write("## Data Description")
    st.write("This section will contain information about the dataset, objectives of the analysis, and other descriptive text.")
    # Isi deskripsi sesuai keinginan Anda di sini.
    st.write("### Overview")
    st.write("This application is designed to analyze and optimize restaurant menu items. The dataset includes various features such as category, ingredients, price, and profitability.")

# Menu kedua: Data Info dan Statistik
elif menu == "Data Info and Statistics":
    st.write("## Data Info and Statistics")
    st.write("This section provides basic information about the dataset, such as data types and summary statistics.")
    st.write("### Dataset")
    st.write(data)
    st.write("### Data Info")

    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("### Descriptive Statistics")
    st.write(data.describe())

    st.write("### Distribusi Kategori Menu")
    fig, ax = plt.subplots()
    sns.countplot(x='MenuCategory', data=data, ax=ax)
    st.pyplot(fig)

    st.write("### Harga vs Profitabilitas")
    fig, ax = plt.subplots()
    sns.boxplot(x='Profitability', y='Price', data=data, ax=ax)
    st.pyplot(fig)

# Menu ketiga: Stage 1
elif menu == "Stage 1":
    st.write("## Stage 1")
    st.write("### Data setelah transformasi")
    label_encoder_menu = LabelEncoder()
    label_encoder_profit = LabelEncoder()
    data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
    data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])

    st.write(data)

    scaler = StandardScaler()
    data[['Price']] = scaler.fit_transform(data[['Price']])

    menu_category_mapping = dict(zip(label_encoder_menu.classes_, label_encoder_menu.transform(label_encoder_menu.classes_)))
    profitability_mapping = dict(zip(label_encoder_profit.classes_, label_encoder_profit.transform(label_encoder_profit.classes_)))

    st.write("### Mapping Kategori Menu")
    st.json(menu_category_mapping, expanded=True)

    st.write("### Mapping Profitabilitas")
    st.json(profitability_mapping, expanded=True)
