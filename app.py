import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Encoding variabel kategorikal
    label_encoder_menu = LabelEncoder()
    label_encoder_profit = LabelEncoder()
    data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
    data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])

    # Penskalaan fitur numerik
    scaler = StandardScaler()
    data[['Price']] = scaler.fit_transform(data[['Price']])

    return data

def train_and_evaluate_models(data):
    # Memisahkan fitur dan target
    X = data[['Price']]
    y = data['Profitability']

    # Memisahkan data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC()
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        results[model_name] = {
            'Accuracy': accuracy_score(y_test, y_preds),
            'Precision': precision_score(y_test, y_preds, average='weighted'),
            'Recall': recall_score(y_test, y_preds, average='weighted'),
            'F1 Score': f1_score(y_test, y_preds, average='weighted')
        }

    return results

def main():
    file_path = "C:/MPML/restaurant_menu_optimization_data.csv"
    data = load_data(file_path)
    data = preprocess_data(data)
    results = train_and_evaluate_models(data)
    
    for model_name, metrics in results.items():
        print(f'{model_name} Results:')
        for metric_name, metric_value in metrics.items():
            print(f'  {metric_name}: {metric_value:.4f}')
        print()

if __name__ == "__main__":
    main()