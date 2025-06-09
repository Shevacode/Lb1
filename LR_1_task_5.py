import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

def yourname_confusion_matrix(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def find_TP(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def my_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def my_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

# Завантаження даних
try:
    df = pd.read_csv('register.csv', encoding='utf-8')
    df.columns = df.columns.str.strip()

    required_columns = {'actual_label', 'model_RF', 'model_LR'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Відсутні необхідні стовпці: {required_columns - set(df.columns)}")

    # Перетворення типів даних у float
    df['actual_label'] = pd.to_numeric(df['actual_label'], errors='coerce')
    df['model_RF'] = pd.to_numeric(df['model_RF'], errors='coerce')
    df['model_LR'] = pd.to_numeric(df['model_LR'], errors='coerce')

    # Перевірка на наявність NaN після конвертації
    if df[['actual_label', 'model_RF', 'model_LR']].isna().sum().sum() > 0:
        raise ValueError("Є некоректні значення (NaN) після перетворення типів")

    # Перетворення прогнозованих значень
    thresh = 0.5
    df['predicted_RF'] = (df['model_RF'] >= thresh).astype(int)
    df['predicted_LR'] = (df['model_LR'] >= thresh).astype(int)

    # Виведення матриці помилок перед assert для перевірки
    print("Confusion Matrix RF:\n", yourname_confusion_matrix(df['actual_label'].values, df['predicted_RF'].values))
    print("Confusion Matrix LR:\n", yourname_confusion_matrix(df['actual_label'].values, df['predicted_LR'].values))

    # Перевірка правильності реалізації
    assert np.array_equal(yourname_confusion_matrix(df['actual_label'].values, df['predicted_RF'].values),
                          confusion_matrix(df['actual_label'].values, df['predicted_RF'].values)), 'Помилка у матриці помилок для RF'
    assert np.array_equal(yourname_confusion_matrix(df['actual_label'].values, df['predicted_LR'].values),
                          confusion_matrix(df['actual_label'].values, df['predicted_LR'].values)), 'Помилка у матриці помилок для LR'
    assert my_accuracy_score(df['actual_label'].values, df['predicted_RF'].values) == accuracy_score(
        df['actual_label'].values, df['predicted_RF'].values), 'Помилка у точності для RF'
    assert my_accuracy_score(df['actual_label'].values, df['predicted_LR'].values) == accuracy_score(
        df['actual_label'].values, df['predicted_LR'].values), 'Помилка у точності для LR'
    assert my_recall_score(df['actual_label'].values, df['predicted_RF'].values) == recall_score(
        df['actual_label'].values, df['predicted_RF'].values), 'Помилка у recall для RF'
    assert my_recall_score(df['actual_label'].values, df['predicted_LR'].values) == recall_score(
        df['actual_label'].values, df['predicted_LR'].values), 'Помилка у recall для LR'

    # Вивід результатів
    print(f'Accuracy RF: {my_accuracy_score(df.actual_label.values, df.predicted_RF.values):.3f}')
    print(f'Accuracy LR: {my_accuracy_score(df.actual_label.values, df.predicted_LR.values):.3f}')
    print(f'Recall RF: {my_recall_score(df.actual_label.values, df.predicted_RF.values):.3f}')
    print(f'Recall LR: {my_recall_score(df.actual_label.values, df.predicted_LR.values):.3f}')
except Exception as e:
    print(f'Помилка: {e}')
