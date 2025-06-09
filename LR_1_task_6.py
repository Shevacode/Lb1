import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Завантаження даних
try:
    data = pd.read_csv("data_multivar_nb.txt", delimiter="\t")
    print("Перші рядки з файлу:")
    print(data.head())
    print("Форма даних:", data.shape)
except Exception as e:
    print("Помилка при зчитуванні файлу:", e)
    exit()

# Переконатися, що дані мають правильний формат
if data.shape[1] < 2:
    print("Помилка: недостатньо стовпців у файлі.")
    exit()

X = data.iloc[:, :-1].values  # Ознаки
y = data.iloc[:, -1].values   # Мітки класів

# Перевірка розмірностей
print("Форма X:", X.shape)
print("Форма y:", y.shape)
if X.shape[1] == 0:
    print("Помилка: X не містить жодної ознаки.")
    exit()

# Розбиття даних на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Модель SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Модель наївного байєса
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Оцінка якості класифікації
print("Support Vector Machine Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))

# Порівняння моделей
if accuracy_score(y_test, y_pred_svm) > accuracy_score(y_test, y_pred_nb):
    print("SVM показує кращі результати класифікації.")
else:
    print("Наївний байєсівський класифікатор показує кращі або аналогічні результати.")