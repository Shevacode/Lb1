import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'

try:
    data = np.loadtxt(input_file, delimiter=',')
    print("Данные успешно загружены. Форма:", data.shape)
except Exception as e:
    print("Ошибка загрузки данных:", e)
    exit()

X, y = data[:, :-1], data[:, -1]

print("Количество признаков:", X.shape[1])
if X.shape[1] < 2:
    print("Ошибка: Должно быть хотя бы 2 признака для визуализации!")
    exit()

X_vis = X[:, :2]

classifier = GaussianNB()
classifier.fit(X_vis, y)

y_pred = classifier.predict(X_vis)

accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

visualize_classifier(classifier, X_vis, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
X_train_vis, X_test_vis = X_train[:, :2], X_test[:, :2]

classifier_new = GaussianNB()
classifier_new.fit(X_train_vis, y_train)

y_test_pred = classifier_new.predict(X_test_vis)

accuracy_new = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy_new, 2), "%")

visualize_classifier(classifier_new, X_test_vis, y_test)

num_folds = 3
accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print("Cross-validation Accuracy:", round(100 * accuracy_values.mean(), 2), "%")

precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Cross-validation Precision:", round(100 * precision_values.mean(), 2), "%")

recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Cross-validation Recall:", round(100 * recall_values.mean(), 2), "%")

f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("Cross-validation F1 Score:", round(100 * f1_values.mean(), 2), "%")
