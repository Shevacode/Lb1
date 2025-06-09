import numpy as np
from sklearn.preprocessing import Binarizer, StandardScaler, MinMaxScaler, Normalizer

# Варіант 18
input_data = np.array([4.6, 3.9, -3.5, -2.9, 4.1, 3.3, 2.2, 8.8, -6.1, 3.9, 1.4, 2.2, 2.2]).reshape(1, -1)

# Бінарізація
binarizer = Binarizer(threshold=0.0)
binary_data = binarizer.transform(input_data)
print("Бінарізація:\n", binary_data)

# Виключення середнього
scaler = StandardScaler()
standardized_data = scaler.fit_transform(input_data)
print("Виключення середнього:\n", standardized_data)

# Масштабування
minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(input_data)
print("Масштабування:\n", scaled_data)

# Нормалізація
normalizer = Normalizer(norm='l2')
normalized_data = normalizer.fit_transform(input_data)
print("Нормалізація:\n", normalized_data)
