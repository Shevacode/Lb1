import numpy as np
from sklearn import preprocessing

# Визначення вибірки даних
input_data = np.array([
    [5.1, -2.9, 3.3],
    [-1.2, 7.8, -6.1],
    [3.9, 0.4, 2.1],
    [7.3, -9.9, -4.5]
])

# 2.1.1 Бінарізація
data_birbarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nData Binarized: \n", data_birbarized)

# 2.1.2 Виключення середбнього значення:
print("\nBEFORE:")
print("Mean = ", input_data.mean(axis=0))
print("Std deviation = ", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean = ", data_scaled.mean(axis=0))
print("Std deviation = ", data_scaled.std(axis=0))

# 2.1.3 Масштабування
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nAfter Min-Max scaling:\n", data_scaled_minmax)

# 2.1.4 Нормалізація
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nAfter normalized L1:\n", data_normalized_l1)
print("\nAfter normalized L2:\n", data_normalized_l2)

# 2.1.5 Кодування міток
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
print("\nLabel Mapping: ")
for i, item in enumerate(encoder.classes_):
    print(item, '--->', i)

test_labels = ['green', 'red', 'black']
encode_values = encoder.transform(test_labels)
print("\nLabels = ", test_labels)
print("\nEncoded Labels = ", list(encode_values) )

# Декодування
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded Labels = ", encoded_values)
print("Decoded Labels = ", list(decoded_list))