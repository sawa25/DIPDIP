import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, hstack
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import traceback
import pickle


# Функция для сохранения данных в pickle
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Функция для извлечения данных из pickle
def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
# Проверка наличия сохраненных данных
try:
    item_user_interactions = load_data('item_user_interactions.pkl')
    item_features = load_data('item_features.pkl')
    properties_filtered = load_data('properties_filtered.pkl')
    property_to_code = load_data('property_to_code.pkl')
except FileNotFoundError:
    # Если файлы не найдены, выполняем код для подготовки данных
    events = pd.read_csv("dfev.csv")
    properties = pd.concat([
        pd.read_csv("item_properties_part1.csv.zip"),
        pd.read_csv("item_properties_part2.csv.zip")
    ])
    categories = pd.read_csv("category_tree.csv")

    # Фильтрация событий, оставляя только те, которые были куплены
    purchased_events = events[events['event'] == 'transaction']
    top_properties = properties.drop_duplicates(['itemid', 'property']).groupby("property")['itemid'].count().sort_values(ascending=False)[:20]
    properties_filtered = properties[properties['property'].isin(set(top_properties.index))]

    # Отбрасываем строки из properties_filtered, где itemid не присутствует в purchased_events
    properties_filtered = properties_filtered[properties_filtered['itemid'].isin(purchased_events['itemid'].unique())]
    # Отбрасываем строки из purchased_events, где itemid не присутствует в filtered_properties
    purchased_events = purchased_events[purchased_events['itemid'].isin(properties_filtered['itemid'].unique())]
    
    unique_properties_set = set()
    properties_filtered['value'].str.split().apply(unique_properties_set.update)
    unique_properties_list = list(unique_properties_set)

    # пробуем уменьшить размерность
    # Вычисляем частоту каждого уникального свойства
    property_frequencies = properties_filtered['value'].str.split().explode().value_counts()
    frequencies = property_frequencies.reset_index(name='Frequency').rename(columns={'index': 'value'})
    # Определяем количество бинов для разделения данных
    num_bins = 100 
    # Разделяем данные на бины
    frequencies['Bin'] = pd.cut(frequencies['Frequency'], bins=num_bins, labels=False)

    filtered_values = frequencies[frequencies['Bin'] >= 1]['value'].unique()
    
    # Кодируем идентификаторы пользователей
    visitorid_encoder = LabelEncoder()
    purchased_events['visitorid_encoded']=visitorid_encoder.fit_transform(purchased_events['visitorid'])

    itemid_encoder = LabelEncoder()
    # Объединяем все уникальные идентификаторы товаров из обоих датасетов
    all_itemids = np.concatenate([purchased_events['itemid'].unique(), properties_filtered['itemid'].unique()])
    # Обучаем LabelEncoder на всех уникальных идентификаторах товаров
    itemid_encoder.fit(all_itemids)
    # Теперь преобразуем идентификаторы товаров в каждом датасете
    purchased_events['itemid_encoded'] = itemid_encoder.transform(purchased_events['itemid'])
    properties_filtered['itemid_encoded'] = itemid_encoder.transform(properties_filtered['itemid'])

    # Теперь создаем матрицу взаимодействий с использованием закодированных идентификаторов
    item_user_interactions = coo_matrix((np.ones(purchased_events.shape[0]), 
                                        (purchased_events['visitorid_encoded'], purchased_events['itemid_encoded'])))

    # Проверим размерность получившейся матрицы
    print("Размерность матрицы взаимодействий после кодирования:", item_user_interactions.shape)
    # Кодирование отфильтрованных уникальных значений свойств товаров
    property_encoder = LabelEncoder()
    property_encoded = property_encoder.fit_transform(filtered_values)
    property_to_code = dict(zip(filtered_values, property_encoded))
    # Сопоставление каждого товара с его закодированными свойствами
    def encode_properties(values):
        # Разделяем значения свойств и фильтруем их, оставляя только те, которые присутствуют в filtered_values
        return [property_to_code[val] for val in values.split() if val in property_to_code]

    # Создаем пустые списки для данных матрицы признаков товаров
    rows, cols, data = [], [], []
    properties_filtered['encoded_values'] = properties_filtered['value'].apply(encode_properties)
    for _, row in properties_filtered.iterrows():
        for val in row['encoded_values']:
            rows.append(row['itemid_encoded'])
            cols.append(val)
            data.append(1)

    print(len(rows))
    print(len(cols))
    print(len(data))

    item_features = coo_matrix((data, (rows, cols)), 
                            shape=(item_user_interactions.shape[1], len(filtered_values)))
    # Проверка размерности матрицы признаков
    print("Размерность матрицы признаков:", item_features.shape)

    # Сохранение подготовленных данных для последующего использования
    save_data(item_user_interactions, 'item_user_interactions.pkl')
    save_data(item_features, 'item_features.pkl')
    save_data(properties_filtered, 'properties_filtered.pkl')
    save_data(property_to_code, 'property_to_code.pkl')

# Проверка наличия пропущенных значений в матрице взаимодействий
print("Пропущенные значения в item_user_interactions:", np.any(np.isnan(item_user_interactions.toarray())))
# Проверка наличия пропущенных значений в матрице признаков
print("Пропущенные значения в item_features:", np.any(np.isnan(item_features.toarray())))
# Проверка размерности матрицы взаимодействий
print("Размерность матрицы взаимодействий:", item_user_interactions.shape)
# Проверка размерности матрицы признаков
print("Размерность матрицы признаков:", item_features.shape)


# Создаем модель https://github.com/lyst/lightfm/issues/690 without the loss='warp' parameter (using default logistic loss).
model = LightFM()
# model = LightFM(loss='bpr', item_alpha=1e-6, user_alpha=1e-6)loss='warp'

try:
# Обучение модели факторизационных машин с признаками товаров
    model.fit(item_user_interactions, item_features=item_features, epochs=1, num_threads=1,verbose=True)
except Exception as e:
    print(f"Ошибка при обучении модели: {e}")
    traceback.print_exc() # Выводит полный стек вызовов

# Оценка модели
test_precision = precision_at_k(model, item_user_interactions, item_features=item_features, k=3).mean()

print(f'Test Precision@3: {test_precision}')