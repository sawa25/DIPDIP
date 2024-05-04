import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, hstack
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import traceback
import pickle


# events — датасет с событиями. Колонки:
# timestamp — время события
# visitorid — идентификатор пользователя
# event — тип события ['view', 'addtocart', 'transaction']
# itemid — идентификатор объекта
# transactionid — идентификатор транзакции, если она проходила
events = pd.read_csv("dfev.csv")#датасет с очищенными некорректными рядами
# item_properties — файл с свойствами товаров.
# timestamp — момент записи значения свойства
# item_id — идентификатор объекта
# property — свойство, кажется, они все, кроме категории, захешированы
# value — значение свойства
properties = pd.concat([
    pd.read_csv("item_properties_part1.csv.zip"),
    pd.read_csv("item_properties_part2.csv.zip")
])
# category_tree — файл с деревом категорий (можно восстановить дерево).
# category_id — идентификатор категорий
# parent_id — идентификатор родительской категории
categories = pd.read_csv("category_tree.csv")
# Добавление новых признаков
events['event_datetime'] = pd.to_datetime(events['timestamp'], unit='ms')
properties['event_datetime'] = pd.to_datetime(properties['timestamp'], unit='ms')
events['day_of_week'] = events['event_datetime'].map(lambda x: x.weekday())
events['Year'] = events['event_datetime'].map(lambda x: x.year)
events['Month'] = events['event_datetime'].map(lambda x: x.month)
events['Day'] = events['event_datetime'].map(lambda x: x.day)
events['Hour'] = events['event_datetime'].map(lambda x: x.hour)
events['minute'] = events['event_datetime'].map(lambda x: x.minute)

def get_time_periods(hour):
    if hour >= 3 and hour < 7:
        return 'Dawn'
    elif hour >= 7 and hour < 12:
        return 'Morning'
    elif hour >= 12 and hour < 16:
        return 'Afternoon'
    elif hour >= 16 and hour < 22:
        return 'Evening'
    else:
        return 'Night'

events['Day Period'] = events['Hour'].map(get_time_periods)

# Фильтрация событий, оставляя только те, которые были куплены
purchased_events = events[events['event'] == 'transaction']
top_properties = properties.drop_duplicates(['itemid', 'property']).groupby("property")['itemid'].count().sort_values(ascending=False)[:20]
properties_filtered = properties[properties['property'].isin(set(top_properties.index))]

# Отбрасываем строки из properties_filtered, где itemid не присутствует в purchased_events
properties_filtered = properties_filtered[properties_filtered['itemid'].isin(purchased_events['itemid'].unique())]
# Отбрасываем строки из purchased_events, где itemid не присутствует в filtered_properties
purchased_events = purchased_events[purchased_events['itemid'].isin(properties_filtered['itemid'].unique())]

filtered_values = set()
properties_filtered['value'].str.split().apply(filtered_values.update)
filtered_values = list(filtered_values)

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

item_features = coo_matrix((data, (rows, cols)), 
                        shape=(item_user_interactions.shape[1], len(filtered_values)))
# Проверка размерности матрицы признаков
print("Размерность матрицы признаков:", item_features.shape)

# Создаем модель https://github.com/lyst/lightfm/issues/690 without the loss='warp' parameter (using default logistic loss).
model = LightFM()
# model = LightFM(loss='bpr', item_alpha=1e-6, user_alpha=1e-6)loss='warp'

# Обучение модели факторизационных машин с признаками товаров
model.fit(item_user_interactions, item_features=item_features, epochs=30, num_threads=1,verbose=True)

# Оценка модели
test_precision = precision_at_k(model, item_user_interactions, item_features=item_features, k=3).mean()

print(f'Test Precision@3: {test_precision}')
