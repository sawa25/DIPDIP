import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import precision_score
import numpy as np
import os
import sys

class RecomendModel():
    # загрузка адаптированной к обучению базы и обучение
    def __init__(self, websocket=None, merged_data_fname="merged_data.csv.zip"):
        self.websocket = websocket
        # папка, относительно которой искать папку photos для сохранения картинки
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        # self.merged_data_fname = f"{script_dir}/fast_api/inputdataset/{merged_data_fname}"
        self.merged_data_fname = f"../inputdataset/{merged_data_fname}"
        self.initialized = False        
    async def initialize(self):
        if self.websocket:
            await self.websocket.send_text(f"{os.path.abspath(self.merged_data_fname)}")
        else:
            print("websocket fail, Starting initialization...")

        # предподготовленный датасет с учетом очистки,генерации факторов и т.д.
        # merged_data=pd.read_csv(f"{script_dir}/../inputdataset/{merged_data}")
        print(f"Загрузка датасета {self.merged_data_fname} ~1мин 25сек....")
        # await asyncio.sleep(5)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            if self.websocket:
                await self.websocket.send_text(f"Загрузка датасета {self.merged_data_fname} ~1мин 25сек.")
            await loop.run_in_executor(pool, self.load_)
            if self.websocket:
                await self.websocket.send_text("Обучение модели ~40сек....")
            await loop.run_in_executor(pool, self.train_)
        
        self.initialized = True
        if self.websocket:
            await self.websocket.send_text("finish")

    def load_(self):
        merged_data=pd.read_csv(f"{self.merged_data_fname}")

        # создаем новый столбец 'purchased_binary', который будет равен 1, если событие было 'transaction', и 0 в противном случае
        merged_data['purchased_binary'] = (merged_data['event'] == 'transaction').astype(int)

        # столбцы для возможности облегчения обучающего датасета и 
        # дальнейшего сопоставления индексов для поиска товаров
        self.item_ids = merged_data[['itemid', 'datetime']].copy()
        # последнее облегчение датасета
        self.purchased_final = merged_data.drop(['Day Period','timestamp','event','itemid','transactionid','datetime','event_datetime'], axis=1)

        X = self.purchased_final.drop(['purchased_binary'], axis=1)
        y = self.purchased_final['purchased_binary']

        # 2. Разделите датасет на обучающую и тестовую подвыборки, обучите модель XGBClassifier
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        train_dates = self.item_ids.loc[self.X_train.index, 'datetime']
        test_dates = self.item_ids.loc[self.X_test.index, 'datetime']

        # Определение диапазона дат для проверки прогноза покупок в тесте на основе данных из трейна
        self.train_date_range = (train_dates.min(), train_dates.max())
        self.test_date_range = (test_dates.min(), test_dates.max())

    def train_(self):
        print(f"Обучение модели ~40сек....")
        self.xgb_clf = xgb.XGBClassifier()
        self.xgb_clf.fit(self.X_train, self.y_train)
        print(f"Модель готова к работе.")
        self.initialized = True

    def precision_at_k(self,y_true, y_pred_proba, k=3):
        sorted_items = np.argsort(-y_pred_proba)
        top_k = sorted_items[:k]
        relevant_items = np.isin(top_k, y_true)
        score = relevant_items.sum() / k
        return score
    def showmetric(self):
        # Определение метрики
        items_predictions = self.xgb_clf.predict_proba(self.X_test)[:, 1]  # Вероятность положительного класса
        precision_at_3 = np.mean([self.precision_at_k([item], prediction, k=3) for item, prediction in zip(self.y_test, items_predictions)])
        return precision_at_3

    def get_top3_recommendations(self,visitorid, purchased_final, saved_item_ids):
        if visitorid in purchased_final.visitorid.unique():
            user_data = purchased_final[purchased_final['visitorid'] == visitorid]
            user_data = user_data.drop('purchased_binary', axis=1)
            y_proba = self.xgb_clf.predict_proba(user_data)
            
            sorted_indices = np.argsort(y_proba[:, 1])[::-1][:3]
            
            # Получаем индексы в `purchased_final`, соответствующие sorted_indices
            top3_indices = user_data.index[sorted_indices]
            
            # Используем эти индексы для получения itemid из saved_item_ids
            recommended_items = saved_item_ids.loc[top3_indices]['itemid'].values
            return True,recommended_items
        # если клиент не проявлял активности в просмотрах, то предложить самые популярные товары
        else:
            return False,saved_item_ids['itemid'].value_counts().head(3).index.tolist()
    def apigettop3(self,visitorid_= "1404265"):
        try:
            visitorid = int(visitorid_)
        except ValueError as e:
            print(f"Error converting visitorid to int: {e}")
            return -1,False,"идентификатор клиента должен быть целым числом"
        # передать только обучающий фрагмент базы (первые 80%, на которых обучалась модель)
        # чтобы проверить, как работают предсказания для покупателей на последних 20%
        # Получение индексов, соответствующих условию пограничной даты деления 80%/20%
        filtered_indices = self.item_ids[self.item_ids['datetime'] >= self.test_date_range[0]].index
        # Применение фильтра к purchased_final на основе индексов
        filtered_purchased_final = self.purchased_final.loc[filtered_indices]
        isuser,top3_recommendations=self.get_top3_recommendations(visitorid,
                                        filtered_purchased_final, self.item_ids.loc[filtered_indices])
        return 0,isuser,top3_recommendations
