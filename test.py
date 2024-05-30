
import sys
import os
# Добавляем путь к корневой папке проекта
apppath=os.path.join(os.getcwd(), 'fast_api/app')
sys.path.append(apppath)
print(sys.path)
# Теперь можно импортировать модуль
from gomodel import RecomendModel
model = RecomendModel(merged_data_fname="merged_data.csv.zip")
print(f'Precision@3: {model.showmetric():.4f}')
print(model.gettop3())