import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect,HTTPException
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from gomodel import RecomendModel
from pydantic import BaseModel
from html import escape

model = None
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    global model
    if model is not None:
        await websocket.close()
        raise HTTPException(status_code=403, detail="Model already initialized")
    await websocket.accept()
    model = RecomendModel(websocket=websocket,merged_data_fname="merged_data.csv.zip")
    # Create a task to send messages periodically
    send_messages_task = asyncio.create_task(send_messages(websocket, model))
    # Initialize the model
    await model.initialize()
    # Wait for the send_messages task to complete
    await send_messages_task

async def send_messages(websocket: WebSocket, model: RecomendModel):
    while not model.initialized:
        await asyncio.sleep(1)  # Wait for 1 second between messages
        await websocket.send_text(f".")
    await websocket.send_text(f"Инициализация модели завершена.")
    await websocket.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    global model
    ws_needed = model is None
    return templates.TemplateResponse("index.html", {"request": request, "ws_needed": ws_needed})

# обработка нажатия кнопки на форме
@app.post("/submit/", response_class=HTMLResponse)
# async def handle_form(request: Request, input_text: str = Form(...)):
async def handle_form(request: Request):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not active. Start from http://127.0.0.1:5000")
    
    form_data = await request.form()
    input_text = form_data.get("input_text")    
    # Проверяем, что поле input_text заполнено
    if not input_text:    
        flag,isuser,top3_recommendations  = model.apigettop3()
        result = f"Клиент не задан, поэтому в качестве примера приводим топ 3 рекомендации для клиента {1404265}: {top3_recommendations}"
    else:
        flag,isuser,top3_recommendations  = model.apigettop3(input_text)
        if flag<0:
            result= f"Возникла ошибка: {top3_recommendations}"
        elif isuser:
            result= f"Для клиента {input_text} рекомендуются товары {str(top3_recommendations)}"
        else:
            result= f"Такой клиент {input_text} отсутствует в базе, поэтому предлагаем наиболее популярные товары: {str(top3_recommendations)}"
    # result2 = result.replace('[', '(').replace(']', ')')
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

class Item(BaseModel):
    text: str
# специальная функция для апи вызовов
@app.post("/apigettop3")
async def post_top3(clientid: Item):
    global model
    if model is None:
        return {"message":"Model is not active. Start from http://127.0.0.1:5000"}
    result,isuser,top3_recommendations=model.apigettop3(visitorid_= clientid.text)
    return {"result":result,"isuser": isuser,"top3_recommendations": str(top3_recommendations)}
# вернуть текущую метрику Precision@3 модели
@app.get("/apishowmetric")
async def post_metric():
    global model
    if model is None:
        return {"message":"Model is not active. Start from http://127.0.0.1:5000"}
    metric=model.showmetric()
    return {"Precision@3": metric}

if __name__ == "__main__":
    pass
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn main:app --port 5000
