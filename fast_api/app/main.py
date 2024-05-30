from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from gomodel import RecomendModel
model = None

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    global model
    await websocket.accept()
    model = RecomendModel(websocket=websocket,merged_data_fname="merged_data.csv.zip")
    await model.initialize()
    try:
        while not model.initialized:
            data = await websocket.receive_text()
            await websocket.send_text(f"Received message: {data}")
    except WebSocketDisconnect:
        pass
    await websocket.send_text(f"Инициализация модели завершена.")
    # больше не будет использоваться для отправки сообщений
    await websocket.close()
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    global model
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit/", response_class=HTMLResponse)
async def handle_form(request: Request, input_text: str = Form(...)):
    global model
    result = model.gettop3(input_text)
    return templates.TemplateResponse("form.html", {"request": request, "result": result, "input_text": input_text})

if __name__ == "__main__":
    pass
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
