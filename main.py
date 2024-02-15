import pickle
from fastapi import FastAPI, File, UploadFile , Request

app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello, I am alive"



@app.post("/section")
async def section(request: Request):

    data = await request.json()
    from question_answer import answer
    answer = answer(data["Question"])


    return {"Answer": answer}



