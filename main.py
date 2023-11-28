import pickle
from fastapi import FastAPI, File, UploadFile , Request

app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello, I am alive"



@app.post("/section")
async def section(request: Request):

    data = await request.json()

    with open("shared_variable.pickle", "wb") as f: 
        pickle.dump(data, f) 

    from summarize import summarize
    summary = summarize()


    return {"summary": summary}



# @app.post("/section/summarize")
# async def section(request: Request):

#     from summarize import summarize
#     summary = summarize()

#     return {"summary": summary}


