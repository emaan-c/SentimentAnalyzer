from fastapi import FastAPI
from sentimentAnalysis import sentiment_analysis
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/sentiment/{sentence}")
async def get(sentence: str):
    return JSONResponse(status_code=200, content={"result": sentiment_analysis(sentence)})