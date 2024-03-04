from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Any
from dotenv import load_dotenv
import os

import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

load_dotenv() # FastAPI에서 .env의 모든 변수를 읽어들이도록 하기 위해 load_dotenv 함수 호출. Jupyter notebook과 달리 fastapi에서는 이걸 호출하지 않으면 os.getenv()에서 .env파일에 적힌 키를 입력해도 값을 얻을 수 없음

# Pinecone은 cloud vector DB임 (FAISS같은 vector store를 cloud 환경에서 제공)
pinecone.init(
    apy_key=os.getenv("PINECONE_API_KEY"),
    environment="gcp-starter"
)


embeddings = OpenAIEmbeddings()
vector_store = Pinecone.from_existing_index("recipes", embeddings)


app = FastAPI(
    title="Chef Dante. The best provider of Indian Recipes in the world.",
    description="Give Chef Dante the name of an ingredient and it will give you multiple recipes to use that ingredient on in return.",
    servers=[
        {"url":"https://quarters-monthly-increase-bras.trycloudflare.com "}
    ]
)

class Document(BaseModel):
    page_content: str

@app.get("/recipes", summary="Returns a list of recipes."
         , description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient."
         , response_description="A Document object that contains the recipe and preparation instructions"
         , response_model = list[Document]
         , openapi_extra={
             "x-openai-isConsequential": False
         })
def get_recipe(ingredient:str):
    # /recipes?ingredient=xxxx 로 입력되는 쿼리 파라미터를 ingredient에 할당
    docs = vector_store.similarity_search(ingredient)
    return docs

user_token_db = {
    "ABCDEF":"test_token"
}

@app.get("/authorize", response_class=HTMLResponse, include_in_schema=False)
def handle_authorize(client_id:str, redirect_uri:str, state:str):
    return f"""
    <html>
        <head>
            <title>Dante Log In</title>
        </head>
        <body>
            <h1>Log Into Dante</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Dante</a>
        </body>
    </html>
    """

@app.post("/token", include_in_schema=False)
def handle_token(code = Form(...)):
    print(code)
    # OpenAI는 OAuth에 대한 return값으로 access_token이 존재하면 OAuth가 성공했다고 생각함
    return {
        "access_token":user_token_db[code]
    }