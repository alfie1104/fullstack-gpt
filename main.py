from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Any
import os

app = FastAPI(
    title="Dante Quote Giver",
    description="Get a real quote said by Dante himself.",
    servers=[
        {"url":"https://registry-mount-pd-calls.trycloudflare.com"}
    ]
)

class Quote(BaseModel):
    quote: str = Field(description="The quote that Dante said.")
    year: int = Field(description="The year when Dante said the quote.")

@app.get("/quote", summary="Returns a random quote by Danten"
         , description="Upon receiving a GET request this endpoint will return a real quiote said by Dante himself."
         , response_description="A Quote object that contains the quote said by Dante and the date when the quote was said."
         , response_model = Quote)
def get_quote(request: Request):
    print(request.headers)
    return {
        "quote":"Life is short so eat it all.",
        "year":1995
    }

user_token_db = {
    "ABCDEF":"test_token"
}

@app.get("/authorize", response_class=HTMLResponse)
def handle_authorize(client_id:str, redirect_uri:str, state:str):
    return f"""
    <html>
        <head>
            <title>Dante Log In</title>
        </head>
        <body>
            <h1>Log Into Dante</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Dante GPT</a>
        </body>
    </html>
    """

@app.post("/token")
def handle_token(code = Form(...)):
    print(code)
    # OpenAI는 OAuth에 대한 return값으로 access_token이 존재하면 OAuth가 성공했다고 생각함
    return {
        "access_token":user_token_db[code]
    }