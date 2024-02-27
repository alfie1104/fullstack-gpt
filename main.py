from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Dante Quote Giver",
    description="Get a real quote said by Dante himself."
)

class Quote(BaseModel):
    quote: str = Field(description="The quote that Dante said.")
    year: int = Field(description="The year when Dante said the quote.")

@app.get("/quote", summary="Returns a random quote by Danten"
         , description="Upon receiving a GET request this endpoint will return a real quiote said by Dante himself."
         , response_description="A Quote object that contains the quote said by Dante and the date when the quote was said."
         , response_model = Quote)
def get_quote():
    return {
        "quote":"Life is short so eat it all.",
        "year":1995
    }