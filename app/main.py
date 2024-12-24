from fastapi import FastAPI
from app.routes import router
import logging

app = FastAPI()
# Set up logging
logging.basicConfig(level=logging.INFO)
# Include the router from routes.py
app.include_router(router)
@app.get("/")
async def root():
    logging.info("GET / request received")
    return {"message": "Welcome to the Image classification"}
