import uvicorn
from fastapi import FastAPI, Form

app = FastAPI()

@app.get("/")
def hello_world():
    return { "hello": "world" }

@app.get("/directory")
def get_directory(a: int = 2, b: int = 3):
    return { "answer": a*b }

@app.post("/form")
def post_form(username: str = Form(), password: str = Form()):
    return { "username": username, "password": password }

# Main driver function
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 4500)