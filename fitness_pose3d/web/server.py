from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app = FastAPI(title="Fitness Pose 3D Web")

app.mount("/", StaticFiles(directory="web/static", html=True), name="static")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}