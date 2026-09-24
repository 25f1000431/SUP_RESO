"""
App entrypoint. Wires together config, CORS, static files, the model
lifecycle, and the routers. Run with:

    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import OUTPUT_DIR
from app import model_loader
from app.routers import health, super_resolve

app = FastAPI(
    title="SAT-SR",
    description="Sentinel-2 Super Resolution Prototype",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    # Any localhost/127.0.0.1 port -- convenient in dev since Vite may
    # pick a different port each run. Tighten this before deploying.
    
   allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.include_router(health.router)
app.include_router(super_resolve.router)


@app.on_event("startup")
def startup_event():
    model_loader.load_model()
