import json
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

import av
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .video import ROOT, analyze, profile
from .alerts import SCHEMA, routes

DATA = Path(os.environ.get("SURAKSHA_STORAGE", ROOT / "local_data")).resolve()
LOCK = Lock()


@contextmanager
def database():
    conn = sqlite3.connect(DATA / "review.db")
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True, exist_ok=True)
    with database() as conn:
        conn.execute(SCHEMA)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
              id TEXT PRIMARY KEY, name TEXT, duration REAL, threshold REAL, windows TEXT
            );
            CREATE TABLE IF NOT EXISTS incidents (
              id INTEGER PRIMARY KEY, video TEXT, start REAL, end REAL, score REAL,
              status TEXT DEFAULT 'New', snapshot BLOB
            );
        """)
        if "model" not in {r[1] for r in conn.execute("PRAGMA table_info(videos)")}:
            conn.execute("ALTER TABLE videos ADD COLUMN model TEXT DEFAULT 'Vishnu ResNet18-LSTM'")
    yield


app = FastAPI(title="Suraksha-Net", lifespan=lifespan)
app.include_router(routes(database))


@app.get("/")
def home():
    return FileResponse(Path(__file__).with_name("index.html"))

@app.get("/api/detector")
def active_model():
    config=profile()
    engine="semester PyTorch" if "model" in config else os.environ.get("SURAKSHA_ENGINE","torch-cpu")
    return {"model":config["name"],"threshold":config["threshold"],"engine":engine}


@app.get("/api/videos")
def videos():
    with database() as conn:
        return [dict(row) for row in conn.execute("SELECT id,name,duration FROM videos ORDER BY rowid DESC")]


@app.get("/api/videos/{vid}")
def detail(vid: str):
    with database() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
        if row is None:
            raise HTTPException(404, "Recording not found.")
        incidents = conn.execute("SELECT id,start,end,score,status FROM incidents WHERE video=? ORDER BY start", (vid,))
        return {**dict(row), "windows": json.loads(row["windows"]), "incidents": [dict(x) for x in incidents]}


@app.get("/api/videos/{vid}/media")
def media(vid: str):
    with database() as conn:
        if not conn.execute("SELECT id FROM videos WHERE id=?", (vid,)).fetchone():
            raise HTTPException(404)
    return FileResponse(DATA / (vid + ".mp4"), media_type="video/mp4")


@app.get("/api/incidents/{incident}/image")
def snapshot(incident: int):
    with database() as conn:
        row = conn.execute("SELECT snapshot FROM incidents WHERE id=?", (incident,)).fetchone()
        if not row:
            raise HTTPException(404)
        return Response(bytes(row[0]), media_type="image/jpeg")


class Review(BaseModel):
    status: Literal["New", "Reviewed", "False alarm"]


@app.patch("/api/incidents/{incident}")
def review(incident: int, body: Review):
    with database() as conn:
        if not conn.execute("UPDATE incidents SET status=? WHERE id=?", (body.status, incident)).rowcount:
            raise HTTPException(404)
    return {"status": body.status}


@app.post("/api/videos", status_code=201)
def upload(file: UploadFile = File(...), threshold: float | None = Form(None, ge=0, le=1)):
    config=profile()
    if threshold is None:threshold=config["threshold"]
    name = Path(file.filename or "").name
    if not name.lower().endswith(".mp4"):
        raise HTTPException(400, "Choose an MP4 recording.")
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another recording is processing. Try again shortly.")
    vid = uuid4().hex
    path = DATA / (vid + ".mp4")
    try:
        size = 0
        with path.open("xb") as target:
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > 250 * 1024**2:
                    raise HTTPException(413, "Choose an MP4 smaller than 250 MB.")
                target.write(block)
        try:
            duration, windows, incidents = analyze(path, threshold)
        except (ValueError, av.FFmpegError) as error:
            raise HTTPException(422, "Unable to analyze recording: " + str(error)) from error
        with database() as conn:
            conn.execute("INSERT INTO videos(id,name,duration,threshold,windows,model) VALUES (?,?,?,?,?,?)", (vid, name, duration, threshold, json.dumps(windows),config["name"]))
            conn.executemany("INSERT INTO incidents(video,start,end,score,snapshot) VALUES (?,?,?,?,?)",
                             [(vid, x.start, x.end, x.score, x.snapshot) for x in incidents])
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        LOCK.release()
    return detail(vid)

