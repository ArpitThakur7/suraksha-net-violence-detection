import json
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

import time

import av
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .video import ROOT, analyze, profile
from .alerts import SCHEMA, routes
from .live import LiveMonitor

SITE = ROOT / "site" / "suraksha_net.html"
SAMPLES = Path(__file__).resolve().parent / "samples"

DATA = Path(os.environ.get("SURAKSHA_STORAGE", Path(__file__).resolve().parent / "data")).resolve()
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
            conn.execute("ALTER TABLE videos ADD COLUMN model TEXT DEFAULT 'Baseline ResNet18-LSTM'")
    yield
    monitor.stop()


app = FastAPI(title="Suraksha-Net", lifespan=lifespan)
app.include_router(routes(database))
monitor = LiveMonitor(database)


@app.get("/")
def home():
    """The project site, with the live monitor switched on because the API is reachable here.
    site/suraksha_net.html is also published as a static page, so it carries no <html> wrapper."""
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
    return HTMLResponse(head + SITE.read_text(encoding="utf-8") + "</body></html>")


@app.get("/review")
def review_page():
    """The original upload-and-review screen for recorded MP4s."""
    return FileResponse(Path(__file__).with_name("index.html"))


class LiveStart(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    camera: str = Field(default="CAM-01", min_length=1, max_length=80)
    location: str = Field(default="Main entrance", min_length=1, max_length=160)
    loop: bool = True


@app.get("/api/live/sources")
def live_sources():
    """Recordings that can be replayed as if they were a live camera."""
    items = []
    if SAMPLES.is_dir():
        items += [{"name": p.name, "path": str(p)} for p in sorted(SAMPLES.iterdir())
                  if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}]
    with database() as conn:
        for row in conn.execute("SELECT id,name FROM videos ORDER BY rowid DESC LIMIT 20"):
            path = DATA / (row["id"] + ".mp4")
            if path.exists():
                items.append({"name": row["name"] + " (uploaded)", "path": str(path)})
    return {"recordings": items}


@app.post("/api/live/start")
def live_start(body: LiveStart):
    try:
        monitor.start(body.source.strip(), body.camera.strip(), body.location.strip(), body.loop)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return monitor.status()


@app.post("/api/live/stop")
def live_stop():
    monitor.stop()
    return monitor.status()


@app.get("/api/live/status")
def live_status():
    return monitor.status()


@app.get("/api/live/stream")
def live_stream():
    """MJPEG preview of the camera (about 12 frames per second)."""
    def frames():
        last = None
        idle = 0
        while idle < 600:  # give up after ~60 s with no new frames
            jpeg = monitor.frame()
            if jpeg is not None and jpeg is not last:
                last, idle = jpeg, 0
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            else:
                idle += 1
            time.sleep(0.08)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})

@app.get("/api/detector")
def active_model():
    config=profile()
    engine=os.environ.get("SURAKSHA_ENGINE","torch-cpu")
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

