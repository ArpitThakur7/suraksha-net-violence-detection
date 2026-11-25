"""Live CCTV monitor: read one camera (webcam index, RTSP/HTTP stream or a recording replayed in real
time), score rolling 4-second windows with the distilled MotionLite model, and raise alerts.

Preprocessing matches the pipeline's test-time path (pipeline/data/dataset.py): frames at <= 15 fps,
short side resized to 256, centre crop 224, 16 frames sampled uniformly across the window, ImageNet
normalisation. Alerts go into the app's camera_alerts table, the same queue the camera workers use.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(__file__).resolve().parent / "models"
CHECKPOINT = MODELS / "motionlite_kd_p2_all.pt"
sys.path.insert(0, str(ROOT / ".ml_deps"))
import cv2  # noqa: E402  (OpenCV lives in .ml_deps)

WINDOW_S = 4.0          # seconds of video per decision
STRIDE_S = 1.0          # one decision per second
CACHE_FPS = 15.0        # the pipeline decodes clips at 15 fps
SHORT_SIDE, CROP = 256, 224
HISTORY, VOTES = 5, 3   # alert when 3 of the last 5 windows are at or above the threshold
COOLDOWN_S = 20.0       # at most one alert per camera every 20 s
TREND_LEN = 180         # keep three minutes of scores for the chart


@lru_cache(maxsize=1)
def model_bundle():
    """Distilled MotionLite trained on all three datasets (p2_all, seed 1, test ROC-AUC 0.903)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pipeline.models import build_model
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], pretrained=False, T=int(ckpt["T"]), **(ckpt.get("model_kw") or {}))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    mean, std = ckpt["norm"]
    mean = torch.tensor(mean, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 1, 3, 1, 1)
    return {"model": model, "device": device, "mean": mean, "std": std, "T": int(ckpt["T"]),
            "threshold": float(ckpt["threshold"]), "name": "MotionLite + distillation (all three datasets)"}


def prepare(bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> RGB uint8 [224,224,3]: short side to 256, centre crop 224."""
    h, w = bgr.shape[:2]
    s = SHORT_SIDE / min(h, w)
    small = cv2.resize(bgr, (max(CROP, round(w * s)), max(CROP, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    top, left = (h - CROP) // 2, (w - CROP) // 2
    return cv2.cvtColor(small[top:top + CROP, left:left + CROP], cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def score(frames: list[np.ndarray]) -> tuple[float, float]:
    """Violence probability for one window and the forward time in ms."""
    b = model_bundle()
    idx = np.linspace(0, len(frames) - 1, b["T"]).round().astype(int)
    x = torch.from_numpy(np.stack([frames[i] for i in idx])).to(b["device"])
    x = x.permute(0, 3, 1, 2).unsqueeze(0).float().div_(255)
    x = (x - b["mean"]) / b["std"]
    t = time.perf_counter()
    p = float(b["model"](x).softmax(1)[0, 1])
    if b["device"].type == "cuda":
        torch.cuda.synchronize()
    return p, (time.perf_counter() - t) * 1000


def open_source(source: str):
    if source.isdecimal():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(int(source))
        return cap, True
    if source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        return cap, True
    path = Path(source)
    if not path.is_file():
        raise ValueError("No camera or file at that source. Use a webcam number, an rtsp:// URL or a video file path.")
    return cv2.VideoCapture(str(path)), False


class LiveMonitor:
    """One camera at a time. Thread-safe status for the web API."""

    def __init__(self, database):
        self.database = database
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self._reset()

    def _reset(self):
        self.running = False
        self.source = self.camera = self.location = None
        self.started = None
        self.error = None
        self.frames: deque = deque()          # (t, rgb224) within the last WINDOW_S
        self.jpeg: bytes | None = None
        self.fps = 0.0
        self.scores: deque = deque(maxlen=HISTORY)
        self.trend: deque = deque(maxlen=TREND_LEN)
        self.latest = None
        self.forward_ms = None
        self.alert_active = False
        self.last_alert_at = -1e9
        self.alerts = 0

    # ---- control -------------------------------------------------------------------------
    def start(self, source: str, camera: str, location: str, loop: bool = True):
        self.stop()
        cap, live = open_source(source)
        if not cap.isOpened():
            cap.release()
            raise ValueError("Could not open that source. Check the webcam number, stream URL or file path.")
        model_bundle()  # load the model before reporting "running"
        with self.lock:
            self._reset()
            self.running, self.source, self.camera, self.location = True, source, camera, location
            self.started = time.time()
        self.stop_event.clear()
        self.threads = [threading.Thread(target=self._capture, args=(cap, live, loop), daemon=True),
                        threading.Thread(target=self._scorer, daemon=True)]
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=5)
        self.threads = []
        with self.lock:
            self.running = False

    # ---- workers -------------------------------------------------------------------------
    def _capture(self, cap, live: bool, loop: bool):
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and 1 <= fps <= 120 else 25.0
        t0 = time.perf_counter()
        n, last_kept, last_jpeg, rate_t, rate_n = 0, -1e9, -1e9, time.perf_counter(), 0
        try:
            while not self.stop_event.is_set():
                ok, bgr = cap.read()
                if not ok:
                    if not live and loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    with self.lock:
                        self.error = "The stream ended or stopped responding." if live else "The recording finished."
                    break
                n += 1
                if live:
                    now = time.perf_counter() - t0
                else:  # replay a file at its own speed so it behaves like a camera
                    now = time.perf_counter() - t0
                    target = n / fps
                    if target > now:
                        time.sleep(target - now)
                    now = max(now, target)
                rate_n += 1
                if time.perf_counter() - rate_t >= 1:
                    with self.lock:
                        self.fps = rate_n / (time.perf_counter() - rate_t)
                    rate_t, rate_n = time.perf_counter(), 0
                if now - last_kept >= 1 / CACHE_FPS - 1e-6:
                    last_kept = now
                    rgb = prepare(bgr)
                    with self.lock:
                        self.frames.append((now, rgb))
                        while self.frames and self.frames[0][0] < now - WINDOW_S:
                            self.frames.popleft()
                if now - last_jpeg >= 0.08:  # ~12 fps preview
                    last_jpeg = now
                    h, w = bgr.shape[:2]
                    if w > 960:
                        bgr = cv2.resize(bgr, (960, round(h * 960 / w)), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self.lock:
                            self.jpeg = buf.tobytes()
        except Exception as exc:  # report, never crash the server
            with self.lock:
                self.error = f"Camera error: {exc}"
        finally:
            cap.release()
            with self.lock:
                self.running = False

    def _scorer(self):
        b = model_bundle()
        thr = b["threshold"]
        while not self.stop_event.wait(STRIDE_S):
            with self.lock:
                if not self.running:
                    break
                window = [f for _, f in self.frames]
                span = (self.frames[-1][0] - self.frames[0][0]) if self.frames else 0
            if len(window) < 8 or span < WINDOW_S * 0.75:
                continue
            try:
                p, ms = score(window)
            except Exception as exc:
                with self.lock:
                    self.error = f"Model error: {exc}"
                continue
            now = time.time()
            with self.lock:
                self.latest, self.forward_ms = p, ms
                self.scores.append(p)
                self.trend.append((now, p))
                hits = sum(s >= thr for s in self.scores)
                self.alert_active = len(self.scores) == HISTORY and hits >= VOTES
                fire = self.alert_active and now - self.last_alert_at >= COOLDOWN_S
                snapshot = window[len(window) // 2] if fire else None
                if fire:
                    self.last_alert_at = now
                    self.alerts += 1
                camera, location = self.camera, self.location
            if fire:
                self._record(camera, location, p, snapshot)

    def _record(self, camera, location, confidence, rgb):
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return
        stamp = datetime.now().astimezone().isoformat()
        with self.database() as con:
            con.execute("INSERT INTO camera_alerts(id,camera,location,captured,confidence,snapshot,received) "
                        "VALUES(?,?,?,?,?,?,?)", (str(uuid4()), camera, location, stamp, confidence, jpeg.tobytes(), stamp))

    # ---- read side -----------------------------------------------------------------------
    def status(self) -> dict:
        b = model_bundle() if CHECKPOINT.exists() else None
        with self.lock:
            return {
                "running": self.running, "source": self.source, "camera": self.camera, "location": self.location,
                "started": self.started, "error": self.error, "fps": round(self.fps, 1),
                "score": self.latest, "forward_ms": None if self.forward_ms is None else round(self.forward_ms, 1),
                "alert": self.alert_active and self.running, "alerts": self.alerts,
                "trend": [[round(t, 2), round(p, 4)] for t, p in self.trend],
                "threshold": b["threshold"] if b else None, "model": b["name"] if b else None,
                "device": ("GPU" if b and b["device"].type == "cuda" else "CPU"),
                "window_s": WINDOW_S, "stride_s": STRIDE_S, "votes": f"{VOTES} of {HISTORY}",
            }

    def frame(self) -> bytes | None:
        with self.lock:
            return self.jpeg
