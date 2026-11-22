"""Local incident intake for laptop cameras and replay workers."""
import base64
import binascii
import hmac
import os
from datetime import datetime
from io import BytesIO
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from PIL import Image, UnidentifiedImageError
from pydantic import AwareDatetime, BaseModel, Field
from typing import Literal

SCHEMA="""CREATE TABLE IF NOT EXISTS camera_alerts (
    id TEXT PRIMARY KEY, camera TEXT, location TEXT, captured TEXT, received TEXT,
    confidence REAL, snapshot BLOB, status TEXT DEFAULT 'New'
)"""

class CameraAlert(BaseModel):
    incident_id: UUID
    camera_id: str=Field(min_length=1,max_length=80)
    location: str=Field(min_length=1,max_length=160)
    timestamp: AwareDatetime
    confidence: float=Field(ge=0,le=1,allow_inf_nan=False)
    detection: Literal["Violence"]
    snapshot: str=Field(min_length=4,max_length=1_500_000)

class CameraReview(BaseModel):
    status: Literal["New","Acknowledged","Resolved","False alarm"]

def auth(x_api_key: str | None=Header(default=None)):
    expected=os.environ.get("SURAKSHA_API_KEY")
    if expected and not hmac.compare_digest(x_api_key or "",expected):
        raise HTTPException(401,"Supply the configured camera API key.")

def routes(database):
    router=APIRouter()

    @router.post("/api/alert",dependencies=[Depends(auth)])
    def receive(event:CameraAlert,response:Response):
        try:
            snapshot=base64.b64decode(event.snapshot,validate=True)
            with Image.open(BytesIO(snapshot)) as image:
                if image.format!="JPEG" or image.width*image.height>4_000_000:raise ValueError("JPEG required")
                image.verify()
        except (ValueError,OSError,UnidentifiedImageError,binascii.Error,Image.DecompressionBombError):
            raise HTTPException(422,"Supply a valid JPEG snapshot up to four megapixels.")
        identity=str(event.incident_id)
        timestamp=event.timestamp.isoformat()
        fields=(event.camera_id,event.location,timestamp,event.confidence,snapshot)
        with database() as con:
            inserted=con.execute("INSERT OR IGNORE INTO camera_alerts(id,camera,location,captured,confidence,snapshot,received) VALUES(?,?,?,?,?,?,?)",
                                (identity,*fields,datetime.now().astimezone().isoformat())).rowcount
            if not inserted:
                prior=con.execute("SELECT camera,location,captured,confidence,snapshot FROM camera_alerts WHERE id=?",(identity,)).fetchone()
                if tuple(prior)!=fields:raise HTTPException(409,"This incident ID already describes a different event.")
        response.status_code=201 if inserted else 200
        return {"incident_id":identity,"created":bool(inserted)}

    @router.get("/api/alerts")
    def list_alerts():
        with database() as con:
            return [dict(r) for r in con.execute("SELECT id,camera,location,captured,received,confidence,status FROM camera_alerts ORDER BY rowid DESC LIMIT 200")]

    @router.get("/api/alerts/{identity}/snapshot")
    def snapshot(identity:UUID):
        with database() as con:
            row=con.execute("SELECT snapshot FROM camera_alerts WHERE id=?",(str(identity),)).fetchone()
        if row is None:raise HTTPException(404,"Unknown incident.")
        return Response(row[0],media_type="image/jpeg")

    @router.patch("/api/alerts/{identity}",dependencies=[Depends(auth)])
    def update(identity:UUID,review:CameraReview):
        with database() as con:
            if not con.execute("UPDATE camera_alerts SET status=? WHERE id=?",(review.status,str(identity))).rowcount:
                raise HTTPException(404,"Unknown incident.")
        return {"status":review.status}
    return router

