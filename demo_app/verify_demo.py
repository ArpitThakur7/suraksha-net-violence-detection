"""Exercise the running local demo using three reference recordings."""
from pathlib import Path
import json
import httpx

ROOT = Path(__file__).resolve().parents[1]
cases = {
    "SFD-normal-nofi001.mp4": "sfd/noFight/nofi001.mp4",
    "AIRTLab-staged-slap-1.mp4": "avd/violence-detection-dataset/violent/cam1/1.mp4",
    "AIRTLab-hug-highfive-1.mp4": "avd/violence-detection-dataset/non-violent/cam1/1.mp4",
}


def main():
    with httpx.Client(base_url="http://127.0.0.1:8741", timeout=180, trust_env=False) as client:
        assert "Recording review" in client.get("/").raise_for_status().text
        existing = {item["name"]: item["id"] for item in client.get("/api/videos").raise_for_status().json()}
        checked = []
        for name, path in cases.items():
            if name in existing:
                result = client.get("/api/videos/" + existing[name]).raise_for_status().json()
            else:
                with (ROOT / "datasets/public" / path).open("rb") as video:
                    result = client.post("/api/videos", files={"file": (name, video, "video/mp4")},
                                         data={"threshold":"0.50"}).raise_for_status().json()
            base = "/api/videos/" + result["id"]
            media = client.get(base + "/media", headers={"Range": "bytes=0-63"}).raise_for_status()
            assert b"ftyp" in media.content[:64]
            windows = result["windows"]
            assert windows and all(0 <= w["score"] <= 1 for w in windows)
            assert windows[0]["start"] == 0 and abs(windows[-1]["end"]-result["duration"]) < 0.01
            intervals = result["incidents"]
            assert all(a["end"] < b["start"] for a,b in zip(intervals,intervals[1:]))
            for incident in intervals:
                url = f"/api/incidents/{incident['id']}"
                assert client.get(url + "/image").raise_for_status().content.startswith(b"\xff\xd8")
                try:
                    for new_status in ("Reviewed", "False alarm"):
                        client.patch(url, json={"status":new_status}).raise_for_status()
                        saved = client.get(base).raise_for_status().json()["incidents"]
                        assert next(x for x in saved if x["id"] == incident["id"])["status"] == new_status
                finally:
                    client.patch(url, json={"status":incident["status"]}).raise_for_status()
            checked.append({"recording":name,"windows":len(windows),"flagged_windows":sum(w["score"] >= .5 for w in windows),
                            "merged_intervals":len(intervals),"max_score":round(max(w["score"] for w in windows),4)})
        before = len(client.get("/api/videos").json())
        assert client.post("/api/videos",files={"file":("broken.mp4",b"bad","video/mp4")}).status_code == 422
        assert len(client.get("/api/videos").json()) == before
        print(json.dumps(checked,indent=2))


if __name__ == "__main__":
    main()

