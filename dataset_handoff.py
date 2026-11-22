"""Build a new SFD/AIRTLab inventory and grouped splits for subsequent training."""
import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import av

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/public"
OUT = ROOT / "handoff"
PARTS = ("train", "validation", "test")


def sfd_groups():
    groups = {}
    source = None
    for line in (DATA / "sfd/videos.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("http"):
            url = urlparse(line.strip())
            source = "sfd:" + parse_qs(url.query).get("v", [url.path.strip("/")])[0]
        match = re.match(r"((?:no)?fi\d{3})\s*:", line)
        if match and source:
            groups[match[1]] = source
    # Published ranges overlap at 076 and identify collections, not individual recordings.
    for number in range(62, 92):
        groups[f"nofi{number:03d}"] = "sfd:camnet-synopsis-pooled"
    return groups


def inspect(job):
    file, dataset, label, group = job
    row = dict(path=file.relative_to(ROOT).as_posix(), dataset=dataset, label=label, event_group=group,
               sha256="", fps="", frames="", width="", height="", duration_s="", split="", exclusion="")
    row["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    try:
        with av.open(str(file)) as video:
            track = video.streams.video[0]
            track.codec_context.thread_count = 1
            fps = float(track.average_rate or 0)
            count = sum(1 for frame in video.decode(track))
            if not fps or not count:
                raise ValueError("empty or missing frame rate")
            row.update(fps=round(fps, 4), frames=count, width=track.width, height=track.height,
                       duration_s=round(float(track.duration*track.time_base) if track.duration else count/fps, 4))
    except (ValueError, IndexError, av.FFmpegError) as error:
        row["exclusion"] = f"decode: {error}"
    if file.stem.lower().startswith("sample_") or not group:
        row["exclusion"] = "synthetic or unmapped source"
    return row


def assign(rows):
    # Choose a deterministic allocation of whole groups that balances both class counts.
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[row["event_group"]][row["label"]] += 1
    keys = sorted(grouped)
    totals = sum(grouped.values(), Counter())
    rng = random.Random(42)
    best_loss, best = float("inf"), {}
    for _ in range(2500):
        picks = rng.choices(PARTS, weights=(70, 15, 15), k=len(keys))
        counts = {k: Counter() for k in PARTS}
        for group, target in zip(keys, picks):
            counts[target].update(grouped[group])
        loss = sum((counts[k][label] / total - fraction)**2
                   for k, fraction in zip(PARTS, (.7, .15, .15)) for label,total in totals.items())
        if loss < best_loss:
            best_loss, best = loss, dict(zip(keys, picks))
    for row in rows:
        row["split"] = best[row["event_group"]]


def main():
    jobs = []
    groups = sfd_groups()
    for file in sorted((DATA / "sfd").rglob("*.mp4")):
        label = {"fight":1, "noFight":0}.get(file.parent.name)
        if label is not None:
            jobs.append((file,"sfd",label,groups.get(file.stem, "")))
    for file in sorted((DATA / "avd").rglob("*.mp4")):
        label = {"violent":1, "non-violent":0}.get(file.parent.parent.name)
        if label is not None:
            jobs.append((file,"avd",label,f"avd:{label}:{file.stem}"))
    if not all(any(job[1] == source for job in jobs) for source in ("avd","sfd")):
        raise SystemExit("Both public collections must be present before generating the handoff.")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(inspect, jobs))
    hashes = defaultdict(list)
    for row in rows:
        hashes[row["sha256"]].append(row)
    for copies in hashes.values():
        if len(copies) > 1:
            for row in copies:
                row["exclusion"] = "repeated SHA256; review before inclusion"
    for ds in ("sfd","avd"):
        assign([r for r in rows if r["dataset"] == ds and not r["exclusion"]])
    OUT.mkdir(exist_ok=True)
    for name, selection in [("dataset_metadata", rows)] + [(k,[r for r in rows if r["split"] == k]) for k in PARTS]:
        dest = OUT / (name + ".csv")
        temp = dest.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selection)
        temp.replace(dest)
    print(Counter((r["dataset"],r["split"] or "excluded",r["label"]) for r in rows))


if __name__ == "__main__":
    main()

