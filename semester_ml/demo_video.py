"""Render two real laptop detections for the semester demonstration."""
from PIL import ImageDraw,ImageFont
import av
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from suraksha.video import analyze

def font(size):
    try:return ImageFont.truetype("segoeui.ttf",size)
    except OSError:return ImageFont.load_default(size=size)

def main():
    from PIL import Image
    dest=ROOT/"optimization/camera_replay/laptop_demo.mp4";dest.parent.mkdir(exist_ok=True,parents=True)
    base=ROOT/"datasets/public/avd/violence-detection-dataset"
    clips=[("Nonviolent scene · hug / high-five",base/"non-violent/cam1/1.mp4"),("Violent scene · staged slap",base/"violent/cam1/1.mp4")]
    with av.open(str(dest),"w") as out:
        stream=out.add_stream("libx264",rate=15);stream.width=854;stream.height=540;stream.pix_fmt="yuv420p";stream.options={"crf":"22"}
        for title,path in clips:
            duration,windows,incidents=analyze(path,.4)
            with av.open(str(path)) as inp:
                decoded=[fr.to_image().convert("RGB") for fr in inp.decode(video=0)]
            for i in range(round(duration*15)):
                t=i/15;img=decoded[min(len(decoded)-1,int(t/duration*len(decoded)))]
                canvas=Image.new("RGB",(854,540),"#13272b");img.thumbnail((854,396))
                canvas.paste(img,((854-img.width)//2,53+(396-img.height)//2))
                draw=ImageDraw.Draw(canvas)
                draw.text((18,11),"Suraksha-Net  |  Recording review",font=font(23),fill="white")
                draw.text((20,456),title,font=font(20),fill="white")
                covering=[w for w in windows if w["start"]<=t<=w["end"]]
                score=max([w["score"] for w in covering] or [0]);flag=score>=.4
                text=f"{t:.1f} s    Window score {score:.3f}    Threshold 0.40    "+("FLAGGED FOR REVIEW" if flag else "Below threshold")
                draw.text((20,498),text,font=font(17),fill="#ffce88" if flag else "#bbe7d4")
                frame=av.VideoFrame.from_image(canvas)
                for pkt in stream.encode(frame):out.mux(pkt)
        for pkt in stream.encode():out.mux(pkt)
    with av.open(str(dest)) as c:
        count=sum(1 for _ in c.decode(video=0))
    print(dest,count,"verified frames")
if __name__=="__main__":main()

