"""Build comparison figures from saved model outputs; no image/model repair."""
import json,shutil,argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
root=Path("dense_uv_parser/runs/v103_semantic_revision_20260907")
p=argparse.ArgumentParser();p.add_argument("--report",default="production_review.json");p.add_argument("--output",default="visual_review");o=p.parse_args()
report=json.loads((root/o.report).read_text())
old=Path("dense_uv_parser/runs/v103_final_uv_retrain_20260907/training_completed/real_6000")
out=root/o.output;out.mkdir(exist_ok=True)
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
names=list(report["cases"])
names=["beard","crown","old_PU418PLGERMFC487_edited","old_img46","nose","old_img27","old_TWRLRRTHQP2UV368_edited"]+[n for n in names if n not in ("beard","crown","old_PU418PLGERMFC487_edited","old_img46","nose","old_img27","old_TWRLRRTHQP2UV368_edited")]
for page in range(0,len(names),3):
    subset=names[page:page+3];canvas=Image.new("RGB",(1536,42+546*len(subset)),"white");d=ImageDraw.Draw(canvas)
    for x,title in enumerate(["Original edited image","Previous v103 (cached evaluation)","Improved v103 (fresh production)"]):d.text((x*512+12,8),title,font=font,fill="black")
    for row,name in enumerate(subset):
        r=report["cases"][name];folder=Path(r["output"]);y=42+546*row
        d.text((12,y+2),name+" | "+r["role"],font=font,fill="black")
        for x,p in enumerate([Path(r["input"]),old/name/"render.png",folder/"simple_inpaint_render.png"]):
            im=Image.open(p).convert("RGB");im.thumbnail((512,512),Image.Resampling.LANCZOS)
            canvas.paste(im,(x*512+(512-im.width)//2,y+30+(512-im.height)//2))
    canvas.save(out/f"comparison_{page//3+1:02d}.png")
for name in ["beard","crown","old_PU418PLGERMFC487_edited","old_img46"]:
    folder=Path(report["cases"][name]["output"])
    for source,dest in [(folder/"pred_uv.png",out/(name+"_uv.png")),(folder/"simple_inpaint_render.png",out/(name+"_render.png"))]:shutil.copy2(source,dest)
    # Nearest-neighbor enlargement preserves every UV texel visibly.
    canvas=Image.new("RGB",(1040,318),"white");d=ImageDraw.Draw(canvas)
    for i,(title,p) in enumerate([("Previous v103",old/name/"uv.png"),("Improved v103",folder/"pred_uv.png")]):
        uv=Image.open(p).convert("RGBA").crop((0,0,64,16)).resize((512,128),Image.Resampling.NEAREST)
        checker=Image.new("RGB",uv.size);cd=ImageDraw.Draw(checker)
        for y in range(0,128,8):
            for x in range(0,512,8):cd.rectangle((x,y,x+7,y+7),fill=(230,230,230) if (x//8+y//8)%2 else (250,250,250))
        checker.paste(uv,mask=uv.getchannel("A"));canvas.paste(checker,(8+i*520,34));d.text((8+i*520,8),title,font=font,fill="black")
        d.text((8+i*520,175),"Inner head (left) | Outer head (right)",font=font,fill="black")
    canvas=canvas.crop((0,0,1040,210));canvas.save(out/(name+"_head_uv_comparison.png"))
print(str(out))
