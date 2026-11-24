"""Make numbered source plates from Quimbo Fig. 6.1 and the SONNE-95 core logs."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from .pinatubo_offshore import REV,read


def plates(folder):
    # Inputs are the original page images extracted using the commands in source-notes.md.
    farm=Image.open(folder/"farm-source.png").convert("RGB")
    logs=Image.open(folder/"prof-076.png").convert("RGB")
    output=REV/"figures"
    output.mkdir(exist_ok=True)
    farm.save(output/"quimbo-figure6-1.png")
    logs.save(output/"so95-sections.png")
    annotated=farm.copy()
    ink=ImageDraw.Draw(annotated)
    sites=read(REV/"quimbo-farms.csv")
    cards=Image.new("RGB",(900,((len(sites)+3)//4)*147),"white")
    pen=ImageDraw.Draw(cards)
    for i,site in enumerate(sites):
        x0,y0=(i%4)*225,(i//4)*147
        pen.text((x0+5,y0+2),site["site"]+" "+site["barangay"],fill="black")
        if not site["printed_figure_x"]:
            pen.text((x0+5,y0+33),"Unresolved location on plate",fill="black")
            continue
        x,y=int(site["printed_figure_x"]),int(site["printed_figure_y"])
        annotated_size=7
        ink.ellipse((x-annotated_size,y-annotated_size,x+annotated_size,y+annotated_size),outline="#005bc7",width=2)
        crop=farm.crop((x-32,y-18,x+32,y+18)).resize((192,108),Image.Resampling.NEAREST)
        cards.paste(crop,(x0+5,y0+18))
        pen.text((x0+5,y0+128),f"{x}, {y} pixels",fill="black")
    cards.save(output/"farm-cards.png")
    annotated.save(output/"farm-locations.png")
    x_ranges={"17949":(480,544),"17951":(252,308),"17952":(328,385),"17962":(521,585)}
    page=logs.copy()
    ink=ImageDraw.Draw(page)
    cards=Image.new("RGB",(680,340),"white")
    pen=ImageDraw.Draw(cards)
    for i,r in enumerate(v for v in read(REV/"so95-surface.csv") if v["profile_top_y"]):
        left,right=x_ranges[r["station"]]
        top,base=int(r["profile_top_y"]),int(r["column_base_y"])
        ink.rectangle((left-3,top-4,right+3,base+4),outline="#005bc7",width=2)
        crop=logs.crop((left-3,top-7,right+4,base+7)).resize(((right-left+7)*2,(base-top+14)*2),Image.Resampling.NEAREST)
        cards.paste(crop,(170*i+4,28))
        pen.text((170*i+4,4),"GIK "+r["station"],fill="black")
        mark=28+2*(int(r["base_of_mixed_y"])-top+7)
        pen.line((170*i+3,mark,170*i+25,mark),fill="#005bc7",width=2)
    page.save(output/"so95-reviewed.png")
    cards.save(output/"so95-cards.png")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_images",type=Path)
    plates(p.parse_args().source_images)
