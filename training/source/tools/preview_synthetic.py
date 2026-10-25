"""Static contact sheet from actual images, with CSV labels and optional geometry."""
import argparse
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset.schema import read_metadata,resolve_image


def make_preview(root,output,indices,overlay=False):
    root=Path(root);output=Path(output)
    if output.resolve().is_relative_to(root.resolve()):raise ValueError('Preview must be outside dataset')
    rows,issues=read_metadata(root/'meta.csv')
    if issues:raise ValueError('Invalid metadata')
    if not indices or any(i<0 or i>=len(rows) for i in indices):raise ValueError('Preview indices outside dataset')
    cols=3;cellw=440;cellh=235
    canvas=Image.new('RGB',(cols*cellw,((len(indices)+cols-1)//cols)*cellh),(223,228,230));draw=ImageDraw.Draw(canvas)
    font=ImageFont.load_default(size=17)
    for position,index in enumerate(indices):
        row=rows[index];x=position%cols*cellw;y=position//cols*cellh
        with Image.open(resolve_image(root,row.image)) as opened:im=opened.convert('RGB')
        original_width=im.width
        im.thumbnail((cellw-20,178));scale=im.width/original_width
        ox=x+(cellw-im.width)//2;oy=y+10+(178-im.height)//2
        canvas.paste(im,(ox,oy))
        if overlay:
            draw.line([(ox+px*scale,oy+py*scale) for px,py in (*row.quad,row.quad[0])],fill=(18,220,70),width=2)
        draw.text((x+12,y+190),f'{index:04}  {row.plate_type}  {row.plate_num}',font=font,fill=(16,28,37))
        tags=', '.join(t for t in row.conditions if t not in ('synthetic','plate_crop','mixed'))
        draw.text((x+12,y+213),tags[:52],font=ImageFont.load_default(size=12),fill=(56,64,72))
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+',default=list(range(18)));p.add_argument('--overlay',action='store_true');a=p.parse_args(argv)
    try:make_preview(a.dataset,a.output,a.indices,a.overlay)
    except (OSError,ValueError) as error:p.exit(2,str(error)+'\n')
    print(str(a.output.resolve()));return 0


if __name__=='__main__':raise SystemExit(main())
