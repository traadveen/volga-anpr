"""Bounded crop degradation with auditable parameters and no hidden target characters."""
import io
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def uniform(rng, config, key):
    return float(rng.uniform(*config[key]))


def geometry(plate, ink, plate_type, config, rng, clean):
    s=config['supersample']; W,H=config['output_sizes'][plate_type]; w,h=plate.size
    g=config['geometry']; scale=.86 if clean else uniform(rng,g,'scale')
    target=min(W*s/w,H*s/h)*scale
    centered=np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]],dtype=np.float32)*target
    angle=0 if clean else uniform(rng,g,'rotation_deg')
    rad=math.radians(angle);rotation=np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad),math.cos(rad)]])
    quad=centered@rotation.T
    if not clean:
        amplitude=uniform(rng,g,'perspective_fraction')*min(w,h)*target
        quad+=rng.uniform(-amplitude,amplitude,(4,2))
    dx=dy=0 if clean else None
    if not clean:
        dx=uniform(rng,g,'translation_fraction')*W*s;dy=uniform(rng,g,'translation_fraction')*H*s
    quad+=np.array([W*s/2+dx,H*s/2+dy])
    # Fit all four corners, never truncate plate text at frame boundaries.
    center=quad.mean(axis=0);extent=np.max(np.abs(quad-center),axis=0)
    fit=min(1.,(min(center[0],W*s-center[0])-2*s)/extent[0],(min(center[1],H*s-center[1])-2*s)/extent[1])
    quad=(quad-center)*fit+center
    src=np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix=cv2.getPerspectiveTransform(src,quad.astype(np.float32))
    gray=uniform(rng,config['appearance'],'background_gray')
    y,x=np.mgrid[:H*s,:W*s];base=gray+6*(x/(W*s)) - 4*y/(H*s)
    background=np.repeat(base[...,None],3,axis=2)
    array=np.array(plate).astype(np.float32)
    if not clean:array+=rng.normal(0,uniform(rng,config['appearance'],'plate_texture_sigma'),array.shape)
    warped=cv2.warpPerspective(array,matrix,(W*s,H*s),flags=cv2.INTER_LINEAR)
    mask=cv2.warpPerspective(np.full((h,w),255,np.uint8),matrix,(W*s,H*s),flags=cv2.INTER_LINEAR)/255.
    result=warped*mask[...,None]+background*(1-mask[...,None])
    # Preserve target-ink support for border-only occlusion/dirt decisions.
    protected=cv2.warpPerspective(np.array(ink),matrix,(W*s,H*s),flags=cv2.INTER_NEAREST)
    protected=cv2.resize(protected,(W,H),interpolation=cv2.INTER_NEAREST)>10
    result=Image.fromarray(np.clip(result,0,255).astype(np.uint8)).resize((W,H),Image.Resampling.LANCZOS)
    matrix=np.diag([1/s,1/s,1.])@matrix
    footprint=Image.fromarray((mask*255).astype(np.uint8)).resize((W,H),Image.Resampling.LANCZOS)
    return result, protected, quad/s, matrix, {'rotation_deg':angle,'scale':scale,'fit_factor':float(fit),'translation_px':[dx/s,dy/s]},footprint


def apply_effect(image, effect, cfg, rng, protected, footprint=None):
    a=np.array(image).astype(np.float32);H,W=a.shape[:2]; params={}
    if effect in ('brightness','contrast','gamma','noise','color_temperature','overexposure','underexposure'):
        key={'noise':'noise_sigma','overexposure':'exposure','underexposure':'exposure'}.get(effect,effect)
        value=uniform(rng,cfg,key);params[key]=value
        if effect=='overexposure':value=float(rng.uniform(1.,cfg['exposure'][1]));params[key]=value
        if effect=='underexposure':value=float(rng.uniform(cfg['exposure'][0],1.));params[key]=value
        if effect=='brightness':a*=value
        elif effect=='contrast':a=(a-127.5)*value+127.5
        elif effect=='gamma':a=255*(a/255)**value
        elif effect=='noise':a+=rng.normal(0,value,a.shape)
        elif effect=='color_temperature':a*=np.array([value,1.,2-value])
        elif effect=='overexposure':a*=max(1.,value)
        else:a*=min(1.,value)
        return Image.fromarray(np.clip(a,0,255).astype(np.uint8)),params
    if effect in ('gaussian_blur','defocus_blur'):
        radius=uniform(rng,cfg,effect+'_px');params['radius_px']=radius
        if effect=='defocus_blur':
            # Disk point-spread function, distinct from Gaussian.
            radius=max(.1,radius);yy,xx=np.mgrid[-1:2,-1:2]
            kernel=(xx*xx+yy*yy<=max(1.,radius*radius)).astype(np.float32);kernel/=kernel.sum()
            out=(1-min(1.,radius))*a+min(1.,radius)*cv2.filter2D(a,-1,kernel)
            return Image.fromarray(np.clip(out,0,255).astype(np.uint8)),params
        return image.filter(ImageFilter.GaussianBlur(radius)),params
    if effect=='motion_blur':
        length=max(1,int(round(uniform(rng,cfg,'motion_blur_px'))));params['length_px']=length
        kernel=np.zeros((length,length),np.float32);vertical=bool(rng.integers(2));params['vertical']=vertical
        if vertical:kernel[:,length//2]=1/length
        else:kernel[length//2,:]=1/length
        return Image.fromarray(cv2.filter2D(a,-1,kernel).clip(0,255).astype(np.uint8)),params
    if effect in ('jpeg','low_resolution'):
        quality=int(round(uniform(rng,cfg,'jpeg_quality')));params['jpeg_quality']=quality
        out=image
        if effect=='low_resolution':
            scale=uniform(rng,cfg,'low_resolution_scale');params['downscale']=scale
            out=image.resize((max(1,round(W*scale)),max(1,round(H*scale))),Image.Resampling.LANCZOS)
        buffer=io.BytesIO();out.save(buffer,format='JPEG',quality=quality,subsampling=0)
        with Image.open(io.BytesIO(buffer.getvalue())) as opened:out=opened.convert('RGB')
        # HR -> downscale -> JPEG -> upscale. Image size/geometry remain unchanged.
        return out.resize((W,H),Image.Resampling.BICUBIC),params
    if effect in ('shadow','glare'):
        strength=uniform(rng,cfg,effect+'_strength');params['strength']=strength
        yy,xx=np.mgrid[:H,:W];cx=rng.uniform(0,W);cy=rng.uniform(0,H)
        field=np.exp(-((xx-cx)**2/(.4*W)**2+(yy-cy)**2/(.7*H)**2))
        if effect=='shadow':a*=1-strength*field[...,None]
        else:a=a*(1-strength*field[...,None])+255*strength*field[...,None]
        return Image.fromarray(a.clip(0,255).astype(np.uint8)),params
    # Weather/dirty border effects avoid all target ink (with a margin).
    margin=cv2.dilate(protected.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    overlay=Image.new('L',image.size);draw=ImageDraw.Draw(overlay)
    if effect=='partial_occlusion':
        fraction=uniform(rng,cfg,'occlusion_fraction');params['fraction']=fraction
        thickness=max(1,round(H*fraction));side=int(rng.integers(2));params['side']=side
        support=np.asarray(footprint)>128 if footprint is not None else np.ones((H,W),bool)
        yy,xx=np.where(support)
        distance=cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,3)
        border=support&~margin&(distance<=thickness+1)
        by,bx=np.where(border)
        threshold=np.quantile(by,.25 if side==0 else .75)
        selected=by<=threshold if side==0 else by>=threshold
        candidates=np.column_stack([bx[selected],by[selected]])
        cx,cy=candidates[int(rng.integers(len(candidates)))]
        length=max(6,round((xx.max()-xx.min())*.12))
        draw.rectangle((int(cx-length//2),int(cy-thickness//2),int(cx+length//2),int(cy+thickness//2+1)),fill=180)
        params['target_ink_occluded']=False
        color=(48,43,35)
    else:
        density=uniform(rng,cfg,effect+'_density');params['density']=density
        for _ in range(round(W*H*density)):
            x=int(rng.integers(W));y=int(rng.integers(H))
            if effect=='rain':draw.line((x,y,x+2,y+6),fill=55,width=1)
            else:draw.ellipse((x-1,y-1,x+2,y+2),fill=90)
        color=(222,228,234) if effect=='rain' else ((240,241,238) if effect=='snow' else (94,75,44))
    alpha=np.array(overlay);alpha[margin]=0
    if effect=='partial_occlusion':alpha[~support]=0
    return Image.composite(Image.new('RGB',image.size,color),image,Image.fromarray(alpha)),params


def degrade(image, protected, config, rng, clean, footprint=None):
    cfg=config['augmentation'];trace=[]
    if clean or rng.random()>cfg['probability']:return image,trace
    n=int(rng.integers(1,cfg['max_effects']+1))
    effects=rng.choice(cfg['effects'],size=n,replace=False).tolist()
    for effect in effects:
        image,params=apply_effect(image,effect,cfg,rng,protected,footprint)
        trace.append({'effect':effect,**params})
    return image,trace
