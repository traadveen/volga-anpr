"""Isolated human quad sidecar. No model training, OCR labels or split edits."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from contextlib import contextmanager
import copy
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
import os
import secrets
import threading
from urllib.parse import urlparse,parse_qs
import uuid
import cv2
import numpy as np
from tools.phase_h_common import ROOT,read,now,load_split,bbox as frozen_bbox
from src.geometry_f import validate_quad,order_obb
from src.inference_io import file_sha256

DEFAULT=ROOT/'reports/phase_i'
ASSETS=ROOT/'src/phase_i_reviewer'


def sha_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp,path)


def training_region(quad,shape):
    q=np.array(quad,dtype=float)
    a,b=q.min(axis=0)
    c,d=q.max(axis=0)
    px=max(24,(c-a)*.65)
    py=max(24,(d-b)*.65)
    w,h=shape
    return [max(0,int(np.floor(a-px))),max(0,int(np.floor(b-py))),
            min(w,int(np.ceil(c+px))),min(h,int(np.ceil(d+py)))]


def geometry(quad,shape):
    if not isinstance(quad,list) or len(quad)!=4 or any(not isinstance(p,list) or len(p)!=2 or
        any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in p) for p in quad):
        raise ValueError('Нужны четыре пары числовых координат')
    w,h=shape
    q=validate_quad(quad,(h,w))
    if int(np.argmin(q.sum(axis=1)))!=0:
        raise ValueError('Первая точка — верхний левый угол; далее по часовой стрелке')
    rect=cv2.minAreaRect(q)
    obb=order_obb(cv2.boxPoints(rect)).tolist()
    angle=float(np.degrees(np.arctan2(q[1,1]-q[0,1],q[1,0]-q[0,0])))
    return q.tolist(),obb,angle


class ProtectedInputs:
    def __init__(self,hashes):
        self.hashes=hashes
        self.stats={}
        self.verify(force=True)

    def verify(self,force=False):
        for name,expected in self.hashes.items():
            p=Path(name)
            stat=p.stat()
            current=(stat.st_mtime_ns,stat.st_size)
            if force or self.stats.get(name)!=current:
                if file_sha256(p)!=expected:
                    raise ValueError('Изменился замороженный файл: '+name)
                self.stats[name]=current


class Review:
    def __init__(self,folder=DEFAULT,allow_fixture=False):
        self.folder=Path(folder).resolve()
        self.index_path=self.folder/'annotation_index.json'
        index=read(self.index_path)
        if index.get('fixture') and not allow_fixture:
            raise ValueError('Fixture mode must be explicitly enabled')
        self.index_sha=file_sha256(self.index_path)
        self.rows=index['records']
        self.by_id={r['review_id']:r for r in self.rows}
        if len(self.by_id)!=len(self.rows) or any(r['split'] not in ('train','val') for r in self.rows):
            raise ValueError('Geometry reviewer accepts only unique frozen train/val IDs')
        if not index.get('fixture'):
            frozen={r['review_id']:{**r,'split':split} for split in ('train','val') for r in load_split(split)}
            if set(frozen)!=set(self.by_id):
                raise ValueError('Geometry queue membership differs from frozen train/val')
            for ident,row in self.by_id.items():
                original=frozen[ident]
                for key in ('plate_num','plate_type','split','source_sha256','source_family',
                            'original_image_group_id','duplicate_group_id'):
                    if row.get(key,'')!=original.get(key,''):
                        raise ValueError('Geometry queue changed frozen label/group: '+ident)
                if Path(row['image']).resolve()!=Path(original['image']).resolve() or row['bbox']!=frozen_bbox(original):
                    raise ValueError('Geometry queue changed frozen image/bbox: '+ident)
        self.guard=ProtectedInputs({str(self.index_path):self.index_sha,**read(self.folder/'input_guard.json')['hashes']})
        self.token=secrets.token_urlsafe(32)
        self.lock=threading.RLock()
        self.state_path=self.folder/'geometry_review_state.json'
        if not self.state_path.exists():
            atomic_json(self.state_path,{'schema_version':1,'revision':uuid.uuid4().hex,
                'decisions':{},'history':[],'operations':{},'fixture':bool(index.get('fixture'))})
        elif any(d.get('index_sha256')!=self.index_sha for d in self.state()['decisions'].values()):
            raise ValueError('Geometry decisions bind another immutable queue version')

    @contextmanager
    def transaction(self):
        with self.lock:
            lock=self.folder/'geometry_review.lock'
            with lock.open('a+b') as stream:
                stream.seek(0)
                stream.write(b'0')
                stream.flush()
                stream.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    yield
                finally:
                    stream.seek(0)
                    if os.name=='nt':
                        msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                    else:
                        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

    def state(self):
        return read(self.state_path)

    def progress(self,state=None):
        state=state or self.state()
        result={}
        for split in ('train','val'):
            rs=[r for r in self.rows if r['split']==split]
            accepted=[r for r in rs if state['decisions'].get(r['review_id'],{}).get('quad_verified')]
            result[split]={'total':len(rs),'verified_quad':len(accepted),'verified_OBB':len(accepted),
                'training_region_complete':sum(state['decisions'][r['review_id']]['training_region_complete'] for r in accepted),
                'per_type':{k:{'total':sum(r['plate_type']==k for r in rs),
                    'verified_quad':sum(r['plate_type']==k for r in accepted)} for k in ('type1','type1a','type1b')}}
        result.update(reviewed=sum(d['status'] in ('accepted','rejected') for d in state['decisions'].values()),total=len(self.rows),
            pending_corrections=sum(d['status']=='pending' for d in state['decisions'].values()),
            held=sum(d['status']=='held' for d in state['decisions'].values()),
            rejected=sum(d['status']=='rejected' for d in state['decisions'].values()),
            training_started=False,scope='PHASE I TRAIN/VAL GEOMETRY ONLY',revision=state['revision'])
        return result

    def page(self,split='all',view='pending',offset=0):
        self.guard.verify()
        if split not in ('all','train','val') or view not in ('pending','corrections','held','accepted','rejected','all','type1a_batch'):
            raise ValueError('Неверный фильтр')
        state=self.state()
        rows=[r for r in self.rows if split=='all' or r['split']==split]
        batch_ids=set()
        if view=='type1a_batch':
            path=self.folder/'type1a_review_batch.json'
            if path.exists():
                batch=read(path)
                ids=batch['review_ids']
                if (batch.get('index_sha256')!=self.index_sha or len(ids)!=len(set(ids)) or len(ids)>50
                        or any(i not in self.by_id or self.by_id[i]['plate_type']!='type1a' for i in ids)):
                    raise ValueError('Партия type1a не соответствует замороженной очереди')
                batch_ids=set(ids)
        def included(r):
            d=state['decisions'].get(r['review_id'])
            if view=='type1a_batch':
                return r['review_id'] in batch_ids and d is not None and d['status']=='pending'
            if view=='pending':
                return d is None or d['status']=='pending'
            if view=='corrections':
                return d is not None and d['status']=='pending'
            return view=='all' or d is not None and d['status']==view
        rows=[r for r in rows if included(r)]
        offset=max(0,int(offset))%len(rows) if rows else 0
        row=copy.deepcopy(rows[offset]) if rows else None
        if row:
            row['decision']=state['decisions'].get(row['review_id'])
            row['proposal_sha256']=sha_json(self.by_id[row['review_id']].get('proposal_quad'))
        return {'item':row,'offset':offset,'remaining':len(rows),'revision':state['revision'],
                'index_sha256':self.index_sha,'progress':self.progress(state)}

    def save(self,value):
        allowed={'review_id','action','quad','reviewer','notes','human_confirmed','expected_revision',
                 'operation_id','index_sha256','source_sha256','proposal_sha256','has_other_plates'}
        if set(value)-allowed:
            raise ValueError('Текст, тип и membership менять в этом инструменте нельзя')
        if value.get('human_confirmed') is not True or value.get('action') not in ('accept','reject'):
            raise ValueError('Нужно явное решение человека')
        if not isinstance(value.get('reviewer'),str) or not value['reviewer'].strip():
            raise ValueError('Укажите проверяющего')
        ident=value.get('review_id')
        row=self.by_id.get(ident)
        if row is None:
            raise ValueError('Карточка не входит в frozen train/val')
        if (value.get('index_sha256')!=self.index_sha or value.get('source_sha256')!=row['source_sha256'] or
            value.get('proposal_sha256')!=sha_json(row.get('proposal_quad'))):
            raise ValueError('Изменился источник или предложение; обновите карточку')
        if not isinstance(value.get('operation_id'),str) or not value['operation_id']:
            raise ValueError('Нужен operation_id')
        self.guard.verify()
        if file_sha256(Path(row['image']))!=row['source_sha256']:
            raise ValueError('Фото изменилось после создания очереди')
        with self.transaction():
            state=self.state()
            op=value['operation_id']
            body_sha=sha_json(value)
            if op in state['operations']:
                if state['operations'][op]['body_sha256']!=body_sha:
                    raise ValueError('operation_id уже использован с другим решением')
                return {'saved':True,'idempotent':True,'revision':state['revision'],'progress':self.progress(state)}
            if value.get('expected_revision')!=state['revision']:
                raise ValueError('Решение изменилось в другой вкладке; перечитайте карточку')
            accepted=value['action']=='accept'
            decision={'review_id':ident,'split':row['split'],'plate_num':row['plate_num'],
                'plate_type':row['plate_type'],'source_sha256':row['source_sha256'],
                'index_sha256':self.index_sha,'status':'accepted' if accepted else 'rejected',
                'quad':None,'quad_verified':False,'quad_source':'human_geometry_reject',
                'obb':None,'obb_verified':False,'training_region_complete':False,
                'human_confirmed':True,'reviewer':value['reviewer'].strip(),'notes':str(value.get('notes',''))[:2000],
                'reviewed_at':now(),'operation_id':op,'proposal_was_GT':False,
                'original_bbox_unchanged':row['bbox'],'original_OCR_labels_unchanged':True}
            if accepted:
                if not isinstance(value.get('has_other_plates'),bool):
                    raise ValueError('Укажите, есть ли другие номера в области обучения')
                quad,obb,angle=geometry(value.get('quad'),row['image_size'])
                unchanged=row.get('proposal_quad') is not None and np.allclose(quad,row['proposal_quad'],atol=.01)
                decision.update(quad=quad,quad_verified=True,obb=obb,obb_verified=True,
                    quad_source='BEST_G_OBB_human_verified' if unchanged else 'human_corrected_or_drawn_quad',
                    proposal_source=row['proposal_source'],proposal_sha256=value['proposal_sha256'],
                    obb_source='minAreaRect_from_human_verified_quad',top_edge_angle_degrees=angle,
                    training_region=training_region(quad,row['image_size']),
                    training_region_complete=not value['has_other_plates'],
                    has_other_plates=value['has_other_plates'])
            previous=state['decisions'].get(ident)
            if previous and previous.get('assistant_proposal'):
                decision['assistant_proposal']=previous['assistant_proposal']
                decision['assistant_proposal_confirmed_by_human']=accepted
            state['history'].append({'review_id':ident,'before':previous,'after':decision,
                'supersedes_operation':previous.get('operation_id') if previous else None,
                'previous_status':'superseded' if previous else None})
            state['decisions'][ident]=decision
            state['revision']=uuid.uuid4().hex
            state['updated_at']=now()
            state['operations'][op]={'body_sha256':body_sha,'review_id':ident}
            self.guard.verify()
            atomic_json(self.state_path,state)
            atomic_json(self.folder/'geometry_progress.json',self.progress(state))
            return {'saved':True,'revision':state['revision'],'progress':self.progress(state)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        return

    def send_json(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        review=self.server.review
        u=urlparse(self.path)
        q=parse_qs(u.query)
        try:
            if u.path in ('/','/quad'):
                data=(ASSETS/'index.html').read_text('utf-8').replace('__TOKEN__',review.token).encode()
                mime='text/html; charset=utf-8'
            elif u.path=='/app.js':
                data=(ASSETS/'app.js').read_bytes()
                mime='text/javascript; charset=utf-8'
            elif u.path=='/api/page':
                return self.send_json(review.page(q.get('split',['all'])[0],q.get('view',['pending'])[0],q.get('offset',[0])[0]))
            elif u.path=='/api/status':
                return self.send_json(review.progress())
            elif u.path=='/api/image':
                row=review.by_id[q.get('id',[''])[0]]
                path=Path(row['image'])
                data=path.read_bytes()
                if hashlib.sha256(data).hexdigest()!=row['source_sha256']:
                    raise ValueError('Image byte binding failed')
                mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            else:
                return self.send_json({'error':'Not found'},404)
            self.send_response(200)
            self.send_header('Content-Type',mime)
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(data)
        except (KeyError,ValueError,OSError) as error:
            self.send_json({'error':str(error)},400)

    def do_POST(self):
        if self.path!='/api/decision':
            return self.send_json({'error':'Not found'},404)
        if self.headers.get('X-Review-Token')!=self.server.review.token:
            return self.send_json({'error':'Неверный токен страницы'},403)
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://127.0.0.1:{self.server.server_port}':
            return self.send_json({'error':'Wrong origin'},403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<65536:
                raise ValueError('Неверный размер запроса')
            value=json.loads(self.rfile.read(size))
            self.send_json(self.server.review.save(value))
        except (ValueError,OSError,TypeError) as error:
            self.send_json({'error':str(error)},400)


def serve(folder=DEFAULT,port=8768,allow_fixture=False):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.review=Review(folder,allow_fixture)
    print(f'PHASE I geometry reviewer http://127.0.0.1:{port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=DEFAULT)
    parser.add_argument('--port',type=int,default=8768)
    parser.add_argument('--allow-fixture',action='store_true')
    args=parser.parse_args()
    serve(args.folder,args.port,args.allow_fixture)
