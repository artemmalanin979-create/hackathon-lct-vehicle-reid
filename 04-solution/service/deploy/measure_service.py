#!/usr/bin/env python3
"""Bounded real HTTP search benchmark; no model/data replacement or external writes."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import platform
import statistics
import time
import urllib.request
from pathlib import Path


def request(url, body=None, content_type=None):
    req = urllib.request.Request(url, data=body, headers={'Content-Type': content_type} if body else {})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def multipart(image, box, gallery_id=None):
    boundary = 'LctMeasuredMultipart20260925'
    values = {**box, 'top_k': 10}
    if gallery_id is not None:
        values = {**box, 'gallery_id': gallery_id}
    chunks = []
    for key, value in values.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="demo.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode(), image,
               f'\r\n--{boundary}--\r\n'.encode()]
    return b''.join(chunks), f'multipart/form-data; boundary={boundary}'


def quantile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:18070')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=20)
    args = ap.parse_args()
    if not 5 <= args.samples <= 100:
        ap.error('samples must be 5..100')
    base = args.url.rstrip('/')
    demo = json.loads(request(base+'/api/demo'))['examples']
    inputs=[]
    for example in demo:
        image=request(base+example['image_url'])
        body,typ=multipart(image,example['bbox'])
        inputs.append((example,image,body,typ))
    if not inputs:
        raise SystemExit('Demo inputs are not installed')
    def search(i):
        example,image,body,typ=inputs[i%len(inputs)]
        begin=time.perf_counter(); response=json.loads(request(base+'/api/search',body,typ))
        elapsed=time.perf_counter()-begin
        assert isinstance(response['refusal'],bool) and response['refusal']==(not response['candidates'])
        assert all(c['confidence']<=1 and c['confidence']>=-1 for c in response['candidates'])
        return {'example':example['id'],'seconds':elapsed,'refusal':response['refusal'],'candidates':len(response['candidates'])},response
    for i in range(3): search(i)
    report={'url':base,'python':platform.python_version(),'host':platform.platform(),'warmup':3,
            'version':json.loads(request(base+'/api/version')),
            'inputs':[{'example':x[0],'sha256':hashlib.sha256(x[1]).hexdigest()} for x in inputs],
            'cold_start':'NOT MEASURED','phases':{}}
    for concurrency,n in [(1,args.samples),(2,10)]:
        start=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results=list(pool.map(search,range(n)))
        wall=time.perf_counter()-start; samples=[x[0] for x in results]; times=[x['seconds'] for x in samples]
        report['phases'][str(concurrency)]={'concurrency':concurrency,'count':n,'elapsed_s':wall,'fps':n/wall,
            'p50_s':statistics.median(times),'p95_s':quantile(times,.95),'samples':samples}
    _,response=search(0)
    first=inputs[0]; body,typ=multipart(first[1],first[0]['bbox'],response['candidates'][0]['gallery_id'])
    start=time.perf_counter(); explanation=json.loads(request(base+'/api/explain',body,typ)); elapsed=time.perf_counter()-start
    report['explain']={'elapsed_s':elapsed,'response_keys':sorted(explanation)}
    report['status']='PASS'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:{k:v for k,v in value.items() if k!='samples'} for key,value in report['phases'].items()},indent=2))

if __name__=='__main__':main()
