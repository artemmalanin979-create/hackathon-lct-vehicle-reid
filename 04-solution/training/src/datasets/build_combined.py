#!/usr/bin/env python3
"""Build combined training set = our data + freely-licensed open datasets.
Emits (aligned, own-first order):
  combined_train.npz : data(uint8 concat JPEG@256), offsets, vehicle_id, camera_id,
                       image_id(<U48), source(<U16), size, quality   [reid_train.py loader]
  combined_crops_208.npy : (N,208,208,3) uint8 raw                    [reid_train4.py loader]
Foreign ids/cams offset into disjoint ranges so MCNL sampler keeps them distinct.
DMT rule (arXiv:2004.10547 3.2): total foreign IDs must not exceed own IDs (1171).
Parsers: roundabout=<id>_cam<NN>_<frame>.jpg ; carla=<datetime>_<cam>_<veh>.jpg
"""
import argparse,glob,io,os,re,sys,collections,numpy as np
from PIL import Image
SIZE=256; SZ208=208; Q=95; MAXPERID=12
def enc(path):
    try: im=Image.open(path).convert("RGB")
    except Exception: return None,None
    j=im.resize((SIZE,SIZE),Image.BILINEAR); b=io.BytesIO(); j.save(b,"JPEG",quality=Q)
    r=np.asarray(im.resize((SZ208,SZ208),Image.BILINEAR),dtype=np.uint8)
    return b.getvalue(),r
def p_round(fn):
    m=re.match(r"(\d+)_cam(\d+)_(\d+)",os.path.basename(fn)); return (int(m.group(1)),int(m.group(2))) if m else None
def p_carla(fn):
    p=os.path.splitext(os.path.basename(fn))[0].split("_"); 
    try: return (int(p[2]),int(p[1])) if len(p)>=3 else None
    except: return None
P={"roundabout":p_round,"carla":p_carla}
def collect(gl,parser):
    byid={}
    for f in sorted(glob.glob(gl,recursive=True)):
        pr=parser(f)
        if pr is None: continue
        vid,cid=pr; byid.setdefault(vid,{}).setdefault(cid,[]).append(f)
    sel={}
    for vid,cams in byid.items():
        picks=[]; idx={c:0 for c in cams}
        while len(picks)<MAXPERID:
            prog=False
            for c in cams:
                if len(picks)>=MAXPERID: break
                if idx[c]<len(cams[c]): picks.append((c,cams[c][idx[c]])); idx[c]+=1; prog=True
            if not prog: break
        sel[vid]=picks
    return sel,len(byid),sum(len(v) for v in byid.values())
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--add",action="append",default=[],help="name:glob:parser:idoff:camoff")
    ap.add_argument("--own-npz",default="/home/fedora/lct-reid/data/train_crops.npz")
    ap.add_argument("--own-raw",default="/home/fedora/lct-reid/data/crops_208.npy")
    ap.add_argument("--out-npz",default="/home/fedora/lct-reid/jobs/job_44/combined_train.npz")
    ap.add_argument("--out-raw",default="/home/fedora/lct-reid/jobs/job_44/combined_crops_208.npy")
    ap.add_argument("--cap",type=int,default=1171)
    a=ap.parse_args()
    z=np.load(a.own_npz,allow_pickle=False); own_raw=np.load(a.own_raw,mmap_mode="r")
    ov,oc=z["vehicle_id"].astype(np.int64),z["camera_id"].astype(np.int64)
    assert own_raw.shape[0]==len(ov),(own_raw.shape,len(ov))
    own_ids=len(np.unique(ov)); print(f"own: imgs={len(ov)} ids={own_ids} cams={len(np.unique(oc))} raw={own_raw.shape}")
    fj=[]; fr=[]; fv=[]; fc=[]; fi=[]; fs=[]; ftot=0
    for spec in a.add:
        parts=spec.split(":"); name,gl,parser=parts[0],parts[1],parts[2]
        idoff=int(parts[3]); camoff=int(parts[4])
        sel,nid,nf=collect(gl,P[parser]); print(f"{name}: files={nf} ids={nid} -> using ids={len(sel)}")
        for vid,picks in sel.items():
            for c,f in picks:
                jb,rr=enc(f)
                if jb is None: continue
                fj.append(jb); fr.append(rr); fv.append(idoff+vid); fc.append(camoff+c)
                fi.append(f"{name}_{vid}_{c}_{len(fj)}"); fs.append(name)
        ftot+=len(sel)
    print("foreign ids total=%d (cap %d) %s"%(ftot,a.cap,"OK" if ftot<=a.cap else "OVER-DMT"))
    # raw npy
    if fr:
        comb_raw=np.concatenate([np.asarray(own_raw),np.stack(fr)],0)
    else:
        comb_raw=np.asarray(own_raw)
    np.save(a.out_raw,comb_raw); print(f"WROTE {a.out_raw} shape={comb_raw.shape}")
    # npz (jpeg schema)
    od=bytes(z["data"]); ooff=z["offsets"].astype(np.int64)
    fdata=b"".join(fj); p=int(ooff[-1]); foff=[]
    for b in fj: p+=len(b); foff.append(p)
    all_off=np.concatenate([ooff,np.array(foff,dtype=np.int64)])
    all_data=np.frombuffer(od+fdata,dtype=np.uint8)
    all_v=np.concatenate([ov,np.array(fv,dtype=np.int64)])
    all_c=np.concatenate([oc,np.array(fc,dtype=np.int64)])
    all_i=np.concatenate([z["image_id"].astype("<U48"),np.array(fi,dtype="<U48")])
    all_s=np.concatenate([np.array(["own"]*len(ov),dtype="<U16"),np.array(fs,dtype="<U16")])
    assert len(all_off)==len(all_v)+1 and len(all_v)==comb_raw.shape[0]
    np.savez(a.out_npz,data=all_data,offsets=all_off,vehicle_id=all_v,camera_id=all_c,
             image_id=all_i,source=all_s,size=np.array([SIZE]),quality=np.array([Q]))
    print(f"WROTE {a.out_npz}: imgs={len(all_v)} ids={len(np.unique(all_v))} cams={len(np.unique(all_c))}")
    print("by source:",dict(collections.Counter(all_s.tolist())))
if __name__=="__main__": main()
