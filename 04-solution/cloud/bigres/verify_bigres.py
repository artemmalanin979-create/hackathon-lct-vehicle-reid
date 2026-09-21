#!/usr/bin/env python3
"""Focused integration checks for changed resolution and mask geometry."""
from pathlib import Path
from types import SimpleNamespace
import json, time
import numpy as np
import onnx
from PIL import Image
from bigres import Experiment, SIZES, MODELS, paired_bootstrap, write_json, sha

HERE = Path(__file__).resolve().parent

def main():
    x = Experiment(SimpleNamespace(payload=HERE/'payload', out=HERE/'out/verification',
            images=Path.home()/'lct-reid/data/images', device='cpu', threads=2,
            batch_size=8, gpu_id=0))
    report = {'status': 'running', 'geometry': x.audit_geometry(), 'checks': {}}
    selection = x.selected(True)
    assert sum(map(len, selection.values())) == 64
    oldsize = x.pp.INPUT_SIZE
    exact = 0
    for sp, indices in selection.items():
        for i in indices:
            r = x.rows[sp][i]
            with Image.open(x.pp.resolve_image_path(x.args.images, r.image_id)) as im:
                crop = np.asarray(im.convert('RGB').crop((r.x, r.y, r.x+r.w, r.y+r.h)))
            assert np.array_equal(x.tensor(sp, i, 208), x.plate.to_input(crop))
            exact += 1
    assert x.pp.INPUT_SIZE == oldsize == 208
    report['checks']['canonical_preprocess_208_bit_exact_crops'] = exact
    # Raster checks at every size: native white rectangle -> canonical resize;
    # nonzero support must remain at the geometrically predicted position.
    allitems = [(sp,i) for sp,rows in x.rows.items() for i in range(len(rows)) if x.boxes[sp][i]]
    chosen = allitems[:8] + [('val_query', 915), ('val_query', 976)]
    for fn in [lambda z: x.rows[z[0]][z[1]].w, lambda z: x.rows[z[0]][z[1]].h]:
        chosen += [min(allitems, key=fn), max(allitems, key=fn)]
    chosen = sorted(set(chosen))
    raster = []
    for sp, i in chosen:
        r = x.rows[sp][i]; bx, by, bw, bh = x.boxes[sp][i][:4]
        arr = np.zeros((r.h, r.w, 3), np.uint8); arr[by:by+bh, bx:bx+bw] = 255
        for size in SIZES:
            with x.input_size(size): out=x.pp.crop_to_input(Image.fromarray(arr),0,0,r.w,r.h)
            yy, xx = np.nonzero(out[0] > 0)
            assert len(xx)
            actual = np.array([xx.min(), yy.min(), xx.max()+1, yy.max()+1])
            expected = np.array([bx/r.w,by/r.h,(bx+bw)/r.w,(by+bh)/r.h])*size
            err=float(np.abs(actual-expected).max())
            assert err <= 2.01, (sp,i,size,actual,expected)
            raster.append({'split':sp,'index':i,'size':size,'max_boundary_error_pixels':err})
        # Every intervention is nonempty, and the masked path shares native crop/resize geometry.
        base = x.tensor(sp, i, 208)
        for v in ['plate_ring','shift_ring','platepad_ring','shiftpad_ring','plate_gray127','shift_gray127']:
            masked=x.tensor(sp,i,208,v)
            assert masked.shape == base.shape and np.isfinite(masked).all()
            assert not np.array_equal(masked,base), (sp,i,v)
    report['checks']['raster_geometry'] = raster
    a=np.array([.2,.4,.8]); own=paired_bootstrap(a,a)
    assert own['delta']==0 and own['ci95']==[0,0] and own['p_two_sided']==1
    delta=paired_bootstrap(a+.1,a)
    assert np.allclose(delta['ci95'],[.1,.1]) and 0 < delta['p_two_sided'] <= 1
    report['checks']['bootstrap_zero_and_constant_delta'] = True
    try:
        x.compare({'indices':np.array([1,2])},{'indices':np.array([2,1])})
    except RuntimeError:
        report['checks']['rejects_unpaired_order'] = True
    else: raise AssertionError('Permuted queries accepted')
    original=x.paths['combined_v1'].with_name('osnet_ain_combined_v1.onnx')
    om=onnx.load(original); dm=onnx.load(x.paths['combined_v1'])
    for i, name in [(2,'height'),(3,'width')]:
        d=om.graph.input[0].type.tensor_type.shape.dim[i];d.ClearField('dim_value');d.dim_param=name
    assert om.SerializeToString()==dm.SerializeToString()
    onnx.checker.check_model(dm)
    report['checks']['only_two_input_dimensions_changed'] = True
    report['model_probe'] = {}
    items=[(sp,i) for sp,idx in selection.items() for i in idx][:3]
    for name in MODELS:
        session=x.session(name); inp=session.get_inputs()[0].name
        rec={}
        for size in SIZES:
            inpdata=x.tensor_batch(items,size)
            y=session.run(None,{inp:inpdata})[0]
            assert y.shape==(3,512) and np.isfinite(y).all()
            rec[str(size)]={'batch':3,'shape':list(y.shape),'finite':True}
            if name=='combined_v1':
                orig=x.session(name,original=True)
                if size==208:
                    raw=orig.run(None,{orig.get_inputs()[0].name:inpdata})[0]
                    rawdiff=float(abs(raw-y).max())
                    diff=float(abs(x.model.l2norm(raw)-x.model.l2norm(y)).max())
                    assert diff < 1e-5 and rawdiff < .001
                    rec[str(size)]['original_dynamic_raw_max_abs_diff']=rawdiff
                    rec[str(size)]['original_dynamic_l2_max_abs_diff']=diff
                else:
                    try: orig.run(None,{orig.get_inputs()[0].name:inpdata})
                    except Exception as e: rec[str(size)]['original_rejection']=str(e)
                    else: raise AssertionError('Expected fixed original input to reject larger size')
        report['model_probe'][name]=rec
    report['code_sha256']=sha(HERE/'bigres.py')
    report['status']='passed'
    write_json(x.out/'verification.json',report)
    print(json.dumps({'status':report['status'],'preprocess_crops':exact,'raster_cases':len(raster),
                      'models':report['model_probe']},indent=2))

if __name__=='__main__':main()
