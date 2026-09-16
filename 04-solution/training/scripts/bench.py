"""Замер реальной пропускной способности карты на шаге обучения ResNet-*."""
import json, time, sys
import torch, torchvision

torch.backends.cudnn.benchmark = True
res = []

def bench(arch, size, bs, last_stride, iters=12):
    m = getattr(torchvision.models, arch)(weights=None)
    if last_stride == 1:
        m.layer4[0].conv2.stride = (1, 1)
        m.layer4[0].downsample[0].stride = (1, 1)
    m.fc = torch.nn.Linear(m.fc.in_features, 1171)
    m = m.cuda().train()
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(bs, 3, size, size, device="cuda")
    y = torch.randint(0, 1171, (bs,), device="cuda")
    lf = torch.nn.CrossEntropyLoss()
    try:
        for _ in range(3):
            opt.zero_grad(set_to_none=True); lf(m(x), y).backward(); opt.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            opt.zero_grad(set_to_none=True); lf(m(x), y).backward(); opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        r = {"arch": arch, "size": size, "bs": bs, "last_stride": last_stride,
             "step_ms": round(dt * 1e3, 1), "img_per_s": round(bs / dt, 1),
             "epoch_s_7248": round(7248 / (bs / dt), 1),
             "peak_MiB": round(torch.cuda.max_memory_reserved() / 2**20)}
    except RuntimeError as e:
        r = {"arch": arch, "size": size, "bs": bs, "last_stride": last_stride,
             "error": str(e)[:120]}
    del m, opt, x, y
    torch.cuda.empty_cache()
    print(json.dumps(r), flush=True)
    return r

for cfg in [("resnet50", 256, 32, 2), ("resnet50", 224, 32, 2), ("resnet50", 256, 32, 1),
            ("resnet50", 192, 48, 2), ("resnet34", 256, 48, 2), ("resnet18", 256, 64, 2),
            ("resnet18", 256, 64, 1)]:
    res.append(bench(*cfg))
json.dump(res, open(r"D:\lct-reid\bench.json", "w"), indent=1)
