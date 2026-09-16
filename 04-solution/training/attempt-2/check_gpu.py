import json, time
import torch, torchvision

info = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "is_available": torch.cuda.is_available(),
    "arch_list": torch.cuda.get_arch_list(),
}
if info["is_available"]:
    info["device_name"] = torch.cuda.get_device_name(0)
    info["capability"] = list(torch.cuda.get_device_capability(0))
    p = torch.cuda.get_device_properties(0)
    info["total_mem_MiB"] = round(p.total_memory / 2**20)
    info["multi_processor_count"] = p.multi_processor_count
    sm = "sm_%d%d" % tuple(info["capability"])
    info["arch_supported"] = sm in info["arch_list"]

    # реальная работа на карте: matmul + conv + backward
    torch.backends.cudnn.benchmark = True
    x = torch.randn(64, 3, 256, 256, device="cuda")
    conv = torch.nn.Conv2d(3, 64, 7, 2, 3).cuda()
    y = conv(x); y.sum().backward(); torch.cuda.synchronize()
    info["conv_fwd_bwd_ok"] = True
    info["conv_out_shape"] = list(y.shape)

    a = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    info["sgemm_2048_ms"] = round(dt * 1e3, 2)
    info["sgemm_TFLOPS"] = round(2 * 2048**3 / dt / 1e12, 3)
    info["mem_alloc_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20)
print(json.dumps(info, indent=1))
