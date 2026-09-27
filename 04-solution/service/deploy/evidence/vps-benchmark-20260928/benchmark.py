#!/usr/bin/env python3
"""Read-only, bounded LCT HTTP search probe for the released demo images."""

import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request


BASE = sys.argv[1].rstrip("/")
PINNED_IPV4 = os.environ.get("LCT_PINNED_IPV4")
if PINNED_IPV4:
    original_create_connection = socket.create_connection

    def pinned_create_connection(address, *args, **kwargs):
        if address == ("iamcp.ru", 443):
            address = (PINNED_IPV4, 443)
        return original_create_connection(address, *args, **kwargs)

    socket.create_connection = pinned_create_connection
SEQUENTIAL = 10
CONCURRENT = 8
WARMUP = 2


def request(path, body=None, content_type=None):
    headers = {"Content-Type": content_type} if content_type else {}
    req = urllib.request.Request(BASE + path, data=body, headers=headers)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = response.read()
            return response.status, payload, time.perf_counter() - start
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.perf_counter() - start


def multipart(image, box):
    boundary = "LctReadOnlyBenchmark20260928"
    values = {**box, "top_k": 10}
    parts = []
    for key, value in values.items():
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\""
                      f"\r\n\r\n{value}\r\n").encode())
    parts.extend([
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         "filename=\"demo.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode(),
        image,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def nearest_rank(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(len(values) * fraction) - 1]


health_code, health_bytes, health_s = request("/api/health")
demo_code, demo_bytes, demo_s = request("/api/demo")
version_code, version_bytes, version_s = request("/api/version")
assert (health_code, demo_code, version_code) == (200, 200, 200)
health = json.loads(health_bytes)
assert health["status"] == "ok" and health["model_loaded"] and health["storage"]["gallery_points"] == 750
examples = json.loads(demo_bytes)["examples"]
assert len(examples) >= 2
inputs = []
for example in examples[:2]:
    code, image, seconds = request(example["image_url"])
    assert code == 200
    body, content_type = multipart(image, example["bbox"])
    inputs.append((example["id"], body, content_type, hashlib.sha256(image).hexdigest(), len(image)))


def search(index):
    example_id, body, content_type, _, _ = inputs[index % len(inputs)]
    code, payload, seconds = request("/api/search", body, content_type)
    sample = {"example": example_id, "http_status": code, "seconds": seconds}
    if code == 200:
        response = json.loads(payload)
        sample["refusal"] = response["refusal"]
        sample["candidates"] = len(response["candidates"])
        assert response["refusal"] == (not response["candidates"])
    return sample


for i in range(WARMUP):
    sample = search(i)
    assert sample["http_status"] == 200

phases = {}
for concurrency, count in ((1, SEQUENTIAL), (2, CONCURRENT)):
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(search, range(count)))
    wall = time.perf_counter() - start
    durations = [sample["seconds"] for sample in samples]
    phases[str(concurrency)] = {
        "concurrency": concurrency,
        "count": count,
        "elapsed_s": wall,
        "throughput_rps": count / wall,
        "p50_s": statistics.median(durations),
        "p95_nearest_rank_s": nearest_rank(durations, 0.95),
        "http_codes": {str(code): sum(sample["http_status"] == code for sample in samples)
                       for code in sorted({sample["http_status"] for sample in samples})},
        "samples": samples,
    }

report = {
    "status": "PASS" if all(phase["http_codes"] == {"200": phase["count"]}
                            for phase in phases.values()) else "FAIL",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "base": BASE,
    "pinned_ipv4": PINNED_IPV4,
    "host": platform.platform(),
    "python": platform.python_version(),
    "health": {"http_status": health_code, "seconds": health_s, "gallery_points": 750},
    "demo_http_status": demo_code,
    "version_http_status": version_code,
    "version_model": json.loads(version_bytes)["model"],
    "version_sha256": {key: json.loads(version_bytes)[key] for key in
                       ("model_sha256", "model2_sha256", "whitening_sha256")},
    "inputs": [{"id": item[0], "sha256": item[3], "size_bytes": item[4]} for item in inputs],
    "warmup": WARMUP,
    "phases": phases,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
