"""Sequential foreground runner; no worker pools or background processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("phase",choices=["small","large","rerank","sensitivity","precision"])
p.add_argument("--python",default=sys.executable)
a=p.parse_args()
steps=[]
if a.phase=="small":
    steps=[("env",750,.08)]
    for n in [750,7500,75000]:
        steps += [(mode,n,.08) for mode in ["flat","current","ivf"]]
elif a.phase=="large":
    steps=[(mode,1000000,.08) for mode in ["flat","current","ivf"]]
elif a.phase=="rerank":
    steps=[("rerank-scale",n,.08) for n in [750,1500,3000,6000,7500,10000,1000000]]
    steps += [("quality",750,.08)]
    steps += [("candidate-cost",k,.08) for k in [30,100,300]]
elif a.phase=="sensitivity":
    steps=[(mode,75000,.25) for mode in ["flat","ivf"]]
else:
    steps=[("truth64",n,.08) for n in [75000,1000000]]

(ROOT/"logs").mkdir(exist_ok=True)
for mode,n,sigma in steps:
    mem=subprocess.check_output(["free","-m"],text=True)
    print(mem,flush=True)
    avail=int(mem.splitlines()[1].split()[-1])
    if avail<2500:
        raise SystemExit(f"Memory threshold reached: {avail} MiB")
    cmd=["systemd-run","--user","--scope","--quiet","-p","MemoryMax=4G",
         "-p","MemoryHigh=3500M","-p","MemorySwapMax=0","-p","CPUQuota=100%",
         "taskset","-c","1","nice","-n","10",a.python,"-B",str(ROOT/"bench.py"),
         mode,"--n",str(n),"--sigma",str(sigma)]
    print(time.strftime("%FT%T%z")," ".join(cmd),flush=True)
    with (ROOT/"logs"/f"{mode}_{n}_{sigma}.log").open("w") as log:
        done=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if done.returncode:
        print((ROOT/"logs"/f"{mode}_{n}_{sigma}.log").read_text()[-5000:],flush=True)
        raise SystemExit(done.returncode)
    data=json.loads((ROOT/"results"/f"{mode}_{n}_{sigma}.json").read_text())
    print(mode,n,data["status"],flush=True)
    if data["status"] not in ("ok","skipped_budget"):
        raise SystemExit("Memory guard stopped the phase safely")
