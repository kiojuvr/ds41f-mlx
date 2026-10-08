"""Bounded, owned anonymous-memory co-resident fixture; no inference/GPU work."""
import argparse
import hashlib
import json
import mmap
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gib', type=float, required=True)
    args = ap.parse_args()
    if not 0 < args.gib <= 64:
        ap.error('pressure fixture bounded to (0,64] GiB')
    import psutil
    size = int(args.gib * 2**30)
    # Entropic pages avoid pretending a compressible zero mapping is pressure.
    block = os.urandom(min(size, 4*2**20))
    begin = time.monotonic()
    memory = mmap.mmap(-1, size)
    try:
        for offset in range(0, size, len(block)):
            memory[offset:min(size,offset+len(block))] = block[:min(len(block),size-offset)]
        print(json.dumps(dict(pid=os.getpid(), bytes=size,
            fixture='touched anonymous nonzero entropic pages; CPU idle while held; NOT severe OS-pressure qualification',
            block_sha256=hashlib.sha256(block).hexdigest(), allocation_s=time.monotonic()-begin,
            rss_bytes=psutil.Process().memory_info().rss,
            available_bytes=psutil.virtual_memory().available)), flush=True)
        sys.stdin.readline()  # Parent owns pipe; EOF also frees the fixture.
    finally:
        memory.close()


if __name__=='__main__': main()
