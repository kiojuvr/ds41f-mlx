#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/m4/actual-layer2-capture/result.json'
def main():
 r=json.loads(ART.read_text())
 assert r['schema']=='ds41f.m4.actual-layer2-capture.v1'
 assert r['real_loaded_omlx'] is True
 assert r['attempt_result']['exit_code']==137
 assert r['layer2_incremental_block']=='INCOMPLETE'
 print(f'Actual Layer2 capture attempt recorded (incomplete): {ART}')
if __name__=='__main__': main()
