#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/m4/layer2-projection-block2/result.json'
def main():
 r=json.loads(ART.read_text())
 assert r['schema']=='ds41f.m4.layer2-projection-block2.v1'
 assert r['layer2_projection']['status']=='INDEPENDENT_SOURCE_DERIVED_COMPLETE'
 assert r['layer2_block']['status']=='INDEPENDENT_SOURCE_DERIVED_COMPLETE'
 assert r['actual_omlx_internal_capture']['status']=='INCOMPLETE'
 assert r['layer2_incremental_block']=='INCOMPLETE'
 print(f'Layer2 projection/block2 source-derived check PASS (actual capture incomplete): {ART}')
if __name__=='__main__': main()
