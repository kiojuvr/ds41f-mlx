"""Integer Engram semantics: independent signed-int64 XOR/modulo and chunk state.

Synthetic token-map input isolates hashing, not tokenizer-normalization or model math.
Official V41 geometry is held in the contract; no floating tensor goldens.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from omlx.patches.deepseek_v41.engram import NgramHash


def signed(value):
    return (int(value)+(1<<63)) % (1<<64) - (1<<63)


def test_integer_hash_and_incremental_state():
    c=SimpleNamespace(**json.loads((Path(__file__).parents[1]/'fixtures/engram-geometry.json').read_text()))
    mapping=np.arange(c.engram_compressed_vocab_size,dtype=np.int64)
    h=NgramHash(c,mapping)
    bound=((1<<63)-1)//c.engram_compressed_vocab_size//2
    for layer,layer_id in enumerate(c.engram_layer_ids):
        rng=np.random.default_rng(10007*layer_id)
        assert np.array_equal(h.multipliers[layer],rng.integers(0,bound,c.engram_max_ngram_size,dtype=np.int64)*2+1)
    # Prime buckets are the first globally unused primes at/above the declared size.
    import math
    used=set(); expected_primes=[]
    for _ in c.engram_layer_ids:
        groups=[]
        for _ in range(c.engram_max_ngram_size-1):
            current=c.engram_vocab_size-1;group=[]
            for _ in range(c.engram_n_heads):
                current+=1
                while current in used or any(current%d==0 for d in range(2,math.isqrt(current)+1)):
                    current+=1
                used.add(current);group.append(current)
            groups.append(group)
        expected_primes.append(groups)
    assert np.array_equal(h.primes,np.array(expected_primes))
    ids=np.array([[0,3,15,99091,8,2,99]],dtype=np.int64)
    actual,history=h(ids)
    expected=np.zeros(actual.shape,dtype=np.int64)
    # Independent scalar arithmetic: Python unbounded ints, explicit signed wrap.
    for position in range(ids.shape[1]):
        for layer in range(len(c.engram_layer_ids)):
            rolling=signed(int(ids[0,position])*int(h.multipliers[layer,0]))
            column=0
            for shift in range(1,c.engram_max_ngram_size):
                token=int(ids[0,position-shift]) if position>=shift else h.pad_id
                rolling=signed(rolling ^ signed(token*int(h.multipliers[layer,shift])))
                for head in range(c.engram_n_heads):
                    expected[0,position,layer,column]=rolling % int(h.primes[layer,shift-1,head])+int(h.offsets[layer,column])
                    column+=1
    assert np.array_equal(actual,expected)
    assert np.array_equal(history,ids[:,-3:])
    before=history.copy();left,state=h(ids[:,:2]);right,end=h(ids[:,2:],state)
    assert np.array_equal(np.concatenate([left,right],axis=1),actual)
    assert np.array_equal(end,history) and np.array_equal(before,history)
    # Request histories cannot cross-contaminate fresh timelines.
    fresh,_=h(ids[:,2:]); assert not np.array_equal(fresh[:,:2],right[:,:2])
