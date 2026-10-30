"""Finite Processor-only diagnostic; not Runtime admission or native qualification.
Run one geometry per fresh process with the isolated environment and PYTHONPATH=.
"""
import gc
from io import BytesIO
import json
import os
import resource
import subprocess
import sys
import tracemalloc
import weakref

from PIL import Image
from ds41f_mlx.model_execution import processing
from ds41f_mlx.model_execution.config import ModelConfig

width, height = map(int, sys.argv[1:])
assert (width, height) in ((512, 512), (2600, 2000))
with Image.new('RGBA', (width, height)) as fixture:
    out = BytesIO()
    fixture.save(out, format='PNG')
data = out.getvalue()
del fixture, out
gc.collect()
tracemalloc.start()
events = []
refs = {}
def sample(stage):
    events.append(dict(stage=stage, rss_bytes=int(subprocess.check_output(
        ['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True).strip()) * 1024,
        maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python_numpy_traced_bytes=tracemalloc.get_traced_memory(),
        rgb_object_live=refs.get('rgb', lambda: None)() is not None))
convert, resize, asarray = Image.Image.convert, Image.Image.resize, processing.np.asarray
sample('baseline_after_fixture')
with Image.open(BytesIO(data)) as source:
    def conversion(image, *args, **kwargs):
        image.load()
        sample('source_loaded_before_rgb')
        result = convert(image, *args, **kwargs)
        refs['rgb'] = weakref.ref(result)
        sample('full_source_and_rgb_live')
        return result
    def resizing(image, *args, **kwargs):
        result = resize(image, *args, **kwargs)
        sample('source_rgb_and_resized_live')
        return result
    def normalization(image, *args, **kwargs):
        sample('before_normalization_source_still_live')
        return asarray(image, *args, **kwargs)
    Image.Image.convert, Image.Image.resize, processing.np.asarray = conversion, resizing, normalization
    patches, h, w, kinds = processing.image_patch_array(source, ModelConfig(vision_n_layers=1))
    sample('patches_returned_source_still_live')
    shape, patch_bytes = patches.shape, patches.nbytes
# __exit__ closes file, not the decoded core; explicit owner release below.
source.close()
del source, patches
gc.collect()
sample('all_cpu_inputs_released')
print(json.dumps(dict(source=[width,height],encoded_bytes=len(data),
    source_core_nominal_bytes=width*height*4, rgb_core_nominal_bytes=width*height*4,
    note='Pillow RGB core uses 4 bytes/pixel; RSS/tracemalloc are observations, not a reservation or complete allocation trace',
    grid=[h,w],expanded_rows=len(kinds),patch_shape=shape,patch_bytes=patch_bytes,events=events),indent=2))
