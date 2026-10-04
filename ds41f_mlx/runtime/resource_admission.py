"""Fail-closed OFF execution resource admission, not an installer or trust service.

The repository pin describes qualified content, never paths or package versions
alone. Full checkpoint payloads are checked once before loading. Local operators
must keep admitted resources static for the model lifetime (no hostile-host or
concurrent replacement protection). No token-loop hashing, conversion or cache.
"""
from dataclasses import asdict, dataclass, field
import ast
from hashlib import sha256
from pathlib import Path
import ctypes
import ctypes.util
import fcntl
import importlib
import importlib.metadata
import json
import marshal
import os
import sys
import struct
import types
import weakref

PIN = Path(__file__).with_name('admitted_resources.json')


class ResourceAdmissionError(RuntimeError):
    pass


def fingerprint(path):
    path = Path(path)
    before = file_token(path.stat())
    digest = sha256()
    with path.open('rb') as reader:
        fcntl.fcntl(reader.fileno(), fcntl.F_NOCACHE, 1)
        while data := reader.read(8 * 1024 * 1024):
            digest.update(data)
    if file_token(path.stat()) != before:
        raise ResourceAdmissionError('resource changed during verification')
    return {'size': before[2], 'sha256': digest.hexdigest()}, before


def file_token(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def require(condition, message):
    if not condition:
        raise ResourceAdmissionError(message)


def check_file(path, expected, label, verified=None):
    try:
        actual, token = fingerprint(path)
    except (OSError, ValueError) as exc:
        raise ResourceAdmissionError(f'missing/unreadable {label}') from exc
    require(actual == expected, f'identity mismatch: {label}')
    if verified is not None:
        verified[Path(path).resolve()] = token
    return token


def _codes(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _codes(constant)


def _normalized(code):
    return code.replace(co_filename='', co_consts=tuple(
        _normalized(v) if isinstance(v, types.CodeType) else v for v in code.co_consts))


def check_module(name, expected, verified=None):
    """Check actual imported source and executable bytecode, not distribution labels."""
    module = importlib.import_module(name)
    spec = getattr(module, '__spec__', None)
    require(spec is not None and spec.origin == getattr(module, '__file__', None),
            f'ambiguous module origin: {name}')
    path = Path(spec.origin).resolve()
    check_file(path, expected, name, verified)
    if path.suffix == '.py':
        require(isinstance(spec.loader, importlib.machinery.SourceFileLoader),
                f'unsupported module loader: {name}')
        compiled = compile(path.read_bytes(), str(path), 'exec', optimize=sys.flags.optimize)
        cached = getattr(module, '__cached__', None)
        if cached and Path(cached).exists():
            try:
                bytecode = marshal.loads(Path(cached).read_bytes()[16:])
                require(_normalized(compiled) == _normalized(bytecode), f'stale bytecode: {name}')
            except (EOFError, ValueError, TypeError) as exc:
                raise ResourceAdmissionError(f'invalid bytecode: {name}') from exc
        compiled_codes = tuple(_normalized(c) for c in _codes(compiled))
        known = {c.co_qualname: c for c in compiled_codes}
        for definition in ast.parse(path.read_bytes()).body:
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                value = vars(module).get(definition.name)
                require(value is not None and getattr(value, '__module__', None) == name,
                        f'substituted module definition: {name}.{definition.name}')
                if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    while hasattr(value, '__wrapped__'):
                        value = value.__wrapped__
                    code = getattr(value, '__code__', None)
                    require(isinstance(code, types.CodeType)
                            and known.get(definition.name) == _normalized(code),
                            f'substituted executable definition: {name}.{definition.name}')
                elif isinstance(definition, ast.ClassDef):
                    for method in definition.body:
                        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        implementation = vars(value).get(method.name)
                        if isinstance(implementation, property):
                            selector = next((d.attr for d in method.decorator_list
                                             if isinstance(d, ast.Attribute) and d.attr in ('setter','deleter')), None)
                            implementation = (implementation.fset if selector == 'setter' else
                                              implementation.fdel if selector == 'deleter' else implementation.fget)
                        if isinstance(implementation, (classmethod, staticmethod)):
                            implementation = implementation.__func__
                        while hasattr(implementation, '__wrapped__'):
                            implementation = implementation.__wrapped__
                        code = getattr(implementation, '__code__', None)
                        require(isinstance(code, types.CodeType) and _normalized(code) in compiled_codes,
                                f'substituted class primitive: {name}.{definition.name}.{method.name}')
        # Decorated MLX functions expose __code__; classes are checked without
        # descriptor evaluation. Imported aliases are checked in their own module.
        values = list(vars(module).values())
        for value in list(values):
            if isinstance(value, type) and value.__module__ == name:
                values.extend(vars(value).values())
        for value in values:
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            code = getattr(value, '__code__', None)
            if isinstance(code, types.CodeType) and getattr(value, '__module__', None) == name:
                # functools caches expose the original implementation.
                while hasattr(value, '__wrapped__'):
                    value = value.__wrapped__
                code = getattr(value, '__code__', code)
                if code.co_qualname not in known and code.co_filename != str(path):
                    continue  # Imported aliases; checked at their implementation origin.
                if code.co_filename == '<string>' and code.co_qualname.startswith('__create_fn__.'):
                    continue  # stdlib dataclass-generated methods; field definitions are pinned.
                require(known.get(code.co_qualname) == _normalized(code),
                        f'live numerical code mismatch: {name}.{code.co_qualname}')
    return module


def loaded_images():
    require(sys.platform == 'darwin', 'unqualified native platform')
    system = ctypes.CDLL(ctypes.util.find_library('System'))
    system._dyld_image_count.restype = ctypes.c_uint32
    system._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    system._dyld_get_image_name.restype = ctypes.c_char_p
    return [Path(system._dyld_get_image_name(i).decode()).resolve()
            for i in range(system._dyld_image_count())]


def _macho_uuid(data):
    require(len(data) >= 32 and struct.unpack_from('<I', data)[0] == 0xfeedfacf,
            'unqualified native image format')
    count = struct.unpack_from('<I', data, 16)[0]
    offset = 32
    for _ in range(count):
        command, length = struct.unpack_from('<II', data, offset)
        require(length >= 8 and offset + length <= len(data), 'invalid native image commands')
        if command == 0x1b:
            return data[offset+8:offset+24]
        offset += length
    raise ResourceAdmissionError('native image has no build UUID')


def check_loaded_image(path):
    """Reject an old in-memory build even if its pathname now holds a new binary."""
    path = Path(path).resolve()
    system = ctypes.CDLL(ctypes.util.find_library('System'))
    system._dyld_image_count.restype = ctypes.c_uint32
    system._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    system._dyld_get_image_name.restype = ctypes.c_char_p
    system._dyld_get_image_header.argtypes = [ctypes.c_uint32]
    system._dyld_get_image_header.restype = ctypes.c_void_p
    matching = []
    for i in range(system._dyld_image_count()):
        if Path(system._dyld_get_image_name(i).decode()).resolve() == path:
            pointer = system._dyld_get_image_header(i)
            header = ctypes.string_at(pointer, 32)
            length = struct.unpack_from('<I', header, 20)[0]
            require(length < 16*1024*1024, 'invalid loaded native header')
            matching.append(_macho_uuid(ctypes.string_at(pointer, 32+length)))
    with path.open('rb') as reader:
        header = reader.read(32)
        length = struct.unpack_from('<I', header, 20)[0]
        disk = _macho_uuid(header+reader.read(length))
    require(matching == [disk], 'missing/stale/mixed loaded native image UUID')


def check_native(modules, pin, verified=None):
    core = Path(modules['mlx.core'].__file__).resolve()
    mlx_root = core.parent
    for name, identity in pin['mlx_resources'].items():
        path = mlx_root / name
        check_file(path, identity, f'mlx/{name}', verified)
        if name.endswith('.dylib'):
            check_loaded_image(path)
            images = [p for p in loaded_images() if p.name == path.name]
            require(images == [path.resolve()], f'mixed/redirected native image: {name}')
    fast = modules['omlx.custom_kernels.glm_moe_dsa.fast']
    directory = Path(fast.__file__).resolve().parent
    found = {p.name for p in directory.iterdir() if p.suffix in ('.so', '.dylib', '.metallib')}
    profile = 'native' if fast._ext is not None else 'portable'
    expected = pin['glm_profiles'][profile]
    require(found == set(expected), 'missing/stale/mixed GLM native bundle')
    for name, identity in expected.items():
        check_file(directory / name, identity, f'glm/{name}', verified)
    if profile == 'native':
        check_loaded_image(fast._ext.__file__)
        check_loaded_image(directory/'libomlx_glm_kernel_ops.dylib')
        require(Path(fast._ext.__file__).resolve() == (directory / pin['glm_extension']).resolve(),
                'redirected GLM extension')
        path = directory / 'libomlx_glm_kernel_ops.dylib'
        require([p for p in loaded_images() if p.name == path.name] == [path.resolve()],
                'mixed/redirected GLM dylib')
    return profile


def check_dispatch(pin):
    switch = importlib.import_module('omlx.patches.deepseek_v4.switch_layers')
    for name, expected in pin['dispatch'].items():
        require(getattr(switch, name) == expected, f'unqualified numerical dispatch: {name}')
    # These flags are queried dynamically by the retained projection path.
    for name, expected in pin['environment'].items():
        require(os.environ.get(name, expected) == expected, f'unqualified execution override: {name}')


@dataclass
class AdmittedResources:
    """A model-lifetime capability. Inspection is pathless; handles stay private."""
    identity: str
    profile: str
    modules: dict = field(repr=False)
    files: dict = field(repr=False)
    pin: dict = field(repr=False)
    active: bool = True
    _model: object = field(default=None, repr=False)
    _binding: tuple = field(default=(), repr=False)
    _engram: tuple = field(default=(), repr=False)
    _config: dict = field(default_factory=dict, repr=False)
    _hash_identity: tuple = field(default=(), repr=False)
    _device: str = ''
    _locations: dict = field(default_factory=dict, repr=False)

    @property
    def math(self):
        return self.modules['omlx.patches.deepseek_v41.language']

    @property
    def cache_type(self):
        return self.modules['omlx.patches.deepseek_v41.cache'].DeepseekV41Cache

    def assert_active(self):
        require(self.active, 'execution resource lease retired')

    def require_backend(self, mx):
        require(mx is self.modules['mlx.core'] and str(mx.default_device()) == self._device,
                'unadmitted tensor backend/device handle')

    def check_files(self):
        for location, resolved in self._locations.items():
            require(location.resolve() == resolved, 'admitted resource locator redirected')
        for path, token in self.files.items():
            try:
                require(file_token(path.stat()) == token, 'admitted resource changed/replaced')
            except OSError as exc:
                raise ResourceAdmissionError('admitted resource disappeared') from exc

    def binding(self, model):
        return (tuple((name, id(module), type(module)) for name, module in model.named_modules()),
                tuple((name, id(value)) for name, value in self.modules['mlx.utils'].tree_flatten(model.parameters())),
                id(model._config), id(model._hasher))

    @staticmethod
    def storage_identity(embed, reader):
        return (embed._weight_key, embed._scale_key, embed._bias_key, embed._bits,
                embed._group_size, json.dumps(reader.header, sort_keys=True), reader._start,
                reader._file_size, id(reader._mapping), id(reader._file))

    @staticmethod
    def hash_identity(hasher):
        return tuple((name, value.shape, str(value.dtype), sha256(value.tobytes()).hexdigest())
                     for name in ('token_map', 'primes', 'multipliers', 'offsets')
                     for value in (getattr(hasher, name),))

    def bind_model(self, model, processor):
        self.assert_active(); self.check_files()
        language = getattr(model, 'language_model', model)
        require(type(language) is self.math.LanguageModel, 'unadmitted model-module type')
        require(language._config.n_layers == 40 and not language._config.preserve_mtp,
                'unqualified model configuration')
        for _, module in language.named_modules():
            cls = type(module)
            origin = self.modules.get(cls.__module__)
            require(origin is not None and vars(origin).get(cls.__name__) is cls,
                    f'unadmitted numerical module class: {cls.__module__}.{cls.__name__}')
        require(sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
                == self.pin['tokenizer_serialization_sha256'],
                'loaded tokenizer mismatch')
        storage = self.modules['omlx.patches.deepseek_v41.storage']
        source = self.modules['omlx.patches.deepseek_v41.convert']
        root = self._checkpoint
        raw = json.loads((root/'config.json').read_text())
        mapping = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        expected_config = self.modules['omlx.patches.deepseek_v41.config'].ModelConfig.from_dict(raw)
        expected_config.preserve_mtp = False
        require(asdict(language._config) == asdict(expected_config), 'loaded configuration differs from admitted checkpoint')
        tables = source.source_engram_tables(mapping, raw)
        readers = []
        for module, spec in tables.items():
            ix = int(module.split('.')[2])
            embed = language.layers[ix].engram.embed
            require(type(embed) is storage.DiskEngramEmbedding and not embed._closed
                    and embed._resident is None, 'unadmitted SSD Engram store/lifecycle')
            for attr, key in (('_weight_key', 'weight_key'), ('_scale_key', 'scale_key'),
                              ('_bias_key', 'bias_key')):
                require(getattr(embed, attr) == spec.get(key), 'Engram tensor/index mismatch')
            for attr, filename in (('_weights', spec['weight_file']),
                                   ('_scales', spec.get('scale_file') or spec['weight_file'])):
                reader = getattr(embed, attr)
                token = self.files[(root/filename).resolve()]
                require(file_token(os.fstat(reader._file.fileno())) == token,
                        'Engram descriptor does not own admitted backing file')
                metadata = self.storage_identity(embed, reader)
                readers.append((embed, reader, token, metadata, attr))
        require(len(tables) == len(language._config.engram_layer_ids) == 2, 'missing Engram resources')
        require(type(language._engram_prefetch) is storage.EngramPrefetch
                and language._engram_prefetch._pending is None, 'unadmitted Engram read scope')
        self._engram = tuple(readers)
        self._model = weakref.ref(language)
        self._binding = self.binding(language)
        self._config = asdict(language._config)
        require(type(language._hasher) is self.math.NgramHash, 'unadmitted Engram hash implementation')
        self._hash_identity = self.hash_identity(language._hasher)
        object.__setattr__(language, '_ds41f_resources', self)

    def validate_binding(self, model):
        self.assert_active(); self.check_files()
        if self._device:
            self.require_backend(self.modules['mlx.core'])
        require(self._model is not None and self._model() is model
                and self.binding(model) == self._binding and asdict(model._config) == self._config,
                'loaded model binding changed')
        if self._hash_identity:
            require(self.hash_identity(model._hasher) == self._hash_identity, 'Engram hash resource changed')
        for embed, reader, token, metadata, attr in self._engram:
            try:
                current = file_token(os.fstat(reader._file.fileno()))
            except (OSError, ValueError) as exc:
                raise ResourceAdmissionError('SSD Engram descriptor retired') from exc
            require(getattr(embed, attr) is reader and not embed._closed
                    and embed._resident is None and current == token
                    and metadata == self.storage_identity(embed, reader),
                    'SSD Engram resource retired/replaced')
        for i in model._config.engram_layer_ids:
            require(any(model.layers[i].engram.embed is embed for embed, _, _, _, _ in self._engram),
                    'SSD Engram module substituted')

    def describe(self):
        return {'schema': 'ds41f.execution-admission.v1', 'resource_set_sha256': self.identity,
                'native_profile': self.profile, 'active': self.active, 'model_bound': self._model is not None,
                'checkpoint_files': self.pin['checkpoint'],
                'numerical_modules': self.pin['modules'],
                'native_artifacts': {'mlx': self.pin['mlx_resources'],
                                     'glm': self.pin['glm_profiles'][self.profile]},
                'protocol_tokenizer': self.pin['protocol_tokenizer'],
                'python_abi': self.pin['python_abi'], 'gpu_architecture': self.pin['gpu_architecture'],
                'dependency_versions': self.pin['versions'],
                'ssd_engram': 'admitted checkpoint descriptors'}

    def retire(self):
        self.active = False


def resources_for(model):
    resources = getattr(model, '_ds41f_resources', None)
    require(type(resources) is AdmittedResources, 'model lacks ds41f resource admission')
    resources.validate_binding(model)
    return resources


def load_protocol_tokenizer(recipe_path):
    """Construct the OFF protocol handle from verified bytes, not an ambient path."""
    pin = json.loads(PIN.read_bytes())
    for name in ('deepseek_recipe', 'deepseek_recipe._native'):
        check_module(name, pin['modules'][name])
    path = Path(recipe_path)/'static/tokenizers/v41/tokenizer.json'
    data = path.read_bytes()
    require({'size': len(data), 'sha256': sha256(data).hexdigest()} == pin['protocol_tokenizer'],
            'protocol tokenizer identity mismatch')
    return importlib.import_module('deepseek_recipe').Tokenizer.from_str(data.decode('utf-8'))


def prepare_resources(checkpoint, recipe_path):
    """Verify the qualified implementation and all checkpoint bytes before allocation."""
    pin_bytes = PIN.read_bytes()
    pin = json.loads(pin_bytes)
    require(sys.implementation.cache_tag == pin['python_abi'], 'unqualified Python ABI')
    files = {}
    modules = {name: check_module(name, identity, files) for name, identity in pin['modules'].items()}
    for package, expected in pin['versions'].items():
        require(importlib.metadata.version(package) == expected, f'dependency version mismatch: {package}')
    for package, resources in pin['package_resources'].items():
        root = Path(importlib.import_module(package).__file__).resolve().parent
        for name, identity in resources.items():
            check_file(root/name, identity, f'{package}/{name}', files)
    profile = check_native(modules, pin, files)
    for module in modules.values():
        if Path(module.__file__).suffix == '.so':
            check_loaded_image(module.__file__)
    check_dispatch(pin)
    require(modules['mlx.core'].device_info()['architecture'] == pin['gpu_architecture'],
            'unqualified numerical device architecture')
    recipe_tokenizer = (Path(recipe_path)/'static/tokenizers/v41/tokenizer.json').resolve(strict=True)
    recipe_token = check_file(recipe_tokenizer, pin['protocol_tokenizer'], 'protocol tokenizer')
    checkpoint = Path(checkpoint).resolve(strict=True)
    require(not (checkpoint/'conversion.inprogress.json').exists(), 'checkpoint conversion in progress')
    files[recipe_tokenizer] = recipe_token
    # Small identity/layout files fail before the expensive payload scan.
    names = sorted(pin['checkpoint'], key=lambda n: n.endswith('.safetensors'))
    locations = {}
    for name in names:
        location = checkpoint/name
        path = location.resolve(strict=True)
        locations[location] = path
        files[path] = check_file(path, pin['checkpoint'][name], f'checkpoint/{name}')
    admission = AdmittedResources(sha256(pin_bytes).hexdigest(), profile, modules, files, pin)
    admission._checkpoint = checkpoint
    for module in modules.values():
        path = Path(module.__file__)
        locations[path] = path.resolve()
    locations[Path(recipe_path)/'static/tokenizers/v41/tokenizer.json'] = recipe_tokenizer
    admission._locations = locations
    require(modules['mlx.core'].default_device().type == modules['mlx.core'].gpu,
            'production admission requires GPU backend')
    admission._device = str(modules['mlx.core'].default_device())
    return admission
