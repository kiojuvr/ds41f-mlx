"""Bounded non-owning reference descriptions. Never retain referents in receipts."""
import gc
import types
import weakref


def characterize(reference):
    target = reference()
    if target is None:
        return {'dead': True, 'referrers': []}
    rows = []
    referrers = gc.get_referrers(target)
    for owner in referrers[:24]:
        row = {'type': type(owner).__module__ + '.' + type(owner).__qualname__}
        if isinstance(owner, dict):
            row['keys_pointing_to_target'] = [str(k)[:80] for k, v in owner.items() if v is target][:12]
        elif isinstance(owner, types.CellType):
            # Cells are expected in registered application route/event closures.
            paths = []
            parents = gc.get_referrers(owner)
            for parent in parents[:24]:
                if isinstance(parent, tuple):
                    for function in gc.get_referrers(parent)[:24]:
                        if isinstance(function, types.FunctionType) and function.__closure__ is parent:
                            path = {'function': function.__module__ + '.' + function.__qualname__, 'owners': []}
                            for root in gc.get_referrers(function)[:24]:
                                item = {'type': type(root).__module__ + '.' + type(root).__qualname__}
                                if isinstance(root, dict):
                                    item['keys'] = [str(k)[:80] for k, v in root.items() if v is function][:8]
                                elif isinstance(root, tuple):
                                    item['tuple_owner_types'] = [type(x).__module__ + '.' + type(x).__qualname__ for x in gc.get_referrers(root)[:12]]
                                path['owners'].append(item)
                            paths.append(path)
            row['closure_functions'] = paths[:16]
        rows.append(row)
    return {'dead': False, 'referrers': rows, 'bounded': len(referrers) > 24}


def snapshot(observation, **references):
    result = {'backend': characterize(observation.backend)}
    if observation.model is not None:
        result['model'] = characterize(observation.model)
    result['task_roots'] = {name: ref() is not None for name, ref in references.items()}
    return result
