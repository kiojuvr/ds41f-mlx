"""Bounded same-forward acceleration receipts. No history/cache or replay API."""
import numpy as np


def detach(mx, value):
    mx.eval(value)
    result = mx.array(np.array(value.view(mx.uint8), copy=True)).view(value.dtype).reshape(value.shape)
    mx.eval(result)
    return result


class CommittedTapReceipt:
    def __init__(self, child, owner, start, rows):
        child.assert_active()
        if rows.shape[0] != 1 or not 0 < rows.shape[1] <= max(child.config.window_size, 32):
            raise ValueError('bounded unpadded taps required')
        if rows.shape[2] != child.config.dim * len(child.config.dspark_target_layer_ids):
            raise ValueError('tap width mismatch')
        self.child, self.owner = child, owner
        self.start, self.end = int(start), int(start) + rows.shape[1]
        self.rows = detach(child.mx, rows)
        self.retired = False
        child.receipts.add(self)

    def take(self, child, owner, start):
        child.assert_active()
        if self.retired or self.child is not child or self.owner is not owner or self.start != start:
            raise RuntimeError('stale/misbound tap receipt')
        rows = self.rows
        self.retire()
        return rows

    def retire(self):
        self.rows = None
        self.retired = True
        self.child.receipts.discard(self)


class PrefillTapCapture:
    def __init__(self, child):
        self.child = child
        self.parts = {}

    def capture(self, layer, start, h):
        c, mx = self.child.config, self.child.mx
        if layer not in c.dspark_target_layer_ids:
            return
        rows = detach(mx, mx.mean(h[:, -c.window_size:], axis=2))
        start += h.shape[1] - rows.shape[1]
        previous = self.parts.get(layer)
        if previous is not None:
            pstart, prows = previous
            if pstart + prows.shape[1] != start:
                # A chunk longer than the retained window replaces it entirely.
                if h.shape[1] < c.window_size:
                    raise RuntimeError('noncontiguous prefill taps')
            else:
                rows = detach(mx, mx.concatenate([prows, rows], 1)[:, -c.window_size:])
                start = pstart + prows.shape[1] + h.shape[1] - rows.shape[1]
        self.parts[layer] = (start, rows)

    def commit(self, owner, frontier):
        c, mx = self.child.config, self.child.mx
        values = [self.parts[i] for i in c.dspark_target_layer_ids]
        starts = {start for start, _ in values}
        if len(starts) != 1 or any(start + rows.shape[1] != frontier for start, rows in values):
            raise RuntimeError('prefill tap frontier mismatch')
        receipt = CommittedTapReceipt(self.child, owner, values[0][0],
                                     mx.concatenate([rows for _, rows in values], -1))
        self.parts.clear()
        return receipt
