"""Metrics for IDIs stored as consecutive anchor/counterfactual row blocks."""
import numpy as np


def _take_blocks(rows, block_size):
    n = rows.shape[0] // block_size
    return n, rows[: n * block_size]


def count_idi(rows, block_size=2):
    """Count unique IDIs, treating a block and its reverse as the same IDI."""
    n, rows = _take_blocks(rows, block_size)
    if n == 0:
        return 0
    d = rows.shape[1]
    blocks = np.ascontiguousarray(rows).reshape(n, block_size, d)
    width = block_size * d * 8
    forward = blocks.astype(">u8").reshape(n, block_size * d).view(f"S{width}").ravel()
    reverse = blocks[:, ::-1, :].astype(">u8").reshape(n, block_size * d).view(f"S{width}").ravel()
    # Big-endian bytes preserve numeric order for the non-negative feature codes.
    canon = np.where(forward <= reverse, forward, reverse)
    return int(np.unique(canon).shape[0])


def _kept_columns(d, exclude_idx):
    if exclude_idx is None:
        skip = set()
    elif isinstance(exclude_idx, int):
        skip = {exclude_idx}
    else:
        skip = set(exclude_idx)
    return [i for i in range(d) if i not in skip]


def _anchor_buckets(rows, ranges, exclude_idx, g, block_size):
    """Map anchors to ``g`` equal buckets per non-protected dimension."""
    n, rows = _take_blocks(rows, block_size)
    if n == 0:
        return np.empty((0, 0), dtype=np.int64)
    anchors = rows[::block_size]
    ranges = np.asarray(ranges, dtype=np.int64)
    low, card = ranges[:, 0], np.maximum(1, ranges[:, 1] - ranges[:, 0] + 1)
    buckets = (anchors - low) * g // card
    np.clip(buckets, 0, g - 1, out=buckets)
    kept = _kept_columns(rows.shape[1], exclude_idx)
    return buckets[:, kept].astype(np.int64, copy=False)


def occupied_cells(rows, ranges, exclude_idx, g, block_size=2):
    """Return the grid cells occupied by IDI anchors as bucket tuples."""
    buckets = _anchor_buckets(rows, ranges, exclude_idx, g, block_size)
    if buckets.shape[0] == 0:
        return set()
    return set(map(tuple, np.unique(buckets, axis=0).tolist()))


def cell_codes(rows, ranges, exclude_idx, g, block_size=2):
    """Pack unique occupied grid cells into comparable int64 codes.

    Bucket tuples use mixed-radix base ``g``. Codes are comparable only when
    ``g`` and ``exclude_idx`` are the same.
    """
    buckets = _anchor_buckets(rows, ranges, exclude_idx, g, block_size)
    n, d = buckets.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if d == 0:
        return np.zeros(1, dtype=np.int64)
    if g ** d > 2 ** 63 - 1:
        raise OverflowError(f"cell codes overflow int64 (g={g}, kept dims={d}); use a smaller g")
    radix = np.array([g ** i for i in range(d)], dtype=np.int64)
    return np.unique(buckets @ radix)


def coverage(rows, ranges, exclude_idx, g, block_size=2):
    """Count distinct grid cells occupied at granularity ``g``."""
    return int(cell_codes(rows, ranges, exclude_idx, g, block_size).shape[0])


def _cell_overlap(left, right):
    intersection = int(
        np.intersect1d(left, right, assume_unique=True).shape[0]
    )
    union = left.shape[0] + right.shape[0] - intersection
    containment = (
        intersection / left.shape[0] if left.shape[0] else float("nan")
    )
    jaccard = intersection / union if union else float("nan")
    return intersection, containment, jaccard


def containment_table(method_codes, methods):
    """Compute |A∩B|, |A∩B|/|A|, and Jaccard over pooled method cells."""
    empty = np.empty(0, dtype=np.int64)
    codes = {
        method: np.asarray(method_codes.get(method, empty), dtype=np.int64)
        for method in methods
    }
    intersection, containment, jaccard = {}, {}, {}
    for left_method in methods:
        for right_method in methods:
            values = _cell_overlap(
                codes[left_method],
                codes[right_method],
            )
            key = (left_method, right_method)
            intersection[key], containment[key], jaccard[key] = values
    return {
        "intersection": intersection,
        "containment": containment,
        "jaccard": jaccard,
    }


def _mean_without_nan(values):
    values = [value for value in values if value == value]
    return float(sum(values) / len(values)) if values else float("nan")


def containment_table_per_run(method_run_codes, methods):
    """Average cell relationships over every ordered pair of method runs.

    This avoids pooling repeats or imposing an arbitrary repeat pairing.
    """
    runs = {
        method: [
            np.asarray(codes, dtype=np.int64)
            for codes in method_run_codes.get(method, [])
        ]
        for method in methods
    }
    intersection, containment, jaccard = {}, {}, {}
    for left_method in methods:
        for right_method in methods:
            intersections, containments, jaccards = [], [], []
            for left_codes in runs[left_method]:
                for right_codes in runs[right_method]:
                    inter, contain, jac = _cell_overlap(
                        left_codes,
                        right_codes,
                    )
                    intersections.append(float(inter))
                    containments.append(contain)
                    jaccards.append(jac)

            key = (left_method, right_method)
            intersection[key] = _mean_without_nan(intersections)
            containment[key] = _mean_without_nan(containments)
            jaccard[key] = _mean_without_nan(jaccards)
    return {
        "intersection": intersection,
        "containment": containment,
        "jaccard": jaccard,
    }


def _mean_pairwise_hamming(sample):
    """Compute mean pairwise Hamming distance in O(n*d)."""
    m = sample.shape[0]
    if m <= 1:
        return 0.0
    total_pairs = m * (m - 1) / 2.0
    diff = 0.0
    for col in range(sample.shape[1]):
        _, counts = np.unique(sample[:, col], return_counts=True)
        same = np.sum(counts * (counts - 1) / 2.0)
        diff += total_pairs - same
    return float(diff / total_pairs)


def diversity(rows, sample_size=1000, repeats=5, rng=None, block_size=2):
    """Estimate mean pairwise Hamming distance between sampled IDI anchors."""
    n, rows = _take_blocks(rows, block_size)
    if n == 0:
        return 0.0
    anchors = rows[::block_size]
    if anchors.shape[0] <= sample_size:
        return _mean_pairwise_hamming(anchors)
    rng = rng or np.random.default_rng(0)
    total = 0.0
    for _ in range(repeats):
        idx = rng.choice(anchors.shape[0], size=sample_size, replace=False)
        total += _mean_pairwise_hamming(anchors[idx])
    return total / repeats
