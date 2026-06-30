import itertools
import logging
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

from Experiments.common.metrics.readers import save_idi_rows


def _install_tensorflow_shim():
    try:
        import tensorflow  # noqa: F401
        return
    except ImportError:
        pass

    tensorflow = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    backend = types.ModuleType("tensorflow.keras.backend")
    tensorflow.constant = lambda value: value
    tensorflow.keras = keras
    keras.backend = backend
    sys.modules.setdefault("tensorflow", tensorflow)
    sys.modules.setdefault("tensorflow.keras", keras)
    sys.modules.setdefault("tensorflow.keras.backend", backend)


_GRFT_SOURCE_DIR = Path(__file__).resolve().parent / "GRFT"
if str(_GRFT_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_GRFT_SOURCE_DIR))
_install_tensorflow_shim()

from .GRFT.GRFT import generate_seeds, global_generation, local_generation_random


class _KerasLikeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def numpy(self):
        return self._values

    def __array__(self, dtype=None):
        if dtype is None:
            return self._values
        return self._values.astype(dtype)

    def __bool__(self):
        return bool(np.any(self._values))

    def __gt__(self, other):
        return _KerasLikeTensor(self._values > _to_numpy(other))

    def __ge__(self, other):
        return _KerasLikeTensor(self._values >= _to_numpy(other))

    def __lt__(self, other):
        return _KerasLikeTensor(self._values < _to_numpy(other))

    def __le__(self, other):
        return _KerasLikeTensor(self._values <= _to_numpy(other))

    def __eq__(self, other):
        return _KerasLikeTensor(self._values == _to_numpy(other))

    def __ne__(self, other):
        return _KerasLikeTensor(self._values != _to_numpy(other))

    def __sub__(self, other):
        return self._values - _to_numpy(other)

    def __rsub__(self, other):
        return _to_numpy(other) - self._values


def _to_numpy(values):
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def _as_2d_array(inputs):
    arr = _to_numpy(inputs)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _unique_rows(rows, num_attribs):
    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        return np.empty(shape=(0, num_attribs))
    arr = arr.reshape(-1, num_attribs)
    seen = {}
    for row in arr:
        seen.setdefault(tuple(row), row)
    return np.asarray(list(seen.values()), dtype=float)


def _combine_rows(num_attribs, *arrays):
    rows = []
    for array in arrays:
        arr = np.asarray(array, dtype=float)
        if arr.size == 0:
            continue
        rows.append(arr.reshape(-1, num_attribs))
    if not rows:
        return np.empty(shape=(0, num_attribs))
    return _unique_rows(np.vstack(rows), num_attribs)


class _KerasLikeBlackBoxModel:
    """Adapter exposing the repo's black-box model through GRFT's Keras-like
    ``model(x).numpy()`` calling convention.

    GRFT itself produces no test-case set (its upstream "generated" return is
    empty), so there is nothing to collect or dedup here. We only keep a raw
    tally of how many instances the model is evaluated on during the search,
    used as a coarse #Test (GRFT's real output metric is the deduped #Disc).
    """

    def __init__(self, black_box_model):
        self.black_box_model = black_box_model
        self.query_count = 0

    def reset_query_log(self):
        self.query_count = 0

    def __call__(self, inputs):
        arr = _as_2d_array(inputs)
        self.query_count += int(arr.shape[0])
        try:
            values = self.black_box_model.predict_proba(arr)
        except Exception:
            values = self.black_box_model.predict(arr)
        values = np.asarray(values)
        if values.ndim == 2:
            if values.shape[1] == 1:
                values = values[:, 0]
            else:
                values = values[:, -1]
        return _KerasLikeTensor(values.reshape(-1, 1).astype(float))


class Grft:
    def __init__(self, black_box_model, protected_list, original_data, show_logging=False,
                 num_seeds=1000, c_num=4, seed_fashion="Distribution", max_iter=10,
                 l_num=1000, s_g=1.0, s_l=1.0, epsilon=1e-6, pop_num=100):
        self.black_box_model = black_box_model
        self.model = _KerasLikeBlackBoxModel(black_box_model)
        self.protected_list_no = list(protected_list)
        self.original_data = np.asarray(original_data, dtype=float)
        self.constraint = np.asarray(self.black_box_model.data_range, dtype=float)

        self.num_seeds = num_seeds
        self.c_num = c_num
        self.seed_fashion = seed_fashion
        self.max_iter = max_iter
        self.l_num = l_num
        self.s_g = s_g
        self.s_l = s_l
        self.epsilon = epsilon
        self.pop_num = pop_num

        self.disc_data = []
        self.test_data = []
        self.no_disc = 0
        self.no_test = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0
        self.search_real_time_consumed = 0
        self.search_cpu_time_consumed = 0
        self.postprocess_real_time_consumed = 0
        self.postprocess_cpu_time_consumed = 0
        self.save_real_time_consumed = 0
        self.global_real_time_consumed = 0
        self.total_iterations = 0
        self.raw_id_count = 0
        self.raw_query_count = 0

        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def _to_feature_row(self, row):
        return [int(round(float(value))) for value in np.asarray(row).reshape(-1)]

    def _predict_labels(self, rows):
        if len(rows) == 0:
            return np.empty(0, dtype=int)
        rows = [self._to_feature_row(row) for row in rows]
        return np.asarray(self.black_box_model.predict(rows), dtype=int).reshape(-1)

    def _run_search(self, deadline, label):
        num_attribs = self.black_box_model.no_attr
        if self.original_data.size == 0:
            empty = np.empty(shape=(0, num_attribs))
            return empty, empty, 0

        self.model.reset_query_log()

        def time_left():
            return deadline is None or time.process_time() < deadline

        id_parts = []
        total_iter = 0
        global_real = 0.0

        # Rounds: re-seed each round (matching GRFT's multi-ROUND design in the
        # original __main__) and keep going until the budget is spent, so a large
        # time budget is actually used instead of stopping after a single pass.
        while time_left():
            seeds = generate_seeds(
                self.original_data,
                c_num=min(self.c_num, len(self.original_data)),
                num_seeds=self.num_seeds,
                fashion=self.seed_fashion,
            )

            # Global phase: a single batched call. global_generation() iterates
            # the seeds internally and now honors the deadline, so this is both
            # faster than per-seed calls (no per-call overhead) and an exact
            # match for upstream GRFT's batched global phase.
            g_start = time.time()
            round_g_id, _gen_g, g_gen_num = global_generation(
                self.original_data, seeds, num_attribs, self.protected_list_no,
                self.constraint, self.model, self.max_iter, self.s_g, label[0],
                pop_num=self.pop_num, deadline=deadline,
            )
            total_iter += g_gen_num
            global_real += time.time() - g_start
            if len(round_g_id):
                id_parts.append(round_g_id)

            # Local phase: deadline-aware (the vendored function now breaks at the
            # deadline), so it cannot overrun the budget even when round_g_id is
            # large -- overshoot is bounded by a single local iteration.
            if len(round_g_id) and time_left():
                l_id_raw, _gen_l, l_gen_num = local_generation_random(
                    num_attribs, self.l_num, round_g_id, self.protected_list_no, self.constraint,
                    self.model, self.s_l, self.epsilon, deadline=deadline,
                )
                total_iter += l_gen_num
                if len(l_id_raw):
                    id_parts.append(_combine_rows(num_attribs, l_id_raw))

            if deadline is None:
                break  # no budget -> a single pass (deterministic amount of work)

        self.global_real_time_consumed = global_real
        ids = _combine_rows(num_attribs, *id_parts) if id_parts else np.empty(shape=(0, num_attribs))
        return ids, total_iter

    def _protected_domains(self):
        domains = []
        for attr in self.protected_list_no:
            low, high = self.constraint[attr]
            domains.append(list(range(int(low), int(high) + 1)))
        return domains

    def _counterfactual_rows(self, row):
        base = self._to_feature_row(row)
        domains = self._protected_domains()
        base_values = tuple(base[attr] for attr in self.protected_list_no)
        for comb in itertools.product(*domains):
            if tuple(comb) == base_values:
                continue
            candidate = list(base)
            for attr, value in zip(self.protected_list_no, comb):
                candidate[attr] = value
            yield candidate

    def _pair_key(self, row_a, row_b):
        row_a = tuple(self._to_feature_row(row_a))
        row_b = tuple(self._to_feature_row(row_b))
        return tuple(sorted((row_a, row_b)))

    def _build_disc_data(self, discriminatory_candidates, batch_candidates=1024):
        seen = set()
        self.disc_data = []
        batch = []

        def flush_batch():
            if not batch:
                return
            bases = [self._to_feature_row(candidate) for candidate in batch]
            base_labels = self._predict_labels(bases)
            counterfactual_rows = []
            slices = []
            for base in bases:
                start = len(counterfactual_rows)
                counterfactual_rows.extend(self._to_feature_row(row) for row in self._counterfactual_rows(base))
                slices.append((start, len(counterfactual_rows)))
            counterfactual_labels = self._predict_labels(counterfactual_rows)
            output_rows = []
            for idx, base in enumerate(bases):
                base_label = int(base_labels[idx])
                start, end = slices[idx]
                for pos in range(start, end):
                    counterfactual_label = int(counterfactual_labels[pos])
                    if counterfactual_label == base_label:
                        continue
                    counterfactual = counterfactual_rows[pos]
                    key = self._pair_key(base, counterfactual)
                    if key in seen or key[0] == key[1]:
                        continue
                    seen.add(key)
                    output_rows.append(base + [base_label])
                    output_rows.append(counterfactual + [counterfactual_label])
                    break
            self.disc_data.extend(output_rows)
            batch.clear()

        for candidate in discriminatory_candidates:
            batch.append(candidate)
            if len(batch) >= batch_candidates:
                flush_batch()
        flush_batch()
        self.no_disc = len(seen)

    def test(self, runtime=None, label=("res", 0), disc_save_to="DiscData", test_save_to="TestData"):
        start_real_time = time.time()
        start_cpu_time = time.process_time()
        deadline = None if runtime is None else start_cpu_time + float(runtime)

        logging.info(f"Starting fairness test -- {label[0]}")
        search_start_real = time.time()
        search_start_cpu = time.process_time()
        ids, total_iter = self._run_search(deadline, label)
        self.total_iterations = total_iter
        self.search_real_time_consumed = time.time() - search_start_real
        self.search_cpu_time_consumed = time.process_time() - search_start_cpu
        # GRFT has no native test-case set. Use the number of candidates it
        # evaluated as #Test: every candidate is queried against all its protected
        # variants, so the model-evaluation tally is an exact multiple of the
        # number of protected-value combinations. Dividing by that recovers the
        # candidate count, matching the "instances/anchors examined" unit of the
        # other methods. Computed during search, independent of saving.
        combos = 1
        for domain in self._protected_domains():
            combos *= max(1, len(domain))
        self.raw_query_count = self.model.query_count
        self.no_test = self.raw_query_count // combos
        self.raw_id_count = len(ids)

        logging.info(f"The fairness search is completed")
        postprocess_start_real = time.time()
        postprocess_start_cpu = time.process_time()
        # GRFT emits no test-case file (it has no test-case set). test_save_to is
        # accepted for a uniform interface; warn if a caller actually requests it.
        if test_save_to is not None:
            print(f"[GRFT] no test-case file is produced (GRFT has no test set); "
                  f"test_save_to={test_save_to!r} is ignored.", file=sys.stderr, flush=True)
        # Build always (fills self.disc_data + computes no_disc); save the binary
        # .npy only when the directory exists.
        self._build_disc_data(ids)
        if disc_save_to is not None and os.path.isdir(disc_save_to):
            logging.info(f"Saving the detected discriminatory instances to {disc_save_to}/{label[0]}-{label[1]}.npy")
            save_idi_rows(os.path.join(disc_save_to, f"{label[0]}-{label[1]}.npy"), self.disc_data)
        self.postprocess_real_time_consumed = time.time() - postprocess_start_real
        self.postprocess_cpu_time_consumed = time.process_time() - postprocess_start_cpu

        total_real_time = time.time() - start_real_time
        total_cpu_time = time.process_time() - start_cpu_time
        self.real_time_consumed = self.search_real_time_consumed
        self.cpu_time_consumed = self.search_cpu_time_consumed

        logging.info(f"The fairness test is completed")
        logging.info(f"Finished")
        print(
            "GRFT_TIMING "
            f"search_real={self.search_real_time_consumed:.6f} "
            f"search_cpu={self.search_cpu_time_consumed:.6f} "
            f"postprocess_real={self.postprocess_real_time_consumed:.6f} "
            f"postprocess_cpu={self.postprocess_cpu_time_consumed:.6f} "
            f"total_real={total_real_time:.6f} "
            f"total_cpu={total_cpu_time:.6f} "
            f"raw_ids={self.raw_id_count} "
            f"raw_queries={self.raw_query_count} "
            f"no_disc={self.no_disc} "
            f"no_test={self.no_test} "
            f"iterations={self.total_iterations}",
            flush=True,
        )
