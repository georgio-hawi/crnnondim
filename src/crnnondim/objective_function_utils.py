"""
Constrained families for nondimensionalisation:
- Level-1: minimise max_i sup_t |log(X_i/K_i)|
- Lexicographic: among Level-1, minimise 2nd largest, then 3rd, ..., until fixed

Constraints: per-coordinate allowed subsets S_i of S.
Selection is O(sum_i log |S_i|) after a single O(k log k) sort of candidates.
Enumeration is amortized O(1) per solution.
"""
from __future__ import annotations
from typing import Any, Iterator, List, Optional, Sequence, Union, Dict
import numpy as np
# --------- Helpers to normalise candidate set & per-i constraints ---------

def _prep_candidates(
    K_candidates: Sequence[float],
    labels: Optional[Sequence[str]] = None,
):
    """
    Preprocess candidate scale values for 1D optimisation.

    Args:
        K_candidates (Sequence[float]): Positive candidate K_l values.
        labels (Optional[Sequence[str]]): Optional string label for each
            candidate value; must be the same length as K_candidates.

    Returns:
        Tuple[..., ...]: A tuple (Lk_sorted, K_sorted, labels_sorted,
        label_to_sorted, inv_order) where the candidates are sorted by log(K)
        and the mappings relate original indices, sorted indices and labels.
    """
    # Sort candidates in log-space and build lookup structures to translate
    # between original indices, sorted indices and human-readable labels.
    K = np.asarray(K_candidates, dtype=float)
    if K.ndim != 1 or K.size == 0 or not np.all(np.isfinite(K)) or np.any(K <= 0.0):
        raise ValueError("Candidate K must be a nonempty vector of finite positive values.")
    Lk = np.log(K)
    order = np.argsort(Lk, kind="mergesort")  # stable tie handling
    Lk_sorted = Lk[order]
    K_sorted = K[order]
    if labels is not None:
        if len(labels) != len(K):
            raise ValueError("labels must match K_candidates length")
        if len(set(map(str, labels))) != len(labels):
            raise ValueError("Candidate labels must be unique.")
        labels_sorted = [str(labels[i]) for i in order]
        label_to_sorted = {labels_sorted[j]: j for j in range(len(K))}
    else:
        labels_sorted = None
        label_to_sorted = None

    # Map original index -> sorted index
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)

    return Lk_sorted, K_sorted, labels_sorted, label_to_sorted, inv_order


def _coerce_one_restriction(
    r,
    k: int,
    labels_sorted: Optional[Sequence[str]],
    label_to_sorted: Optional[Dict[str, int]],
    inv_order: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Normalise a single per-coordinate restriction into sorted index form.

    Args:
        r: One restriction specification. Can be:
            - None / "all": no restriction.
            - int or sequence of ints: candidate indices.
            - str or sequence of str: candidate labels.
            - slice: range of indices.
            - np.ndarray or list of bool: mask over candidates.
        k (int): Total number of candidate values.
        labels_sorted (Optional[Sequence[str]]): Candidate labels in the
            sorted order returned by _prep_candidates.
        label_to_sorted (Optional[Dict[str, int]]): Map from label to sorted
            index.
        inv_order (np.ndarray): Inverse permutation from sorted index to
            original index.

    Returns:
        Optional[np.ndarray]: Sorted array of allowed candidate indices (in
        sorted order), or None if the coordinate is unrestricted.
    """
    # Convert all supported specification formats into a sorted array of
    # indices in the sorted-candidate order, or None for "no restriction".

    def sorted_unique(arr: np.ndarray) -> np.ndarray:
        """
        Return the sorted unique elements from a NumPy array.

        Args:
            arr (np.ndarray): Input array (any dtype that supports sorting).

        Returns:
            np.ndarray: New 1D array of sorted unique values.
        """
        # Use np.unique to get sorted unique values.
        arr = np.unique(arr.astype(np.int64, copy=False))
        if arr.size == 0:
            raise ValueError("Restriction yields empty allowed set.")
        return arr

    # Unrestricted
    if r is None or (isinstance(r, str) and r.lower() == "all"):
        return None
    if isinstance(r, slice):
        return sorted_unique(inv_order[np.arange(k)[r]])

    # NumPy arrays first (to avoid ambiguous truth pitfalls)
    if isinstance(r, np.ndarray):
        # Boolean mask in ORIGINAL order
        if r.dtype == bool:
            if r.size != k:
                raise ValueError("Boolean mask must have length k.")
            allowed_orig = np.flatnonzero(r)
            if allowed_orig.size == 0:
                raise ValueError("Boolean mask yields empty allowed set.")
            return np.sort(inv_order[allowed_orig])

        # Integer indices in ORIGINAL order
        if np.issubdtype(r.dtype, np.integer):
            arr = r.astype(np.int64, copy=False)
            if np.any((arr < 0) | (arr >= k)):
                raise ValueError("Restriction index out of range.")
            return sorted_unique(inv_order[arr])

        # String labels
        if np.issubdtype(r.dtype, np.str_) or np.issubdtype(r.dtype, np.object_):
            if labels_sorted is None or label_to_sorted is None:
                raise ValueError("String labels provided but no labels were passed.")
            idx = [label_to_sorted[str(x)] for x in r.tolist()]
            return sorted_unique(np.array(idx, dtype=np.int64))

        raise ValueError(f"Unsupported numpy array dtype: {r.dtype}")

    # Single scalar int: ORIGINAL index
    if isinstance(r, (int, np.integer)):
        ri = int(r)
        if ri < 0 or ri >= k:
            raise ValueError("Restriction index out of range.")
        return np.array([int(inv_order[ri])], dtype=np.int64)

    # Single string label
    if isinstance(r, str):
        if labels_sorted is None or label_to_sorted is None:
            raise ValueError("String label provided but no labels were passed.")
        return np.array([label_to_sorted[r]], dtype=np.int64)

    # Python containers
    if isinstance(r, (list, tuple, set)):
        if len(r) == 0:
            raise ValueError("Restriction yields empty allowed set.")
        if isinstance(r, (list, tuple)) and all(isinstance(v, (bool, np.bool_)) for v in r):
            return _coerce_one_restriction(
                np.asarray(r, dtype=bool), k, labels_sorted,
                label_to_sorted, inv_order,
            )
        first = next(iter(r))
        # Integers (ORIGINAL indices)
        if isinstance(first, (int, np.integer)):
            arr = np.array(list(r), dtype=np.int64)
            if np.any((arr < 0) | (arr >= k)):
                raise ValueError("Restriction index out of range.")
            return sorted_unique(inv_order[arr])
        # Strings (labels)
        if isinstance(first, str):
            if labels_sorted is None or label_to_sorted is None:
                raise ValueError("String labels provided but no labels were passed.")
            idx = [label_to_sorted[str(x)] for x in r]
            return sorted_unique(np.array(idx, dtype=np.int64))
        raise ValueError(f"Unsupported restriction element type: {type(first)}")

    raise ValueError(f"Unsupported restriction format: {type(r)}")

def _normalise_restrictions(
    n: int,
    k: int,
    restrictions: Optional[Sequence[Any]],
    labels_sorted: Optional[Sequence[str]],
    label_to_sorted: Optional[Dict[str, int]],
    inv_order: np.ndarray,
) -> List[Optional[np.ndarray]]:
    """
    Normalise per-coordinate restrictions for all coordinates.

    Args:
        n (int): Number of coordinates (typically number of species).
        k (int): Number of candidate scale values.
        restrictions (Optional[Sequence[Any]]): Sequence of per-coordinate
            restriction specs, or None for all free.
        labels_sorted (Optional[Sequence[str]]): Candidate labels in sorted
            order.
        label_to_sorted (Optional[Dict[str, int]]): Label -> sorted-index map.
        inv_order (np.ndarray): Inverse permutation to original index order.

    Returns:
        List[Optional[np.ndarray]]: For each coordinate i, either None
        (unrestricted) or a sorted array of allowed sorted indices.
    """
    # Apply _coerce_one_restriction to each coordinate and collect results.
    if restrictions is None:
        return [None] * n
    if len(restrictions) != n:
        raise ValueError("restrictions must have length n (one per coordinate).")
    out: List[Optional[np.ndarray]] = []
    for r in restrictions:
        out.append(_coerce_one_restriction(r, k, labels_sorted, label_to_sorted, inv_order))
    return out


# --------- Core computation with constraints ---------

class Level1Family:
    def __init__(self):
        """
        Container for Level-1 optimal assignments and their error structure.

        Args:
            self: New Level1Family instance being created.

        Returns: None.
        """
        # Initialise fields that will be filled by compute_families_...
        self.Rstar: float = 0.0
        self.Lk: Optional[np.ndarray] = None
        self.K_sorted: Optional[np.ndarray] = None
        self.labels_sorted: Optional[List[str]] = None

        # Per-coordinate representation of allowed Level-1 ranges:
        # use_sub[i] == False: use global [lo_full[i], hi_full[i]) on Lk
        # use_sub[i] == True : use subset sub_idx[i] and [lo_sub[i], hi_sub[i]) over sub_idx
        self.use_sub: Optional[np.ndarray] = None
        self.lo_full: Optional[np.ndarray] = None
        self.hi_full: Optional[np.ndarray] = None
        self.sub_idx: List[Optional[np.ndarray]] = []
        self.lo_sub: Optional[np.ndarray] = None
        self.hi_sub: Optional[np.ndarray] = None

        # Per-i argmin tie set (width 1 or 2) in the same representation:
        self.arg_use_sub: Optional[np.ndarray] = None
        self.arg_lo_full: Optional[np.ndarray] = None
        self.arg_hi_full: Optional[np.ndarray] = None
        self.arg_lo_sub: Optional[np.ndarray] = None
        self.arg_hi_sub: Optional[np.ndarray] = None

        self.counts: Optional[np.ndarray] = None  # Level-1 counts per i

    def stream(self, max_solutions: Optional[int] = None, as_labels: bool = True) -> Iterator[List[Union[str, float]]]:
        """
        Iterate over all Level-1 optimal assignments.

        Args:
            max_solutions (Optional[int]): Maximum number of assignments to
                yield. None means no explicit limit.
            as_labels (bool): If True and labels are available, yield label
                strings. Otherwise yield numeric candidate values.

        Returns:
            Iterator[List[Union[str, float]]]: Iterator over assignments,
            each assignment being a list of labels or float values, one per
            coordinate, that attain the minimal maximal sup-error.
        """
        # Iterate over the argmin sets per coordinate and yield assignments.
        if max_solutions is not None and max_solutions <= 0:
            return
        Lk = self.Lk; K_sorted = self.K_sorted
        assert Lk is not None and K_sorted is not None
        use_sub = self.use_sub; lo_full = self.lo_full; 
        sub_idx = self.sub_idx; lo_sub = self.lo_sub
        counts = self.counts
        n = len(use_sub)
        idx = [0] * n
        produced = 0

        def proj(global_idx: int):
            """
            Project a global index into a (coordinate, local_index) pair.

            Args:
                global_idx (int): Flat index in the product of per-coordinate choices.

            Returns:
                Tuple[int, int]: Pair (i, j) such that i is the coordinate index and
                j is the index within the allowed set for that coordinate.
            """
            # Map a flat index into coordinate- and local-choice indices.
            if as_labels and self.labels_sorted is not None:
                return self.labels_sorted[global_idx]
            return float(K_sorted[global_idx])

        while True:
            out = []
            for i in range(n):
                if not use_sub[i]:
                    g = int(lo_full[i] + idx[i])
                else:
                    g = int(sub_idx[i][lo_sub[i] + idx[i]])
                out.append(proj(g))
            yield out
            produced += 1
            if max_solutions is not None and produced >= max_solutions:
                return
            # odometer
            p = 0
            while p < n:
                idx[p] += 1
                if idx[p] < counts[p]:
                    break
                idx[p] = 0
                p += 1
            if p == n:
                return


class LexicographicFamily:
    def __init__(self, base: Level1Family):
        """
        Wrapper that imposes lexicographic optimality on Level-1 assignments.

        Args:
            base (Level1Family): Fully initialised Level1Family whose argmin
                ranges encode all Level-1 optimal choices.

        Returns: None.
        """
        # Store base so that stream() can iterate over lexicographically
        # optimal assignments within the Level-1 solution set.
        self.base = base
        # We reuse the same representation, but with argmin ranges
        self.use_sub = base.arg_use_sub
        self.lo_full = base.arg_lo_full
        self.hi_full = base.arg_hi_full
        self.sub_idx = base.sub_idx
        self.lo_sub = base.arg_lo_sub
        self.hi_sub = base.arg_hi_sub
        self.Lk = base.Lk
        self.K_sorted = base.K_sorted
        self.labels_sorted = base.labels_sorted
        # counts
        counts = np.empty_like(base.counts)
        for i in range(len(counts)):
            if not self.use_sub[i]:
                counts[i] = self.hi_full[i] - self.lo_full[i]
            else:
                counts[i] = self.hi_sub[i] - self.lo_sub[i]
        self.counts = counts

    def stream(self, max_solutions: Optional[int] = None, as_labels: bool = True) -> Iterator[List[Union[str, float]]]:
        """
        Stream assignments that are lexicographically optimal among Level-1 optima.

        Args:
            max_solutions (Optional[int]): Maximum number of assignments to
                generate, or None for no explicit limit.
            as_labels (bool): If True, emit labels when available; otherwise
                emit numeric candidate values.

        Returns:
            Iterator[List[Union[str, float]]]: Iterator over assignments that
            minimise the sorted error vector lexicographically.
        """
        # Iterate over combinations of Level-1 argmin choices in lexicographic
        # order and yield the corresponding assignments.
        yield from Level1Family.stream(self, max_solutions=max_solutions, as_labels=as_labels)


def compute_families_sup_lexi_constrained(
    Xmin: Sequence[float],
    Xmax: Sequence[float],
    K_candidates: Sequence[float],
    restrictions: Optional[Sequence[Any]] = None,
    labels: Optional[Sequence[str]] = None,
    tol: float = 1e-12,
):
    """
    Build Level-1 and lexicographic families for the sup-norm objective.

    Args:
        Xmin (Sequence[float]): Componentwise minima of the trajectories X_i(t).
        Xmax (Sequence[float]): Componentwise maxima of the trajectories X_i(t).
        K_candidates (Sequence[float]): Positive candidate scale values K_l.
        restrictions (Optional[Sequence[Any]]): Per-coordinate allowed subsets
            S_i in the same formats as _coerce_one_restriction accepts.
        labels (Optional[Sequence[str]]): Optional labels for candidates.
        tol (float): Numerical tolerance for comparing log-ranges and
            tie-breaking equalities.

    Returns:
        Tuple[Level1Family, LexicographicFamily]: The Level-1 family and its
        lexicographically optimal subfamily under the given constraints.
    """
    # Compute componentwise log-ranges, build candidate error tables, then
    # populate a Level1Family and wrap it in LexicographicFamily.
    Xmin = np.asarray(Xmin, dtype=float)
    Xmax = np.asarray(Xmax, dtype=float)
    if Xmin.shape != Xmax.shape:
        raise ValueError("Xmin/Xmax shape mismatch.")
    if (Xmin.ndim != 1 or Xmin.size == 0 or not np.all(np.isfinite(Xmin))
            or not np.all(np.isfinite(Xmax)) or np.any(Xmin <= 0.0)
            or np.any(Xmax <= 0.0)):
        raise ValueError("Extrema must be nonempty vectors of finite positive values.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")

    # Log summaries
    a = np.log(np.minimum(Xmin, Xmax))
    b = np.log(np.maximum(Xmin, Xmax))
    m = 0.5 * (a + b)
    h = 0.5 * (b - a)

    # Candidates
    Lk, K_sorted, labels_sorted, label_to_sorted, inv_order = _prep_candidates(K_candidates, labels)
    k = Lk.size
    n = m.size

    # Normalise restrictions to sorted-index arrays
    Rlist = _normalise_restrictions(n, k, restrictions, labels_sorted, label_to_sorted, inv_order)

    # Outputs we populate
    lvl = Level1Family()
    lvl.Lk = Lk; lvl.K_sorted = K_sorted; lvl.labels_sorted = labels_sorted
    lvl.use_sub = np.zeros(n, dtype=bool)
    lvl.lo_full = np.zeros(n, dtype=np.int64)
    lvl.hi_full = np.zeros(n, dtype=np.int64)
    lvl.sub_idx = [None] * n
    lvl.lo_sub = np.zeros(n, dtype=np.int64)
    lvl.hi_sub = np.zeros(n, dtype=np.int64)

    lvl.arg_use_sub = np.zeros(n, dtype=bool)
    lvl.arg_lo_full = np.zeros(n, dtype=np.int64)
    lvl.arg_hi_full = np.zeros(n, dtype=np.int64)
    lvl.arg_lo_sub = np.zeros(n, dtype=np.int64)
    lvl.arg_hi_sub = np.zeros(n, dtype=np.int64)

    # Pass 1: per-i nearest in S_i to m_i  -> r_i and argmin tie sets
    r_i = np.empty(n, dtype=float)
    for i in range(n):
        Sidx = Rlist[i]  # None or sorted indices into Lk
        if Sidx is None:
            # unrestricted: search in full Lk
            j = int(np.searchsorted(Lk, m[i], side="left"))
            # neighbors
            left_ok = (j > 0); right_ok = (j < k)
            left_dist  = (m[i] - Lk[j-1]) if left_ok  else np.inf
            right_dist = (Lk[j] - m[i])   if right_ok else np.inf
            dmin = min(left_dist, right_dist)
            r_i[i] = h[i] + dmin
            # argmin tie set within full
            if np.isfinite(left_dist) and np.isfinite(right_dist) and abs(left_dist - right_dist) <= tol:
                lvl.arg_use_sub[i] = False
                lvl.arg_lo_full[i] = j - 1
                lvl.arg_hi_full[i] = j + 1
            elif left_dist <= right_dist:
                lvl.arg_use_sub[i] = False
                lvl.arg_lo_full[i] = j - 1
                lvl.arg_hi_full[i] = j
            else:
                lvl.arg_use_sub[i] = False
                lvl.arg_lo_full[i] = j
                lvl.arg_hi_full[i] = j + 1
            # Preserve every symbolic label sharing a selected numerical scale.
            lvl.arg_lo_full[i] = int(np.searchsorted(Lk, Lk[lvl.arg_lo_full[i]], side="left"))
            lvl.arg_hi_full[i] = int(np.searchsorted(Lk, Lk[lvl.arg_hi_full[i] - 1], side="right"))
        else:
            # restricted: search in subarray Lk[Sidx]
            Lsub = Lk[Sidx]
            js = int(np.searchsorted(Lsub, m[i], side="left"))
            left_ok = (js > 0); right_ok = (js < Lsub.size)
            left_dist  = (m[i] - Lsub[js-1]) if left_ok  else np.inf
            right_dist = (Lsub[js] - m[i])   if right_ok else np.inf
            dmin = min(left_dist, right_dist)
            r_i[i] = h[i] + dmin
            # argmin tie set inside subarray (width 1 or 2), store as sub positions
            lvl.arg_use_sub[i] = True
            lvl.sub_idx[i] = Sidx  # keep for later mapping
            if np.isfinite(left_dist) and np.isfinite(right_dist) and abs(left_dist - right_dist) <= tol:
                lvl.arg_lo_sub[i] = js - 1
                lvl.arg_hi_sub[i] = js + 1
            elif left_dist <= right_dist:
                lvl.arg_lo_sub[i] = js - 1
                lvl.arg_hi_sub[i] = js
            else:
                lvl.arg_lo_sub[i] = js
                lvl.arg_hi_sub[i] = js + 1
            # Expand equal-valued labels within the allowed subset only.
            lvl.arg_lo_sub[i] = int(np.searchsorted(Lsub, Lsub[lvl.arg_lo_sub[i]], side="left"))
            lvl.arg_hi_sub[i] = int(np.searchsorted(Lsub, Lsub[lvl.arg_hi_sub[i] - 1], side="right"))

    Rstar = float(np.max(r_i))
    lvl.Rstar = Rstar
    # Pass 2: Level-1 admissible ranges: |L - m_i| <= R* - h_i intersect S_i
    counts = np.empty(n, dtype=np.int64)
    for i in range(n):
        thr = Rstar - h[i] + tol  # include boundary
        left = m[i] - thr
        right = m[i] + thr
        Sidx = Rlist[i]
        if Sidx is None:
            # unrestricted: global [lo, hi)
            lo = int(np.searchsorted(Lk, left,  side="left"))
            hi = int(np.searchsorted(Lk, right, side="right"))
            if hi <= lo:
                # widen slightly if numerics bite
                eps = 10 * max(1e-15, tol)
                lo = int(np.searchsorted(Lk, left - eps,  side="left"))
                hi = int(np.searchsorted(Lk, right + eps, side="right"))
                if hi <= lo:
                    raise RuntimeError("Empty Level-1 interval under constraints (unrestricted case).")
            lvl.use_sub[i] = False
            lvl.lo_full[i] = lo
            lvl.hi_full[i] = hi
            counts[i] = hi - lo
        else:
            Lsub = Lk[Sidx]
            lo_s = int(np.searchsorted(Lsub, left,  side="left"))
            hi_s = int(np.searchsorted(Lsub, right, side="right"))
            if hi_s <= lo_s:
                eps = 10 * max(1e-15, tol)
                lo_s = int(np.searchsorted(Lsub, left - eps,  side="left"))
                hi_s = int(np.searchsorted(Lsub, right + eps, side="right"))
                if hi_s <= lo_s:
                    raise RuntimeError("Empty Level-1 interval under constraints for some coordinate.")
            lvl.use_sub[i] = True
            # sub_idx[i] already set if restricted in pass 1; ensure it exists:
            if lvl.sub_idx[i] is None:
                lvl.sub_idx[i] = Sidx
            lvl.lo_sub[i] = lo_s
            lvl.hi_sub[i] = hi_s
            counts[i] = hi_s - lo_s

    lvl.counts = counts
    lexi = LexicographicFamily(lvl)
    return lvl, lexi
