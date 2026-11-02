from itertools import product

class _LEXIWrapper:
    def __init__(self, labels, restrictions):
        """
        Initialise the wrapper that lexicographically streams assignments.

        Args:
            labels (Sequence[str]): Canonical labels for the candidate
                base scales (e.g., symbols for K_l).
            restrictions (Sequence[Any]): Per-coordinate allowed options.
                Each entry can be:
                    - None: all labels allowed for that coordinate.
                    - str: a single allowed label.
                    - Iterable: explicit list/tuple of allowed labels.
                    - NumPy array / boolean mask: True entries mark
                      allowed indices into labels.

        Returns: None
        """
        # Store the canonical label list and the per-coordinate restrictions
        self.labels = labels
        self.restrictions = restrictions

    def stream(self):
        """
        Stream all assignments consistent with the per-coordinate restrictions.

        Returns:
            Iterator[List[Any]]: Iterator over all allowed assignments, where
            each assignment is a list with one choice per coordinate/species.
        """
        # Shorthand to avoid repeated attribute lookups
        L = self.labels

        # Build per-coordinate choice lists according to the restrictions
        choices = []
        for ritem in self.restrictions:
            if ritem is None:
                # Fully unrestricted: all labels allowed at this position
                choices.append(L)
            elif isinstance(ritem, str):
                # Single explicit label choice
                choices.append([ritem])
            else:
                # ritem is a list of labels / indices or a boolean mask
                if ((hasattr(ritem, 'dtype') and ritem.dtype == bool)
                        or (isinstance(ritem, (list, tuple)) and ritem
                            and all(isinstance(flag, bool) for flag in ritem))):
                    # Boolean mask: keep labels where mask is True
                    if len(ritem) != len(L):
                        raise ValueError("Boolean mask must have one entry per label.")
                    choices.append([L[i] for i, flag in enumerate(ritem) if flag])
                else:
                    # Interpret as an iterable of labels/indices directly
                    choices.append(list(ritem))

        # Stream all cartesian-product combinations as assignments
        for tup in product(*choices):
            yield list(tup)
