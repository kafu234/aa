import numpy as np


def make_recording_group_ids(subjects, sessions, trials):
    """Return one integer group id for each complete subject/session/trial."""
    keys = np.stack([subjects, sessions, trials], axis=1)
    _, group_ids = np.unique(keys, axis=0, return_inverse=True)
    return group_ids.astype(np.int64)


def group_holdout(groups, val_ratio=0.15, seed=42):
    """Hold out complete groups without consulting labels."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two complete groups are required for holdout")

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    target_samples = max(1, int(round(len(groups) * val_ratio)))
    val_groups = []
    selected_samples = 0
    for group_id in unique_groups[:-1]:
        val_groups.append(group_id)
        selected_samples += int(np.sum(groups == group_id))
        if selected_samples >= target_samples:
            break

    val_mask = np.isin(groups, np.asarray(val_groups))
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def stratified_group_holdout(labels, groups, val_ratio=0.15, seed=42):
    """Hold out complete groups while preserving every class in the fit split."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")

    rng = np.random.RandomState(seed)
    val_groups = []
    for class_id in np.unique(labels):
        class_mask = labels == class_id
        class_groups = np.unique(groups[class_mask])
        if len(class_groups) < 2:
            raise ValueError(
                f"class {class_id} has only {len(class_groups)} complete group(s); "
                "cannot create a leakage-free validation split"
            )

        rng.shuffle(class_groups)
        target_samples = max(1, int(round(class_mask.sum() * val_ratio)))
        selected_samples = 0
        for group_id in class_groups[:-1]:
            val_groups.append(group_id)
            selected_samples += int(np.sum(class_mask & (groups == group_id)))
            if selected_samples >= target_samples:
                break

    val_mask = np.isin(groups, np.asarray(val_groups))
    fit_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(fit_idx) == 0 or len(val_idx) == 0:
        raise ValueError("group holdout produced an empty fit or validation split")
    return fit_idx, val_idx
