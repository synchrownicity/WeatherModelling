import numpy as np

def groupwise_split(X, y, group_size=100):
    assert len(X) == len(y)
    num_points = len(X)
    num_groups = num_points // group_size
    indices = np.arange(num_groups)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_groups = indices[:int(0.8 * num_groups)]
    test_groups = indices[int(0.8 * num_groups):]

    train_idx = np.concatenate([np.arange(g * group_size, (g + 1) * group_size) for g in train_groups])
    test_idx = np.concatenate([np.arange(g * group_size, (g + 1) * group_size) for g in test_groups])

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
