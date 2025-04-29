import numpy as np

def find_removed_indices(ori, new):
    used = np.zeros(len(ori), dtype=bool)
    removed_indices = []

    for i, val in enumerate(ori):
        found = False
        for j, nval in enumerate(new):
            if not used[j] and val == nval:
                used[j] = True
                found = True
                break

        if not found:
            removed_indices.append(i)
            
    return removed_indices

def modified_z_socre(data: np.array):
    if len(data) < 2:
        return data > 0
    
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    if mad == 0:
        return data > 0
    
    modified_z_scores = 0.6745 * (data - median) / mad
    
    return np.abs(modified_z_scores)

def z_score(data: np.array):
    if len(data) < 2:
        return data > 0
    
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return data > 0

    z_score = np.abs((data - mean) / std)

    return z_score
