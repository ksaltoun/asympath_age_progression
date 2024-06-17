import numpy as np

def impute_zeros(X, rem_neg = False, rem_neg_ONLY = False, seed = 0):
    np.random.seed(seed)
    X = np.asarray(X)
    # print(f"Removing {len(np.where(X == 0)[0])} zero values")
    # non-parametric single-column imputation
    for i_col in range(X.shape[1]):
        vals = X[:, i_col] #pick off column
        vals_set = vals[~np.isnan(vals)] #get all values that aren't nans
        vals_set = vals_set[np.where(vals_set != 0)] #remove all zero values your goal set
        inds_zero = np.where(vals== 0)[0] #get idxs of zero rows
        if rem_neg_ONLY == True: 
            #overwrite with values that contain only negative
            vals_set = vals_set[np.where(vals_set >= 0)]
            inds_zero = np.where(vals < 0)[0] #get idxs of zero rows   
        elif rem_neg == True: 
            #overwrite with values that contain zero and negative
            vals_set = vals_set[np.where(vals_set > 0)]
            inds_zero = np.where(vals <= 0)[0] #get idxs of zero rows   
        n_misses = len(inds_zero)
        if len(vals_set) > 0: 
            inds_repl = np.random.randint(0, len(vals_set), n_misses)
            vals[inds_zero] = vals_set[inds_repl]
            # assert np.all(np.isfinite(vals))
            X[:, i_col] = vals
    return X


def impute(X, seed = 0):
    np.random.seed(seed)
    X = np.asarray(X)
    # non-parametric single-column imputation
    for i_col in range(X.shape[1]):
        vals = X[:, i_col] #pick off column
        vals_set = vals[~np.isnan(vals)] #get all values that aren't nans
        inds_nan = np.where(np.isnan(vals))[0] #get idxs of nan rows
        n_misses = len(inds_nan)
        if len(vals_set) > 0: 
            inds_repl = np.random.randint(0, len(vals_set), n_misses)
            vals[inds_nan] = vals_set[inds_repl]
            assert np.all(np.isfinite(vals))
            X[:, i_col] = vals
    return X


def my_impute(arr, seed = 0):
    np.random.seed(seed)
    print('Replacing %i NaN values!' % np.sum(np.isnan(arr)))
    arr = np.array(arr)
    b_nan = np.isnan(arr)
    b_negative = arr < 0
    b_bad = b_nan | b_negative #Or operator

    arr[b_bad] = np.random.choice(arr[~b_bad], np.sum(b_bad))
    return arr
