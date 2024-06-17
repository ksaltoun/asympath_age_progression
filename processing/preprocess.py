import pandas as pd 
import numpy as np 
import processing.impute_fncs as impute_fncs
from nilearn.signal import clean
import time

expt_type_dict = {
        "Av_Vol" : lambda R_Side, L_Side : (R_Side + L_Side) / 2 ,
        "Tot_Vol" : lambda R_Side, L_Side : (R_Side + L_Side) ,
        "Vol_R_side" : lambda R_Side, L_Side : R_Side ,
        "Vol_L_side" : lambda R_Side, L_Side : L_Side ,
        "Vol_Sub_R-L" : lambda R_Side, L_Side : R_Side - L_Side ,
        "Vol_Sub_Abs" : lambda R_Side, L_Side : abs(R_Side - L_Side) ,
        "Vol_Sub_Sign" : lambda R_Side, L_Side : np.sign(R_Side - L_Side) ,
        "Lat_Idx" : lambda R_Side, L_Side : (R_Side - L_Side) / (R_Side + L_Side), 
        "Lat_Idx_Av" : lambda R_Side, L_Side : 2* (R_Side - L_Side) / (R_Side + L_Side), 
        "Lat_Idx_Abs" : lambda R_Side, L_Side : abs(R_Side - L_Side) / (R_Side + L_Side) 
}

def remove_rows(df, rows_2_rem):
    rows_2_rem = list(set(rows_2_rem))
    bad_df = df.index.isin(rows_2_rem)
    df = df[~bad_df]
    return(df)

def deconf(conf_mat, X, rows_2_rem = None):
    t1 = time.time()
    X = np.asarray(X)
    conf_mat = np.asarray(conf_mat)
    
    if conf_mat.shape[0] != X.shape[0]:
        if rows_2_rem is None:
            raise ValueError("""Conf matrix and X values are not the same size
                                 and no rows to remove are given""")
        print(f"Removing {len(rows_2_rem)} rows from conf_mat")
        conf_mat = np.delete(conf_mat, rows_2_rem, axis = 0)

    print('Deconfounding volume space!')
    X_clean = clean(X, confounds=conf_mat, detrend=False, standardize=False)
        
    t2=time.time()
    print(f"Deconfounding took  {(t2-t1)/60:.4f} min")
    print('\a')
    return X_clean

def one_side_perm(dfX, regions_list, regions_left, regions_right, seed=0):
    #dfX is an pandas array with symmetrical values
    #regions list is list of all symmetrical values
    #regions right (left) are dict mapping region : col_name
    #RETURNS an array of n x m 
    # where n is the incoming length 
    # and m is the number of values in regions list
    np.random.seed(seed)
    #PERMUTUATION
    #for every column, randomly select the brain region from left or right side
    N_PPL = dfX.shape[0]
    regions_rands = np.zeros((N_PPL, len(regions_list)))
    for ii, c in enumerate(regions_list):
        np.random.seed(seed + ii**2)
        curr_combo =  np.vstack((dfX[regions_left[c]] , 
                                    dfX[regions_right[c]] )).T
        RL_mask = np.random.randint(2, size = (N_PPL))
        region_rand = np.vstack(
                    (curr_combo[:,1]*(RL_mask == 1).astype(int), 
                        curr_combo[:,0]*(RL_mask == 0).astype(int))
                    ).sum(axis = 0)
        regions_rands[:,ii] = region_rand
    return(regions_rands)
    

class MRI_processing:
    def __init__(self, ukbb_MRI, regions, desc = None, clr_uneven = True):
        #ukbb_MRI is a dataframe with column headers as ukbb col numbers
        #regions is a dict of regions of the form {col_name : 'AREA (SIDE)'}
        self.ukbb_MRI = ukbb_MRI
        self.type = desc if desc is not None else 'None_Given'
        self.regions_dict = regions
        self.regions_all = list(regions.values())
        self.expt_allowed = ["Regular", *list(expt_type_dict.keys()), "One_Side_Perm"]

        self.emp_rows = np.where((~np.nan_to_num(ukbb_MRI).any(axis = 1)))[0]
        self.mask = ~np.bincount(self.emp_rows, minlength = self.ukbb_MRI.shape[0]).astype(bool)

        # split into L and R regions by making two dictionaries with 
        #{col name:region name} format
        self.regions_right = {}
        self.regions_left = {}
        self.regions_other = {}

        for c in ukbb_MRI.columns:
            curr_region = self.regions_dict[c]
            if 'Stem' in curr_region: 
                pass
            elif "right" in curr_region:
                curr_region = curr_region.replace('(right)','').strip()
                self.regions_right[curr_region] = c 
            elif "left" in curr_region:
                curr_region = curr_region.replace('(left)','').strip()
                self.regions_left[curr_region] = c
            else:
                self.regions_other[curr_region.strip()] = c

        #Check that all regions have entries for left and right sides
        self.regions_list = [region for region in self.regions_left.keys() & self.regions_right.keys()]
        # assert ukbb_Cereb.isna().sum().sum() == 0

        if clr_uneven == True:
            #Check for nans: make sure there are no columns where one region is nan but other is not. 
            for region in self.regions_list:
                #This map will give True if and only if one region side is Nan and the other is not
                nan_map = np.isnan(ukbb_MRI[self.regions_right[region]])!=np.isnan(ukbb_MRI[self.regions_left[region]])

                #use the map to set all the things where one or other isnt nan to be nan
                self.ukbb_MRI[self.regions_right[region]][nan_map] = np.nan
                self.ukbb_MRI[self.regions_left[region]][nan_map] = np.nan
    
    def one_side_perm(self, seed=0):
        #Returns an array of size ukbb(rows) X regions list
        # where each column has one brain region
        #with values coming from either the left or right brain
        np.random.seed(seed)
        #Change variables to not have self for ease
        ukbb_MRI = self.ukbb_MRI
        regions_list = self.regions_list

        #PERMUTUATION
        #for every column, randomly select the brain region from left or right side
        N_PPL = ukbb_MRI.shape[0]
        regions_rands = np.zeros((N_PPL, len(regions_list)))
        for ii, c in enumerate(regions_list):
            curr_combo =  np.vstack((ukbb_MRI[self.regions_left[c]] , 
                                        ukbb_MRI[self.regions_right[c]] )).T
            RL_mask = np.random.randint(2, size = (N_PPL))
            region_rand = np.vstack(
                        (curr_combo[:,1]*(RL_mask == 1).astype(int), 
                            curr_combo[:,0]*(RL_mask == 0).astype(int))
                        ).sum(axis = 0)
            regions_rands[:,ii] = region_rand
        return(regions_rands)

    def make_rep(self, expt_type, impute = False, return_df = True, ADD_NON_SYMM = False):
        expt_type = expt_type.replace(' ', '_')
        if expt_type not in self.expt_allowed:
            raise NameError(f"{expt_type} is not a valid representation space")
        
        #Change variables to not have self for ease
        ukbb_MRI = pd.DataFrame(np.nan_to_num(self.ukbb_MRI.values.copy()), 
                        columns = self.ukbb_MRI.columns)
        regions_list = self.regions_list

        if impute == True:
        #     #fully null row; sum since its a true/false array
        #     full_row = np.where((np.nan_to_num(ukbb_MRI).any(axis = 1)))[0]
        #     ukbb_full_vals = ukbb_MRI.values[full_row]
            ukbb_full_vals = self.ukbb_MRI.values

            #count of nans, negative, zeros
            n_z = (ukbb_full_vals == 0).sum()
            n_neg = (ukbb_full_vals < 0).sum()
            n_nan = np.isnan(ukbb_full_vals).sum()

            print(f"Now imputing out {n_z} zero-value and {n_neg} negative values, {n_nan} NaN values remain")
            ukbb_rep = impute_fncs.impute_zeros(ukbb_MRI.values, rem_neg = True)
            ukbb_MRI = pd.DataFrame(ukbb_rep, columns = ukbb_MRI.columns)

        print(f"Changing {self.type} data to {expt_type}")
        if expt_type == "Regular":
            ukbb_rep = ukbb_MRI.values
            #ukbb_dMRI_vol_sub = impute_zeros(ukbb_dMRI.values)
            headers = self.regions_all
        elif expt_type == "One_Side_Perm":
            ukbb_rep =  self.one_side_perm()
            # ukbb_dMRI_vol_sub =  one_side_perm(impute_zeros(ukbb_dMRI_no_nan))
            headers = regions_list
        elif expt_type in expt_type_dict.keys():
            ukbb_rep =  np.asarray(
                [expt_type_dict[expt_type](ukbb_MRI[self.regions_right[c]], ukbb_MRI[self.regions_left[c]]) 
                        for c in regions_list]
                        ).T
            headers = regions_list
            if ADD_NON_SYMM == True:
                #may have to double check this
                headers = regions_list + list(self.regions_other.keys())
                others =  ([ukbb_MRI[self.regions_other[c]].values for c in self.regions_other]).T
                ukbb_rep = np.concatenate([ukbb_rep, others], axis = 1)
        
        #remove empty rows
        dfX = pd.DataFrame(ukbb_rep, columns=headers)
        return dfX if return_df == True else (ukbb_rep, headers)

def change_rep(dfX, expt_type, preprocess = True):
    #Given a df containing symm brain structures
    #col names are the 'area (side)'
    #RETURNS: a dataframe with data in the form expt_type
    regions = dfX.columns
    expt_allowed = ["Regular", *list(expt_type_dict.keys()), "One_Side_Perm"]

    # split into L and R regions by making two dictionaries with 
    #{col name:region name} format
    regions_right = {}
    regions_left = {}
    regions_other = {}

    for curr_region in regions:
        if 'Stem' in curr_region: 
            pass
        elif "right" in curr_region:
            new_region = curr_region.replace('(right)','').strip()
            regions_right[new_region] = curr_region
        elif "left" in curr_region:
            new_region = curr_region.replace('(left)','').strip()
            regions_left[new_region] = curr_region
        else:
            regions_other[curr_region.strip()] = curr_region

    #Check that all regions have entries for left and right sides
    regions_list = [region for region in regions_left.keys() & regions_right.keys()]

    if not regions_list: #i.e. if regions list is empty
        raise ValueError("Passed df does not contain symmetrical values/column headers")

    expt_type = expt_type.replace(' ', '_')
    if expt_type not in expt_allowed:
        raise NameError(f"{expt_type} is not a valid representation space")

    if preprocess == True:
        #Check for nans: make sure there are no columns where one region is nan but other is not. 
        for region in regions_list:
            #This map will give True if and only if one region side is Nan and the other is not
            nan_map = np.isnan(dfX[regions_right[region]])!=np.isnan(dfX[regions_left[region]])

            #use the map to set all the things where one or other isnt nan to be nan
            dfX[regions_right[region]][nan_map] = np.nan
            dfX[regions_left[region]][nan_map] = np.nan
        
        #Deal with empty rows and nan/neg/zero values
        emp_rows = np.where((~np.nan_to_num(dfX).any(axis = 1)))[0]
        mask = ~np.bincount(emp_rows, minlength = dfX.shape[0]).astype(bool)
        dfX = pd.DataFrame(np.nan_to_num(dfX.values), 
                        columns = dfX.columns)

        #count of nans, negative, zeros
        n_z = (dfX.values == 0).sum()
        n_neg = (dfX.values < 0).sum()
        n_nan = np.isnan(dfX.values).sum()

        print(f"Now imputing out {n_z} zero-value and {n_neg} negative values, {n_nan} NaN values remain")
        ukbb_rep = impute_fncs.impute_zeros(dfX.values, rem_neg = True)
        dfX = pd.DataFrame(ukbb_rep[mask, :], columns = dfX.columns)

    print(f"Changing data to {expt_type}")
    if expt_type == "Regular":
        ukbb_rep = dfX.values
        #ukbb_dMRI_vol_sub = impute_zeros(ukbb_dMRI.values)
        headers = regions
    elif expt_type == "One_Side_Perm":
        ukbb_rep =  one_side_perm(dfX.values, regions_list, regions_left= regions_left, regions_right= regions_right)
        # ukbb_dMRI_vol_sub =  one_side_perm(impute_zeros(ukbb_dMRI_no_nan))
        headers = regions_list
    elif expt_type in expt_type_dict.keys():
        ukbb_rep =  np.asarray(
            [expt_type_dict[expt_type](dfX[regions_right[c]], dfX[regions_left[c]]) 
                    for c in regions_list]
                    ).T
        headers = regions_list
        
    dfX = pd.DataFrame(ukbb_rep, columns=headers)
    return dfX
            
