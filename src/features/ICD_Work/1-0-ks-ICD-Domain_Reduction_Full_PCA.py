


# I may be missing import statements, so add if you find something I missed
import os 
import pandas as pd
import numpy as np
# from tqdm import tqdm
from sklearn.decomposition import PCA
from itertools import compress

# CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
# os.chdir(CURR_PATH)
print("successfully loaded all modules")

# DATA_FOLDER = 'data/interim/icd10_work/full_data'
# OUT_FOLDER = 'data/interim/icd10_work/full_data/PCA_transformed'

DATA_FOLDER = '~/scratch/ICD'
OUT_FOLDER = '~/scratch/ICD/PCA_transformed'

# from notebooks.ICD10_Work import ks_1_0_MeDiWAS_fncs as MeDiWAS

#  Use this to load in the correct catergory to group number
# ppl_phecodes_small, ppl_xcldphe_small, phecodes_idx, groupid = MeDiWAS.load_ICD(DATA_FOLD = 'data/interim/icd10_work', useDefaults=True)

print("Loading Data")
# Load in full dataset
# ppl_xcldphe = pd.read_csv(os.path.join(DATA_FOLDER, 'exclusions_full.csv'), index_col = 0)
ppl_phecodes = pd.read_csv(os.path.join(DATA_FOLDER, 'inclusions_full.csv'), index_col = 0)
phecodes_idx = pd.read_csv(os.path.join(DATA_FOLDER, 'phecodes.csv'), index_col = 0)

all_phecodes = ppl_phecodes.columns.values
N_PPL = len(ppl_phecodes)

print('Extracting Columns by Category')
# group all columns by category
cat_cols = {}
all_cats = np.unique(phecodes_idx['group'])
for cat in all_cats:
# for cat in tqdm(all_cats):
    cat_cols[cat] = [col for col in all_phecodes 
                     if phecodes_idx.loc[col]['group'] == cat]


# We don't know how many PCs to extract
# so run permutations as well as real PCA
# And keep # of PCs which exceed the permutation VE
n_perms = 5
n_code = 4

dfReduced_dict = {'eid': ppl_phecodes['eid'].values}


# curr_cat = all_cats[0]
for curr_cat in all_cats:
    N_COL  = len(cat_cols[curr_cat])

    # Short hand code for the current category
    cat_code = curr_cat[0:n_code].upper()
    index_name = [cat_code+'-'+str(ii+1) for ii in range(N_COL)]

    dfX_real = ppl_phecodes[cat_cols[curr_cat]].copy()

    print("First PCA")
    PCA_real = PCA(n_components= N_COL, svd_solver = 'full')
    PCA_real.fit(dfX_real.values)

    # If we need to enforce a sign flip
    # Which we don't since we are only interested in explained variance
    # Which is not affected by sign flipping
    # cur_flip = np.abs(PCA_real.components_).min(0).argmax()
    '''
    In [650]: np.abs(PCA_mini.components_).min(0).shape
    Out[650]: (155,)

    In [651]: PCA_mini.components_.shape
    Out[651]: (15, 155)

    In [652]: np.abs(PCA_mini.components_).min(0).argmax()
    Out[652]: 36
    '''

    # PCA_mini = PCA(n_components= 15)
    # PCA_mini.fit(dfX_real.values)

    perm_comp_all = []
    perm_sv = []
    perm_xvar = []
    for i_perm in range(n_perms):
    # for i_perm in tqdm(range(n_perms)):
        dfX_perm = dfX_real.copy()

        print("Shuffling Data")
        # For this analysis we want to perturb every ROI separately
        # So we will have 184 random seeds
        for i_roi, roi in enumerate(dfX_perm.columns):
        # for i_roi, roi in tqdm(enumerate(dfX_perm.columns)):
            np.random.seed(i_perm**2 + i_roi)
            y_inds_perm = np.arange(0, N_PPL)
            np.random.shuffle(y_inds_perm)
            dfX_perm[roi] = dfX_real[roi].iloc[y_inds_perm].values

        # Now need PCA
        # As a reminder
        # TO convert from SV to explained var
        # eig = sv**2/(n-1)

        print("Conducting PCA")
        PCA_perm = PCA(n_components= N_COL)
        PCA_perm.fit(dfX_perm.values)

        perm_comp_all.append(PCA_perm.components_)
        perm_sv.append(PCA_perm.singular_values_)
        perm_xvar.append(PCA_perm.explained_variance_ratio_)


    perm_comp_all = np.asarray(perm_comp_all)
    perm_sv = np.asarray(perm_sv)
    perm_xvar = np.asarray(perm_xvar)

    # Save permutation results
    file_order = ['all_comps', 'singVal', 'ExpVar']
    for nm, fl in enumerate([perm_comp_all, perm_sv, perm_xvar]):
        np.save(os.path.join(OUT_FOLDER, 'Permutations', f"{cat_code}_Perm_{file_order[nm]}"), fl)
        
    # Select how many components to keep

    # ve_perm = PCA_perm.explained_variance_ratio_
    ve_perm = perm_xvar.mean(0)
    ve_real = PCA_real.explained_variance_ratio_


    pass_noise = ve_real > ve_perm
    n_pass = list(compress(np.arange(len(pass_noise)), pass_noise))
    n_pass = max(n_pass) + 1 # +1 to account for Zero index 

    # To reconstruct the PCA we need means, components, and explained variance
    # Keep the whole object and also the reduce version of it

    dfComp = pd.DataFrame(PCA_real.components_, columns=cat_cols[curr_cat], index=index_name)
    dfCompPass = pd.DataFrame(PCA_real.components_[0:n_pass], columns=cat_cols[curr_cat], index=index_name[0:n_pass])
    dfVE = pd.DataFrame(PCA_real.explained_variance_ratio_, index=index_name)
    dfMean = ppl_phecodes[cat_cols[curr_cat]].mean(0).copy()
    dfMean.name = 'mean'
    dfVE.name = 'exVar'

    dfTransformed = (dfX_real - dfMean)@dfCompPass.T

    # We need to save all these as csvs

    dfComp.to_csv(os.path.join(OUT_FOLDER, f'{cat_code}_PC_Components_All.csv'))
    dfCompPass.to_csv(os.path.join(OUT_FOLDER, f'{cat_code}_PC_Components_Sel.csv'))
    dfTransformed.to_csv(os.path.join(OUT_FOLDER, f'{cat_code}_Transformed_Sel.csv'))
    dfMean.to_csv(os.path.join(OUT_FOLDER, f'{cat_code}_means.csv'))
    dfVE.to_csv(os.path.join(OUT_FOLDER, f'{cat_code}_explained_var.csv'))

    # Add to our full dictionary
    for comp in dfTransformed.columns:
        dfReduced_dict[comp] = dfTransformed[comp].copy().values
        
# drop the column we put in to initiate the data
dfReduced = pd.DataFrame.from_dict(dfReduced_dict)
dfReduced = dfReduced.set_index('eid').sort_index()

dfReduced.to_csv(os.path.join(OUT_FOLDER, f'FULL_Transformed_Sel.csv'))
