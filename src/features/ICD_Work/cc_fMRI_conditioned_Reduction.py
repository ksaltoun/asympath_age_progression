# I may be missing import statements, so add if you find something I missed
import os 
import pandas as pd
import numpy as np
# from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from itertools import compress
import time

print("successfully loaded all modules")


DATA_FOLDER = '~/scratch/ICD'
fMRI_FOLDER = '~/scratch/ICD_fMRI'
OUT_FOLDER = '~/scratch/ICD_fMRI/fMRI_informed2'
OUTERMOST = OUT_FOLDER.split('/')[-1]


##############################################################################################
print("Loading Data")
tic = time.time()
# Load in full dataset
# ppl_xcldphe = pd.read_csv(os.path.join(DATA_FOLDER, 'exclusions_full.csv'), index_col = 0)
ppl_phecodes = pd.read_csv(os.path.join(DATA_FOLDER, 'inclusions_full.csv'), index_col = 0)
phecodes_idx = pd.read_csv(os.path.join(DATA_FOLDER, 'phecodes.csv'), index_col = 0)

all_phecodes = ppl_phecodes.columns.values
N_PPL = len(ppl_phecodes)
print("Loaded in data")
toc = time.time()
print(f"Loading took {toc-tic:.2f}s") 
#print(f"Main dataframe memory usage: {ppl_phecodes.memory_usage(index=True).sum():,.0f} B")
print('\n', ppl_phecodes.shape)

GEN_FOLDER = os.path.join(OUTERMOST, 'General')
if not os.path.isdir(GEN_FOLDER):
   os.makedirs(GEN_FOLDER)

# # Standard Scale ICD
# print("Standard Scaling ICD Data")
# toc = time.time()
# scaler_ICD = StandardScaler()
# ICD_scl = scaler_ICD.fit_transform(ppl_phecodes)
# dfMean_ICD = pd.DataFrame([scaler_ICD.mean_, scaler_ICD.scale_], columns=ppl_phecodes.columns, index=['Mean', 'Scale'])
# dfMean_ICD.to_csv(os.path.join(GEN_FOLDER, f'ICD_means.csv'))
# ppl_icdscale = pd.DataFrame(ICD_scl, columns = ppl_phecodes.columns, index=ppl_phecodes.index)
# ppl_icdscale.to_csv(os.path.join(GEN_FOLDER, f'ICD_Z_scored.csv'))

# toc = time.time()
# print(f"Standard Scaling took {toc-tic:.2f}s")

# Standard Scaling Takes a long time
# So load in the pre-computed values
ppl_icdscale =pd.read_csv(os.path.join(GEN_FOLDER, f'ICD_Z_scored.csv'), index_col = 0)


print('Extracting Columns by Category')
# group all columns by category
tic = time.time()
cat_cols = {}
all_cats = np.unique(phecodes_idx['group'])
for cat in all_cats:
# for cat in tqdm(all_cats):
    cat_cols[cat] = [col for col in all_phecodes 
                     if phecodes_idx.loc[float(col)]['group'] == cat]
toc = time.time()
print(f"Column categorization took {toc-tic:.2f}s")

##############################################################################################
# Import fMRI Data
print("Loading fMRI")
tic = time.time()
fMRI_full = pd.read_csv(os.path.join(fMRI_FOLDER, 'fMRI_all.csv'), index_col = 0)
# fMRI_full = fMRI_full.set_index('eid').sort_index()
toc = time.time()
print(f"Loading fMRI took {toc-tic:.2f}s")

# Standard Scale fMRI
scaler_fMRI = StandardScaler()
fMRI_scl = scaler_fMRI.fit_transform(fMRI_full)
dfMean_fMRI = pd.DataFrame([scaler_fMRI.mean_, scaler_fMRI.scale_], columns=fMRI_full.columns, index=['Mean', 'Scale'])

# Reduce the PCA
n_fm = 100
fIndex = [f"fMRI_PC{i+1}" for i in range(n_fm)]

PCA_fMRI = PCA(n_components= n_fm)
fMRI_red = PCA_fMRI.fit_transform(fMRI_scl)

dfComp = pd.DataFrame(PCA_fMRI.components_, columns=fMRI_full.columns, index=fIndex)
dfVE = pd.DataFrame(PCA_fMRI.explained_variance_ratio_, index=fIndex)
dfVE.name = 'exVar'

# dfMRI_pc = (fMRI_red - dfMean_fMRI['Mean'])@dfComp.T

# assert dfMRI_pc.values == fMRI_red, " Homemade transform does not match the result of PCA transform"
dfMRI_pc = pd.DataFrame(fMRI_red, columns = fIndex, index=fMRI_full.index)


# We need to save all these as csvs
print("Saving fMRI PCA")
tic=time.time()
dfComp.to_csv(os.path.join(GEN_FOLDER,   f'fMRI_{n_fm}_PC_Components.csv'))
dfMRI_pc.to_csv(os.path.join(GEN_FOLDER, f'fMRI_{n_fm}_PC_Transformed.csv'))
dfMean_fMRI.to_csv(os.path.join(GEN_FOLDER, f'fMRI_means.csv'))
dfVE.to_csv(os.path.join(GEN_FOLDER, f'fMRI_{n_fm}_PC_explained_var.csv'))
toc = time.time()
print(f"Saving fMRI PCA took {toc-tic:.2f}s")

##############################################################################################

# Before we  do the CCA/ PLS of the data
# Some more preprocessing is needed

# Grab all the people within the fMRI dataset
# And make a reduced ppl_phecodes with them in it

tic = time.time()
print("Reducing ppl_phecodes to only relevant people")
ppl_icdsmall = ppl_icdscale[ppl_icdscale.index.isin(fMRI_full.index.values)].copy()
ppl_icdsmall = ppl_icdsmall.loc[fMRI_full.index.values]
toc = time.time()
print(f"Loading took {toc-tic:.2f}s") 

n_code = 4

# pc_summary.to_csv(os.path.join('data/processed/ICD_work/PCA_Pipeline/PCA_transformed2', 'n_comps_kept.csv'))

# this will guide how many cca/pls components to keep
pc_summary = pd.read_csv(os.path.join(fMRI_FOLDER, 'n_comps_kept.csv'), index_col = 0)


##############################################################################################

# Here we will do the PLS of the data


PLS_FOLDER = os.path.join(OUTERMOST, 'PLS')
if not os.path.isdir(PLS_FOLDER):
   os.makedirs(PLS_FOLDER)

dfReduced_dict = {'eid': ppl_phecodes.index.values}
dfPearson = {key:[] for key in ['Code', 'r', 'p']}

# Run CCA/PLS
for curr_cat in all_cats:
    N_COL  = len(cat_cols[curr_cat])
    print('\n')

    # Short hand code for the current category
    cat_code = curr_cat[0:n_code].upper()
    print(curr_cat, '\nShorthand: ', cat_code, '\nn columns: ', N_COL)

    n_c_perm = pc_summary.loc[cat_code]['N_Red_Alt']
    n_c_ve50 = pc_summary.loc[cat_code]['N_Red_50']

    tic = time.time()
    dfX_real = ppl_icdsmall[cat_cols[curr_cat]].copy()
    dfX_full = ppl_icdscale[cat_cols[curr_cat]].copy()
    toc = time.time()
    print(f"Extracting current dataset took {toc-tic:.2f}s")


    # n_k = min(N_COL, n_fm)
    n_k = max(n_c_perm, n_c_ve50)
    index_name = [cat_code+'-'+str(ii+1) for ii in range(n_k)]

    print("First PLS")
    tic=time.time()
    plsca_chg = PLSCanonical(n_components=n_k, scale=False)
    X_chg, Y_chg = plsca_chg.fit_transform(dfX_real, dfMRI_pc)
    toc = time.time()
    print(f"Real PLS took {toc-tic:.2f}s")
    
    '''
    In [650]: np.abs(PCA_mini.components_).min(0).shape
    Out[650]: (155,)

    In [651]: PCA_mini.components_.shape
    Out[651]: (15, 155)

    In [652]: np.abs(PCA_mini.components_).min(0).argmax()
    Out[652]: 36
    '''

    # We now decide what to save
    # I want to keep
    #   [x] fMRI Loadings 
    #   [x] ICD Loadings 
    #   [x] ICD Rotations 
    #   [x] ICD Transformed
    #   [x] ICD-fMRI Correlation 

    # Add in the correlation of ICD - fMRI combined data
    pears = np.array([pearsonr(X_chg[:,lk], Y_chg[:, lk]) for lk in range(n_k)])
    rs, ps = pears[:, 0], pears[:,1]
    for ii, id in enumerate(index_name):
        dfPearson['Code'].append(id)
        dfPearson['r'].append(rs[ii])
        dfPearson['p'].append(ps[ii])

    # To reconstruct the PCA we need means, components, and explained variance
    # Keep the whole object and also the reduce version of it

    dfMRI_loading = pd.DataFrame(plsca_chg.y_loadings_, index=fIndex, columns=index_name)
    dfICD_loading = pd.DataFrame(plsca_chg.x_loadings_, index=cat_cols[curr_cat], columns=index_name)
    dfICD_rotation = pd.DataFrame(plsca_chg.x_rotations_, index=cat_cols[curr_cat], columns=index_name)

    # Transform Data
    pls_trans = plsca_chg.transform(dfX_full)
    dfTransformed = pd.DataFrame(pls_trans, columns = index_name, index=dfX_full.index)

    # We need to save all these as csvs
    print("Saving Real PCA")
    tic=time.time()
    dfMRI_loading.to_csv(os.path.join(PLS_FOLDER, f'{cat_code}_fMRI_loadings.csv'))
    dfICD_loading.to_csv(os.path.join(PLS_FOLDER, f'{cat_code}_ICD_loadings.csv'))
    dfICD_rotation.to_csv(os.path.join(PLS_FOLDER, f'{cat_code}_ICD_rotation.csv'))
    toc = time.time()
    print(f"Saving Real PCA took {toc-tic:.2f}s")

    # Add to our full dictionary
    for comp in dfTransformed.columns:
        dfReduced_dict[comp] = dfTransformed[comp].copy().values
        
dfPearson = pd.DataFrame.from_dict(dfPearson)
dfPearson = dfPearson.set_index('Code').sort_index()

dfReduced = pd.DataFrame.from_dict(dfReduced_dict)
dfReduced = dfReduced.set_index('eid').sort_index()

dfReduced.to_csv(os.path.join(OUT_FOLDER, f'PLS_Transformed.csv'))
dfPearson.to_csv(os.path.join(OUT_FOLDER, f'PLS_Correlation.csv'))


##############################################################################################

# Here we will do the PLS of the data


CCA_FOLDER = os.path.join(OUTERMOST, 'CCA')
if not os.path.isdir(CCA_FOLDER):
   os.makedirs(CCA_FOLDER)

dfReduced_dict = {'eid': ppl_phecodes.index.values}
dfPearson = {key:[] for key in ['Code', 'r', 'p']}

# Run CCA/PLS
for curr_cat in all_cats:
    N_COL  = len(cat_cols[curr_cat])
    print('\n')

    # Short hand code for the current category
    cat_code = curr_cat[0:n_code].upper()
    print(curr_cat, '\nShorthand: ', cat_code, '\nn columns: ', N_COL)

    n_c_perm = pc_summary.loc[cat_code]['N_Red_Alt']
    n_c_ve50 = pc_summary.loc[cat_code]['N_Red_50']

    tic = time.time()
    dfX_real = ppl_icdsmall[cat_cols[curr_cat]].copy()
    dfX_full = ppl_icdscale[cat_cols[curr_cat]].copy()
    toc = time.time()
    print(f"Extracting current dataset took {toc-tic:.2f}s")


    # n_k = min(N_COL, n_fm)
    n_k = max(n_c_perm, n_c_ve50)
    index_name = [cat_code+'-'+str(ii+1) for ii in range(n_k)]

    print("First PLS")
    tic=time.time()
    plsca_chg = CCA(n_components=n_k, scale=False)
    X_chg, Y_chg = plsca_chg.fit_transform(dfX_real, dfMRI_pc)
    toc = time.time()
    print(f"Real PLS took {toc-tic:.2f}s")
    
    '''
    In [650]: np.abs(PCA_mini.components_).min(0).shape
    Out[650]: (155,)

    In [651]: PCA_mini.components_.shape
    Out[651]: (15, 155)

    In [652]: np.abs(PCA_mini.components_).min(0).argmax()
    Out[652]: 36
    '''

    # We now decide what to save
    # I want to keep
    #   [x] fMRI Loadings 
    #   [x] ICD Loadings 
    #   [x] ICD Rotations 
    #   [x] ICD Transformed
    #   [x] ICD-fMRI Correlation 

    # Add in the correlation of ICD - fMRI combined data
    pears = np.array([pearsonr(X_chg[:,lk], Y_chg[:, lk]) for lk in range(n_k)])
    rs, ps = pears[:, 0], pears[:,1]
    for ii, id in enumerate(index_name):
        dfPearson['Code'].append(id)
        dfPearson['r'].append(rs[ii])
        dfPearson['p'].append(ps[ii])

    # To reconstruct the PCA we need means, components, and explained variance
    # Keep the whole object and also the reduce version of it

    dfMRI_loading = pd.DataFrame(plsca_chg.y_loadings_, index=fIndex, columns=index_name)
    dfICD_loading = pd.DataFrame(plsca_chg.x_loadings_, index=cat_cols[curr_cat], columns=index_name)
    dfICD_rotation = pd.DataFrame(plsca_chg.x_rotations_, index=cat_cols[curr_cat], columns=index_name)

    # Transform Data
    pls_trans = plsca_chg.transform(dfX_full)
    dfTransformed = pd.DataFrame(pls_trans, columns = index_name, index=dfX_full.index)

    # We need to save all these as csvs
    print("Saving Real PCA")
    tic=time.time()
    dfMRI_loading.to_csv(os.path.join(CCA_FOLDER, f'{cat_code}_fMRI_loadings.csv'))
    dfICD_loading.to_csv(os.path.join(CCA_FOLDER, f'{cat_code}_ICD_loadings.csv'))
    dfICD_rotation.to_csv(os.path.join(CCA_FOLDER, f'{cat_code}_ICD_rotation.csv'))
    toc = time.time()
    print(f"Saving Real PCA took {toc-tic:.2f}s")

    # Add to our full dictionary
    for comp in dfTransformed.columns:
        dfReduced_dict[comp] = dfTransformed[comp].copy().values
        
dfPearson = pd.DataFrame.from_dict(dfPearson)
dfPearson = dfPearson.set_index('Code').sort_index()

dfReduced = pd.DataFrame.from_dict(dfReduced_dict)
dfReduced = dfReduced.set_index('eid').sort_index()

dfReduced.to_csv(os.path.join(OUT_FOLDER, f'CCA_Transformed.csv'))
dfPearson.to_csv(os.path.join(OUT_FOLDER, f'CCA_Correlation.csv'))
