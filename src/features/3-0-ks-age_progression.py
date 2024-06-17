#%%
import numpy as np
import os
# import nibabel as nib
import pandas as pd
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import (train_test_split, cross_val_score,
    KFold)
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr, scoreatpercentile

import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import SpectralClustering

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)

from processing.utilities import lighten_color

import  processing.impute_fncs as impute_fncs
import  processing.preprocess as preprocess

import importlib
# importlib.reload(preprocess)
# importlib.reload(impute_fncs)
import processing.utilities as utilities

DATA_FOLDER = 'data/raw'
OUT_FOLDER = 'data/interim/'
'''
Main loading of all data
Extraction of people from UKBB
Data Cleaning
Calculate asymmetry pattern
Calculate LBAC + MBAC
Calculate reference measure (total grey matter change)
'''
#%%
print("Checking for ukbb")
t1= time.time()
if 'ukbb' not in locals():
    print("UKBB not found, now loading")
    ukbb = pd.read_csv(os.path.join(DATA_FOLDER, 'ukb40500_cut_merged.csv'), low_memory = False)
ukbb.shape
t2=time.time()
print(f"Loading took  {(t2-t1)/60:.4f} min ")


descr_dict = joblib.load(os.path.join(DATA_FOLDER, 'descr_dict'))

#Load atlases which let us map a 3D space into brain regions
print("Loading other datasets")
from nilearn import datasets as ds
HO_atlas_cort = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm', symmetric_split=True)
HO_atlas_sub = ds.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm', symmetric_split=True)

#Grab Values for deconfounding
age = StandardScaler().fit_transform(ukbb['21022-0.0'].values[:, np.newaxis])  # Age at recruitment
agevi2 = StandardScaler().fit_transform(ukbb['21003-2.0'].values[:, np.newaxis])  # Age at recruitment
agevi3 = StandardScaler().fit_transform(ukbb['21003-3.0'].values[:, np.newaxis])  # Age at recruitment
#     age2 = age ** 2
sex = np.array(pd.get_dummies(ukbb['31-0.0']).values, dtype=int)  # Sex
sex_x_age = sex * age
# sex_x_age2 = sex * age2
# head_motion_rest = np.array(ukbb['25741-2.0'].values[inds_2myorder], dtype=np.float)  # Mean rfMRI head motion
# head_motion_task = np.nan_to_num(np.array(ukbb['25742-2.0'].values[inds_2myorder], dtype=np.float))  # Mean tfMRI head motion

# added during revision 1
head_size = StandardScaler().fit_transform(np.nan_to_num(ukbb['25006-2.0'].values[:, None]))  # Volume of grey matter
body_mass = StandardScaler().fit_transform(np.nan_to_num(ukbb['21001-0.0'].values[:, None]))  # BMI

fluid_iq = StandardScaler().fit_transform(np.nan_to_num(ukbb['20016-2.0'].values[:, None]))

NEW_CONF_MAT = True
if NEW_CONF_MAT == False:
    conf_mat = np.hstack([np.atleast_2d(head_size), np.atleast_2d(body_mass)])
else:
   # deconfound the brain space once - behavior one-by-one later
    from nilearn.signal import clean

    beh = ukbb

    age = StandardScaler().fit_transform(beh['21022-0.0'].values[:, np.newaxis])  # Age at recruitment
    age2 = age ** 2
    sex = np.array(pd.get_dummies(beh['31-0.0']).values, dtype=int)  # Sex
    sex_x_age = sex * age
    sex_x_age2 = sex * age2
    head_motion_rest = np.nan_to_num(beh['25741-2.0'].values)  # Mean rfMRI head motion
    head_motion_task = np.nan_to_num(beh['25742-2.0'].values)  # Mean tfMRI head motion

    # added during revision 1
    head_size = np.nan_to_num(beh['25006-2.0'].values)  # Volume of grey matter
    body_mass = np.nan_to_num(beh['21001-0.0'].values)  # BMI

    # motivated by Elliott et al., 2018
    head_pos_x = np.nan_to_num(beh['25756-2.0'].values)  # exact location of the head and the radio-frequency receiver coil in the scanner
    head_pos_y = np.nan_to_num(beh['25757-2.0'].values)
    head_pos_z = np.nan_to_num(beh['25758-2.0'].values)
    head_pos_table = np.nan_to_num(beh['25759-2.0'].values)
    scan_site_dummies = pd.get_dummies(beh['54-2.0']).values

    neurotic = np.nan_to_num(beh['20127-0.0'].values)

    # Genotype PCAs
    gene_col = [ii for ii in ukbb if ii.startswith('22009')]
    gene_PCA = np.nan_to_num(beh[gene_col].values)


    assert np.any(np.isnan(head_motion_rest)) == False
    assert np.any(np.isnan(head_motion_task)) == False
    assert np.any(np.isnan(head_size)) == False
    assert np.any(np.isnan(body_mass)) == False
    assert np.any(np.isnan(gene_PCA)) == False

    print('Deconfounding network connectivity space!')
    conf_mat = np.hstack([
        # age, age2, sex, sex_x_age, sex_x_age2,
        np.atleast_2d(head_motion_rest).T, np.atleast_2d(head_motion_task).T,
        np.atleast_2d(head_size).T, np.atleast_2d(body_mass).T,

        np.atleast_2d(head_pos_x).T, np.atleast_2d(head_pos_y).T,
        np.atleast_2d(head_pos_z).T, np.atleast_2d(head_pos_table).T,
        np.atleast_2d(scan_site_dummies)

        # , np.atleast_2d(gene_PCA)

        # np.atleast_2d(neurotic).T
        ])

    df_conf_mat = pd.DataFrame(conf_mat, index=ukbb['eid'].values)
    df_conf_mat.columns = ['head_motion_rest', 'head_motion_task', 'head_size', \
                           'body_mass', 'head_pos_x', 'head_pos_y', 'head_pos_z', \
                            'head_pos_table', 'scan_site_dummies_1', 'scan_site_dummies_2', \
                            'scan_site_dummies_3']
#%%
## Structural

#Grab columns that correspond to structural MRI data. We only want ones that are from the 2.0 timepoint
# ukbb_sMRI = ukbb.loc[:, '25782-2.0':'25920-2.0']  # FSL atlas including Diederichsen cerebellar atlas
ukbb_sMRI = ukbb.loc[:, '25782-2.0':'25892-2.0']  # FSL atlas without Diederichsen cerebellar atlas
colu =  [col for col in ukbb_sMRI if col.endswith("-2.0")]
ukbb_sMRI = ukbb_sMRI[colu]

#Split the names so it only contains area of interest (i.e. no superfluous data)
sMRI_vol_names = np.array([descr_dict[c]['descr'].split('Volume of grey matter in ')[1]
    for c in ukbb_sMRI.columns])

#Split the names so it only contains area of interest (i.e. no superfluous data)
sMRI_vol_dict = {c: descr_dict[c]['descr'].split('Volume of grey matter in ')[1]
    for c in ukbb_sMRI.columns}

sMRI = preprocess.MRI_processing(ukbb_sMRI, regions= sMRI_vol_dict, desc="sMRI")
print('\a')

fname = 'Cereb_target_UKB_IDs.txt'
fname = os.path.join(DATA_FOLDER, fname)
COLS_IDS = []
COLS_NAMES = []
with open(fname) as f:
    lines=f.readlines()
    f.close()
    for line in lines:
        # if "(R)" in line:
        #     COLS_NAMES.append(line.split('\t'))
        a = line[:line.find('\t')]
        b = line[line.find('\t') + 1:].rsplit('\n')[0]
        b = b.split('Volume of grey matter in ')[1].split('\tReg')[0]
        COLS_IDS.append(a + '-2.0')
        COLS_NAMES.append(b)
COLS_NAMES = np.array(COLS_NAMES)
COLS_IDS = np.array(COLS_IDS)
sub_dict = {COLS_IDS[i_col]: COLS_NAMES[i_col] for i_col in range(len(COLS_IDS))}

cereb = preprocess.MRI_processing(ukbb.loc[:, COLS_IDS], regions= sub_dict, desc="Cereb")
# (X_full, headers) = cereb.make_rep(expt_type_abbr, impute = False, return_df = False )
# dfX_vol_sub = gen_process(cereb, expt_type_abbr)
mri_type = cereb.type

# VISIT 3
COLS_IDS_v3 = [c.replace('-2.0', '-3.0') for c in COLS_IDS]
sub_dict_v3 = {COLS_IDS_v3[i_col]: COLS_NAMES[i_col] for i_col in range(len(COLS_IDS))}

cerebv3 = preprocess.MRI_processing(ukbb.loc[:, COLS_IDS_v3], regions= sub_dict_v3, desc="Cereb")


#Diffusion MRI
meas_map = {
    'FA' :['25056-2.0','25103-2.0'],
    'MD' :['25104-2.0','25151-2.0'],
    'MO' :['25152-2.0','25199-2.0'],
    'L1' :['25200-2.0','25247-2.0'],
    'L2' :['25248-2.0','25295-2.0'],
    'L3' :['25296-2.0','25343-2.0'],
    'ICVF' :['25344-2.0','25391-2.0'],
    'OD' :['25392-2.0','25439-2.0'],
    'ISOVF' :['25440-2.0','25487-2.0']
}

meas = "FA"
ukbb_dMRI = ukbb.loc[:, meas_map[meas][0]:meas_map[meas][1]]  # FSL atlas without Diederichsen cerebellar atlas
colu =  [col for col in ukbb_dMRI if col.endswith("-2.0")]
ukbb_dMRI = ukbb_dMRI[colu]

# print("{} has {} negative values".format( meas, len(np.where(ukbb_dMRI < 0 )[0])))

#Split the names so it only contains area of interest (i.e. no superfluous data)
dMRI_vol_names = np.array([descr_dict[c]['descr'].split(f'Mean {meas} in ')[1].replace(' on FA skeleton', '').strip()
    for c in ukbb_dMRI.columns])

#Split the names so it only contains area of interest (i.e. no superfluous data)
dMRI_vol_dict = {c: meas + ' ' + dMRI_vol_names[ii]
    for ii, c in enumerate(ukbb_dMRI.columns)}
dMRI_FA = preprocess.MRI_processing(ukbb_dMRI, regions= dMRI_vol_dict, desc= "dMRI " + meas)

# DMRI VISIT 3 ONLY
ukbb_dMRI = ukbb.loc[:, meas_map[meas][0]:meas_map[meas][1].replace("-2.0", "-3.0")]  # FSL atlas without Diederichsen cerebellar atlas
colu =  [col for col in ukbb_dMRI if col.endswith("-3.0")]
ukbb_dMRIv3 = ukbb_dMRI[colu]
dMRI_vol_dict_v3 =  {k.replace("-2.0", "-3.0"): v for k, v in dMRI_vol_dict.items()}
dMRI_FA_v3 = preprocess.MRI_processing(ukbb_dMRIv3.copy(), regions= dMRI_vol_dict_v3, desc= "dMRI " + meas + " visit 3")

#SMRI - Visit 3 only
#Grab columns that correspond to structural MRI data. We only want ones that are from the 2.0 timepoint
# ukbb_sMRI = ukbb.loc[:, '25782-2.0':'25920-2.0']  # FSL atlas including Diederichsen cerebellar atlas
ukbb_sMRIv3 = ukbb.loc[:, '25782-3.0':'25892-3.0']  # FSL atlas without Diederichsen cerebellar atlas
coluv3 =  [col for col in ukbb_sMRIv3 if col.endswith("-3.0")]
ukbb_sMRIv3 = ukbb_sMRIv3[coluv3]

#Split the names so it only contains area of interest (i.e. no superfluous data)
sMRI_vol_dictv3 = {c: descr_dict[c.replace('-3.0', '-2.0') ]['descr'].split('Volume of grey matter in ')[1]
    for c in ukbb_sMRIv3.columns}

sMRIvi3 = preprocess.MRI_processing(ukbb_sMRIv3, regions= sMRI_vol_dictv3, desc="sMRI vis3")

#Two visits
sMRI_2meas_mask = (sMRI.mask & sMRIvi3.mask)
sMRI_1meas_mask = (sMRI.mask & ~sMRIvi3.mask)
#%%
# I will use the means/patterns established by project 1

fullcomps_1vis_df = pd.read_csv('data/Project_1/Processed/df_Pattern_Feature_Contrib_All.csv', index_col = 0)
dfX_raw = pd.read_csv('data/Project_1/Processed/df_Lat_Raw.csv', index_col=0)
df_deconf_feature_coefs = pd.read_csv(os.path.join('models/Asymm_Patterns/Deconfounds', 'asymm_11_technical_deconf_coefs.csv'), index_col=0)

ROILI_means = dfX_raw.iloc[:,1:].mean(0)
ROILI_scales = dfX_raw.iloc[:,1:].std(0)

expt_type_abbr = 'Lat_Idx_Av'

print("\n Double Visits")
print("Visit 1")

#All three types in a row
mri_types = [sMRI, dMRI_FA, cereb]
ZSCORE_ON_SEX = False
mri_type = ''
X_all = []; headers_all = []
X_rgs = []; headers_rgs = []
for mri in mri_types:
    (X_full, headers) = mri.make_rep(expt_type_abbr, impute = True, return_df = False)
    (X_reg, headers_reg) = mri.make_rep("Regular", impute = True, return_df = False)
    # X = impute_fncs.impute(X_full)
    mri_type += mri.type + '/'

    # ukbb_dMRI_vol_sub = impute(ukbb_dMRI_vol_sub)
    X_all.append(X_full)
    headers_all.extend(headers)
    X_rgs.append(X_reg)
    headers_rgs.extend(headers_reg)

mri_type = mri_type[:-1]
X = np.concatenate([t.T for t in X_all]).T
X_reg = np.concatenate([t.T for t in X_rgs]).T
headers = headers_all

rows_2_rem = list(set(np.concatenate([mri.emp_rows for mri in mri_types])))
#remove empty rows
row_mask = ~np.bincount(rows_2_rem, minlength = X.shape[0]).astype(bool)

mask_orig = row_mask
mask_1vis = (row_mask & sMRI_1meas_mask)
mask_2vis = (row_mask & sMRI_2meas_mask)

row_mask = mask_2vis
rows_2_rem = np.where(~row_mask)[0]
N_PPL = sum(row_mask)

# X = X_everyone[row_mask, :]
X_raw = X.copy()
X = X[row_mask, :]

# First Z-scale the data
dfX_pre = pd.DataFrame(X, index=ukbb['eid'][row_mask].values, columns=headers)
dfX_pre2 = dfX_pre[ROILI_means.index] - np.tile(ROILI_means, (N_PPL, 1))
dfX_pre2 = dfX_pre2 / np.tile(ROILI_scales, (N_PPL, 1))

# Deconfound
# Use the coefficients to create an estimate of the brain region
# Then keep the residual
print("Deconfounding")
region_pred = df_deconf_feature_coefs.loc['intercept'] + df_conf_mat.dot(df_deconf_feature_coefs.drop(index='intercept'))
dfX_2vis_v1_orig = dfX_pre2[headers] - region_pred[headers][row_mask]
dfX_2vis_v1_orig = dfX_2vis_v1_orig[ROILI_means.index]

# Transform Data
print("Doing PCA")
dfX_new_2vis_v1 = dfX_2vis_v1_orig@fullcomps_1vis_df.T
print("Done!")

# Process Regular Data
dfX_2vis_v1_raw = pd.DataFrame(X_reg[row_mask, :], columns = headers_rgs)
dfX_2vis_v1_raw['eid'] = ukbb['eid'][row_mask].values
dfX_2vis_v1_raw = dfX_2vis_v1_raw.set_index('eid').sort_index()


X = X_raw[mask_orig, :]

# First Z-scale the data
dfX_pre = pd.DataFrame(X, index=ukbb['eid'][mask_orig].values, columns=headers)
dfX_pre2 = dfX_pre[ROILI_means.index] - np.tile(ROILI_means, (sum(mask_orig), 1))
dfX_pre2 = dfX_pre2 / np.tile(ROILI_scales, (sum(mask_orig), 1))

# Deconfound
# Use the coefficients to create an estimate of the brain region
# Then keep the residual
print("Deconfounding")
region_pred = df_deconf_feature_coefs.loc['intercept'] + df_conf_mat.dot(df_deconf_feature_coefs.drop(index='intercept'))
dfX_v1_orig = dfX_pre2[headers] - region_pred[headers][mask_orig]
dfX_v1_orig = dfX_v1_orig[ROILI_means.index]

# Transform Data
print("Doing PCA")
dfX_new_v1 = dfX_v1_orig@fullcomps_1vis_df.T
print("Done!")

# ---------------------------------
print("\nVisit 2")
#All three types in a row
mri_types = [sMRIvi3, dMRI_FA_v3, cerebv3]
ZSCORE_ON_SEX = False
mri_type = ''
X_all = []; headers_all = []
X_rgs = []; headers_rgs = []
for mri in mri_types:
    (X_full, headers) = mri.make_rep(expt_type_abbr, impute = True, return_df = False)
    (X_reg, headers_reg) = mri.make_rep("Regular", impute = True, return_df = False)
    # X = impute_fncs.impute(X_full)
    mri_type += mri.type + '/'

    # ukbb_dMRI_vol_sub = impute(ukbb_dMRI_vol_sub)
    X_all.append(X_full)
    headers_all.extend(headers)
    X_rgs.append(X_reg)
    headers_rgs.extend(headers_reg)

mri_type = mri_type[:-1]
X_v3 = np.concatenate([t.T for t in X_all]).T
X_reg = np.concatenate([t.T for t in X_rgs]).T
headers = headers_all

row_mask = mask_2vis
rows_2_rem = np.where(~row_mask)[0]

# Z-scale data First

dfX_2vis_v2_orig = pd.DataFrame(X_v3[row_mask, :], index=ukbb['eid'][row_mask].values, columns=headers)

dfX_2vis_v2_orig = dfX_2vis_v2_orig[ROILI_means.index] - np.tile(ROILI_means, (N_PPL, 1))
dfX_2vis_v2_orig = dfX_2vis_v2_orig / np.tile(ROILI_scales, (N_PPL, 1))

# Deconfound
# Use the coefficients to create an estimate of the brain region
# Then keep the residual
print("Deconfounding")
region_pred = df_deconf_feature_coefs.loc['intercept'] + df_conf_mat.dot(df_deconf_feature_coefs.drop(index='intercept'))
dfX_2vis_v2_decn = dfX_2vis_v2_orig[headers] - region_pred[headers][row_mask]
dfX_2vis_v2_decn = dfX_2vis_v2_decn[ROILI_means.index]

print("Doing PCA")
dfX_new_2vis_v2 = dfX_2vis_v2_decn@fullcomps_1vis_df.T
print("Done!")

# Process Regular Data
dfX_2vis_v2_raw = pd.DataFrame(X_reg[row_mask, :], columns = headers_rgs)
dfX_2vis_v2_raw['eid'] = ukbb['eid'][row_mask].values
dfX_2vis_v2_raw = dfX_2vis_v2_raw.set_index('eid').sort_index()


#%%

# MAKE CHANGE
agevi2_org = ukbb['21003-2.0'].values[:, np.newaxis]  # Age at recruitment
agevi3_org = ukbb['21003-3.0'].values[:, np.newaxis]  # Age at recruitment
age_diff = (agevi3_org - agevi2_org)[mask_2vis]


df_reg_change = (dfX_2vis_v2_orig - dfX_2vis_v1_orig).copy()
df_pca_change = (dfX_new_2vis_v2 - dfX_new_2vis_v1).copy()
df_amt_change = (dfX_new_2vis_v2 - dfX_new_2vis_v1).copy().abs()

#%%

# Extract Wait Time
import datetime
# NB check out column col 53 as another possible useful date variable
date_format = "%Y-%m-%dT%H:%M:%S"
v1datestr = ukbb['21862-2.0'][mask_2vis].values
v2datestr = ukbb['21862-3.0'][mask_2vis].values

date_2vis = ukbb[['eid','21862-2.0','21862-3.0', '21022-0.0','31-0.0']].merge(df_pca_chg_rate.reset_index(), how= 'right',right_on='eid', left_on = 'eid')

v1datestr = date_2vis['21862-2.0'].values
v2datestr = date_2vis['21862-3.0'].values

v1date = [datetime.datetime.strptime(t, date_format) for t in v1datestr]
v2date = [datetime.datetime.strptime(t, date_format) for t in v2datestr]

waittime_days = np.asarray([(v2date[ii] - v1date[ii]).days for ii in range(N_PPL)])
waittime_days = np.asarray([(v2date[ii] - v1date[ii]).days for ii in range(date_2vis.shape[0])])
waittime_orgd = waittime_days.copy()
waittime_SS = StandardScaler()
waittime_days = waittime_SS.fit_transform(waittime_days.reshape(-1,1))
df_waittime = pd.Series(waittime_orgd, index = df_pca_change.index, name='Days')

# Deconfound waittime from the change
df_pca_change_wt = pd.DataFrame(preprocess.deconf(conf_mat= waittime_days, X = df_pca_change, rows_2_rem = None),
                                columns = df_pca_change.columns, index = df_pca_change.index).sort_index()
# df_symm_shift_wt = pd.DataFrame(preprocess.deconf(conf_mat= waittime_days, X = df_symm_shift, rows_2_rem = None),
#                                 columns = df_symm_shift.columns, index = df_symm_shift.index).sort_index()
df_amt_change_wt = pd.DataFrame(preprocess.deconf(conf_mat= waittime_days, X = df_amt_change, rows_2_rem = None),
                                columns = df_amt_change.columns, index = df_amt_change.index).sort_index()

# Re-express everything as rate of change (in days)

N_DAYS_PER_YEAR = 365
df_reg_chg_rate = ((N_DAYS_PER_YEAR*df_reg_change).T.divide(waittime_orgd)).copy().T
df_pca_chg_rate = ((N_DAYS_PER_YEAR*df_pca_change).T.divide(waittime_orgd)).copy().T
df_amt_chg_rate = ((N_DAYS_PER_YEAR*df_amt_change).T.divide(waittime_orgd)).copy().T
# df_abs_sym_rate = ((N_DAYS_PER_YEAR*df_symm_shift).T.divide(waittime_orgd)).copy().T


#%%
###############################################################################################
# SAVE
################################################################################################%%
SAVE_FOLDER = 'models/Asymm_Patterns/Changes/2023_02_post_deconf'
SAVE = False
if SAVE == True:
    # Save Original 4
    #       Regional Change
    #       Asymm Change
    #       Amount of Asymm Change
    #       Absolute Change from Symmetry
    df_pca_change.to_csv(os.path.join(SAVE_FOLDER, 'Original', 'Asymm_Change.csv'))
    df_amt_change.to_csv(os.path.join(SAVE_FOLDER, 'Original', 'Amount_Change.csv'))

    # Save Calculated Asymm from just before Pattern compution
    dfX_2vis_v1_orig.to_csv(os.path.join(SAVE_FOLDER, 'Original', 'T1_Regional_Asymm.csv'))
    dfX_2vis_v2_decn.to_csv(os.path.join(SAVE_FOLDER, 'Original', 'T2_Regional_Asymm.csv'))

    # Save Rate-Adjusted 3
    df_pca_chg_rate.to_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Asymm_Change.csv'))
    df_amt_chg_rate.to_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Amount_Change.csv'))
    df_waittime.to_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Time_btwn_Visits_Days.csv'))

    # Save Raw
    dfX_2vis_v1_raw.to_csv(os.path.join(SAVE_FOLDER, 'Raw_Regional', 'T1_Regional_Sizes.csv'))
    dfX_2vis_v2_raw.to_csv(os.path.join(SAVE_FOLDER, 'Raw_Regional', 'T2_Regional_Sizes.csv'))
#%%

###############################################################################################
# Make directionality more understandable by relating to hemispheric grey matter decline
###############################################################################################

grey_matter_L_t1 = sMRI.make_rep("Vol_L_side", impute = True, return_df = True)
grey_matter_R_t1 = sMRI.make_rep("Vol_R_side", impute = True, return_df = True)
grey_matter_L_t2 = sMRIvi3.make_rep("Vol_L_side", impute = True, return_df = True)
grey_matter_R_t2 = sMRIvi3.make_rep("Vol_R_side", impute = True, return_df = True)

rg_order = grey_matter_R_t2.columns
grey_matter_L_chg = grey_matter_L_t2[rg_order][row_mask] - grey_matter_L_t1[rg_order][row_mask]
grey_matter_R_chg = grey_matter_R_t2[rg_order][row_mask] - grey_matter_R_t1[rg_order][row_mask]

cort = HO_atlas_cort['labels']
cort = [sub.replace('Left', '').replace('Right', '').replace("Accumbens", "Ventral Striatum").strip()
            for sub in cort]
cort = list(set(cort))
cort.remove('Background')

grey_matter_L_rtc = ((N_DAYS_PER_YEAR*grey_matter_L_chg).T.divide(waittime_orgd)).copy().T
grey_matter_R_rtc = ((N_DAYS_PER_YEAR*grey_matter_R_chg).T.divide(waittime_orgd)).copy().T

grey_matter_hemi_chg = grey_matter_R_chg[cort].sum(1) - grey_matter_L_chg[cort].sum(1)
grey_matter_hemi_rtc = ((N_DAYS_PER_YEAR*grey_matter_hemi_chg).T.divide(waittime_orgd)).copy().T

# For me:
# How related are regional declines for a particular region in left vs right
region_change_rel = {rg:{} for rg in rg_order}
for rg in rg_order:
    r, p = pearsonr(grey_matter_R_rtc[rg], grey_matter_L_rtc[rg])
    n_R_fast = sum(grey_matter_R_rtc[rg] > grey_matter_L_rtc[rg])
    region_change_rel[rg]['r'] = r
    region_change_rel[rg]['p'] = p
    region_change_rel[rg]['n_right'] = n_R_fast
region_change_rel = pd.DataFrame.from_dict(region_change_rel).T

# Positive value means:
#   Right Hemi got bigger faster than the L hemi
#   Left Hemi got smaller faster than the R hemi

pearsonr(grey_matter_R_chg.sum(1), df_pca_chg_rate['Pattern 1'])
pearsonr(grey_matter_R_chg.sum(1), grey_matter_L_chg.sum(1))
ttest_rel(grey_matter_R_chg.sum(1), grey_matter_L_chg.sum(1))

pearsonr(grey_matter_hemi_rtc, df_pca_chg_rate['Pattern 1'])

'''
In [1671]: pearsonr(grey_matter_hemi_chg, df_pca_chg_rate['Pattern 1'])
Out[1671]: (-0.20669050579989717, 3.2530263502894727e-15)

Negative association between pattern 1 expression change and volume difference asymmetry means:
A positive shift in Pattern 1 corresponds to a Negative shift in the hemisphere change
Positive shift in Pattern 1 means that the R hemi got smaller faster than the L hemi
(or that if they both grew the rate of growth is smaller in the R hemi)
'''
labels = [fullcomps_1vis_df.index[0].split(' ')[0] + ' '+str(i+1) for i in range(85)]

# For flipping:
# How related are pattern changes to
pattern_change_rel = {ptrn:{} for  ptrn in labels[0:33]}
for ptrn in labels[0:33]:
    r, p = pearsonr(grey_matter_hemi_rtc.values, df_pca_chg_rate[ptrn].values)
    pattern_change_rel[ptrn]['r'] = r
    pattern_change_rel[ptrn]['p'] = p
pattern_change_rel = pd.DataFrame.from_dict(pattern_change_rel).T

### Graphic
# Rate of Change

#####

grey_matter_L_t1_hemi = grey_matter_L_t1[cort][row_mask].sum(1)
grey_matter_R_t1_hemi = grey_matter_R_t1[cort][row_mask].sum(1)
grey_matter_L_t2_hemi = grey_matter_L_t2[cort][row_mask].sum(1)
grey_matter_R_t2_hemi = grey_matter_R_t2[cort][row_mask].sum(1)

grey_cort = pd.concat([grey_matter_L_t1_hemi, grey_matter_R_t1_hemi, \
                       grey_matter_L_t2_hemi, grey_matter_R_t2_hemi],
                       keys=['L_T1', 'R_T1', "L_T2", "R_T2"], axis=1)

grey_cort_melt = grey_cort.melt()
grey_cort_melt['hemisphere'] = grey_cort_melt['variable'].map(lambda x:x.split('_')[0])
grey_cort_melt['time'] = grey_cort_melt['variable'].map(lambda x:x.split('_T')[1])

fig, ax = plt.subplots()
sns.violinplot(data=grey_cort_melt,
               hue="hemisphere", y='value', x='time',
               scale='width', palette='spring')
# ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
plt.xlabel('')
plt.ylabel('Adjusted $R^2$ Score ($\pm$ Bootstrap st. dev.)')
plt.axhline(0, c='k')
plt.tight_layout()
ax.legend(title='Predictor Variables')

# Bar plot of the rate of change of each hemisphere
grey_matter_R_pct = grey_matter_R_rtc/grey_matter_R_t1[cort][row_mask]
grey_matter_L_pct = grey_matter_L_rtc/grey_matter_L_t1[cort][row_mask]

labels = ['Left', 'Right']
means = [grey_matter_L_rtc.sum(1).mean(), grey_matter_R_rtc.sum(1).mean()]
stds = [grey_matter_L_rtc.sum(1).std()/np.sqrt(N_PPL), grey_matter_R_rtc.sum(1).std()/np.sqrt(N_PPL)]

means = [grey_matter_L_pct.sum(1).mean(), grey_matter_R_pct.sum(1).mean()]
stds = [grey_matter_L_pct.sum(1).std()/np.sqrt(N_PPL), grey_matter_R_pct.sum(1).std()/np.sqrt(N_PPL)]

# Width of the bars
bar_width = 0.35
x = np.arange(2)
# Create subplots for means and standard deviations
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - bar_width/2, means, bar_width, label='Mean', color='b', alpha=0.7, yerr=stds, capsize=5)
ax.set_ylabel('Cortical Grey Matter Volume Change (mm3; yearly rate)')
ax.set_xlabel('Hemisphere')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)

###############################################################################################
# Include region change for cereb and whilte matter
###############################################################################################


white_matter_L_t1 = dMRI_FA.make_rep("Vol_L_side", impute = True, return_df = True)
white_matter_R_t1 = dMRI_FA.make_rep("Vol_R_side", impute = True, return_df = True)
white_matter_L_t2 = dMRI_FA_v3.make_rep("Vol_L_side", impute = True, return_df = True)
white_matter_R_t2 = dMRI_FA_v3.make_rep("Vol_R_side", impute = True, return_df = True)

rg_order_wm = white_matter_R_t2.columns
white_matter_L_chg = white_matter_L_t2[rg_order_wm][row_mask] - white_matter_L_t1[rg_order_wm][row_mask]
white_matter_R_chg = white_matter_R_t2[rg_order_wm][row_mask] - white_matter_R_t1[rg_order_wm][row_mask]

white_matter_L_rtc = ((N_DAYS_PER_YEAR*white_matter_L_chg).T.divide(waittime_orgd)).copy().T
white_matter_R_rtc = ((N_DAYS_PER_YEAR*white_matter_R_chg).T.divide(waittime_orgd)).copy().T

white_matter_hemi_chg = white_matter_R_chg[rg_order_wm].sum(1) - white_matter_L_chg[rg_order_wm].sum(1)
white_matter_hemi_rtc = ((N_DAYS_PER_YEAR*white_matter_hemi_chg).T.divide(waittime_orgd)).copy().T

# For me:
# How related are regional declines for a particular region in left vs right
region_chg_rel_wm = {rg:{} for rg in rg_order_wm}
for rg in rg_order_wm:
    r, p = pearsonr(white_matter_R_rtc[rg], white_matter_L_rtc[rg])
    n_R_fast = sum(white_matter_R_rtc[rg] > white_matter_L_rtc[rg])
    region_chg_rel_wm[rg]['r'] = r
    region_chg_rel_wm[rg]['p'] = p
    region_chg_rel_wm[rg]['n_right'] = n_R_fast
region_chg_rel_wm = pd.DataFrame.from_dict(region_chg_rel_wm).T


cereb_matter_L_t1 = cereb.make_rep("Vol_L_side", impute = True, return_df = True)
cereb_matter_R_t1 = cereb.make_rep("Vol_R_side", impute = True, return_df = True)
cereb_matter_L_t2 = cerebv3.make_rep("Vol_L_side", impute = True, return_df = True)
cereb_matter_R_t2 = cerebv3.make_rep("Vol_R_side", impute = True, return_df = True)

rg_order_cb = cereb_matter_R_t2.columns
cereb_matter_L_chg = cereb_matter_L_t2[rg_order_cb][row_mask] - cereb_matter_L_t1[rg_order_cb][row_mask]
cereb_matter_R_chg = cereb_matter_R_t2[rg_order_cb][row_mask] - cereb_matter_R_t1[rg_order_cb][row_mask]

cereb_matter_L_rtc = ((N_DAYS_PER_YEAR*cereb_matter_L_chg).T.divide(waittime_orgd)).copy().T
cereb_matter_R_rtc = ((N_DAYS_PER_YEAR*cereb_matter_R_chg).T.divide(waittime_orgd)).copy().T

cereb_matter_hemi_chg = cereb_matter_R_chg[rg_order_cb].sum(1) - cereb_matter_L_chg[rg_order_cb].sum(1)
cereb_matter_hemi_rtc = ((N_DAYS_PER_YEAR*cereb_matter_hemi_chg).T.divide(waittime_orgd)).copy().T

# For me:
# How related are regional declines for a particular region in left vs right
region_chg_rel_cb = {rg:{} for rg in rg_order_cb}
for rg in rg_order_cb:
    r, p = pearsonr(cereb_matter_R_rtc[rg], cereb_matter_L_rtc[rg])
    n_R_fast = sum(cereb_matter_R_rtc[rg] > cereb_matter_L_rtc[rg])
    region_chg_rel_cb[rg]['r'] = r
    region_chg_rel_cb[rg]['p'] = p
    region_chg_rel_cb[rg]['n_right'] = n_R_fast
region_chg_rel_cb = pd.DataFrame.from_dict(region_chg_rel_cb).T

pd.concat([region_chg_rel_cb, region_chg_rel_wm, region_change_rel])