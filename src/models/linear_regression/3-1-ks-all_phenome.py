
import processing.manhattan_plot_util as man_plot
import processing.impute_fncs as impute_fncs
from scipy.stats import scoreatpercentile
import pandas as pd
import os
from tqdm import tqdm

'''
Predict phenotype changes from LBACs (or MBACs)

Repeat code block with changing MAIN_FOLDER to keep track of different iterations
Subtract out results from a age/sex model only from results when LBAC/MBAC are included in model

Visualization included
'''

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

SAVE_FOLDER = 'models/Asymm_Patterns/Changes/2023_02_post_deconf'

# These will be used for comparison much later
df_pca_change = pd.read_csv(os.path.join(SAVE_FOLDER, 'Original', 'Asymm_Change.csv'), index_col=0)
df_amt_change = pd.read_csv(os.path.join(SAVE_FOLDER, 'Original', 'Amount_Change.csv'), index_col=0)
df_pca_chg_rate = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Asymm_Change.csv'), index_col=0)
df_amt_chg_rate = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Amount_Change.csv'), index_col=0)
pattern_change_rel = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Change_rel_deltaTBV.csv'), index_col=0)
pos_left_hemi_faster = np.sign(pattern_change_rel['r'])
N_PPL = df_pca_chg_rate.shape[0]

meta_df = pd.read_csv('data/Project_1/Processed/df_meta.csv', index_col=0)

meta_df_idx = meta_df.set_index('eid').sort_index()

# load in 977 phenotypes
ukbb_y, y_desc_dict, y_cat_dict = man_plot.load_phenom(BASE_FOLDER = 'data/processed/')


LOAD = True
if LOAD == True:
    print('Loading Phenome')
    # df_pheno_change = pd.read_csv('data/processed/ukbb_phenotype_change.csv')
    df_pheno_change = pd.read_csv('data/processed/2309_Investigate_double/post_Phesant/ukbb_y_2vis_subtract_final.csv')


    if '3089' in df_pheno_change.columns:
        df_pheno_change = df_pheno_change.drop(columns='3089')

    change_desc = pd.read_csv('data/processed/ukbb_mh_miller_v3_description.txt',
                                sep='\t', header=None, index_col=0).to_dict()[1]

    def find_phenome(col):
        col = str(col)
        col_v = col.split('_')[0].split('#')[0].split('-')[0]
        col_v = [ii for ii in y_cat_dict.index
                    if ii.startswith(col_v + '-')][0]
        return(col_v)

def adjusted_R2_v2(r2, d, n):
    # n =number variables
    # d = number of parameters
    # y_true; y_pred = true and predicted y value

    RSS_P = 1 - r2
    R2 = 1-RSS_P
    Adj_R2 = 1-RSS_P*(n-1)/(n-d-1)
    return R2, Adj_R2

import datetime
date_format = "%Y-%m-%dT%H:%M:%S"
v1datestr = ukbb.set_index('eid')['21862-2.0'].loc[df_pca_chg_rate.index].values
v2datestr = ukbb.set_index('eid')['21862-3.0'].loc[df_pca_chg_rate.index].values

v1date = [datetime.datetime.strptime(t, date_format) for t in v1datestr]
v2date = [datetime.datetime.strptime(t, date_format) for t in v2datestr]

waittime_days_org = np.asarray([(v2date[ii] - v1date[ii]).days for ii in range(df_pca_chg_rate.shape[0])])
waittime_SS = StandardScaler()
waittime_days = waittime_SS.fit_transform(waittime_days_org.reshape(-1,1))

ptrn_keep = 33
ptrn_keep = ['Pattern '+str(i+1) for i in range(ptrn_keep)]

# Include Rate of Change and Amount of Change
dfX = (pos_left_hemi_faster*df_pca_chg_rate[ptrn_keep]).copy()
for col in ptrn_keep:
    dfX[f'Amount {col}'] = df_amt_chg_rate[col]

# Original
# dfX = df_pca_change[ptrn_keep].copy()
# dfX.set_index('eid', inplace = True)
dfX['Waittime'] = waittime_days_org
dfX['Age_v1'] = meta_df_idx.loc[df_pca_chg_rate.index]['age_v2']
dfX['Age2'] = dfX['Age2']**2
dfX['Female'] = meta_df_idx.loc[df_pca_chg_rate.index]['sex_F']


# Normalize Everything
X_SS = StandardScaler(with_std=False)
dfX_org = dfX.copy()
dfX = pd.DataFrame(X_SS.fit_transform(dfX), columns = dfX.columns, index=dfX.index)

dfX_perm = dfX.copy()
for i_roi, roi in enumerate(dfX_perm.columns):
# for i_roi, roi in tqdm(enumerate(dfX_perm.columns)):
    if roi in ['Female', 'Age_v1', 'Waittime']:
        continue
    np.random.seed( i_roi**2)
    y_inds_perm = np.arange(0, N_PPL)
    np.random.shuffle(y_inds_perm)
    dfX_perm[roi] = dfX_perm[roi].iloc[y_inds_perm].values.copy()

Y_SS = StandardScaler(with_std=False)
ukbb_y_idx =  df_pheno_change.set_index('userID').sort_index()
dfY_normed = pd.DataFrame(Y_SS.fit_transform(ukbb_y_idx.loc[df_pca_chg_rate.index]), columns = ukbb_y_idx.columns, index=df_pca_chg_rate.index)

#Evaluate on a col by col basis in
#
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# Y values
cat_dict = {
    3  : 'Early_Life',
    10 : "Lifestyle_Gen",
    11 : "Exercise",
    13 : "Alcohol",
    14 : "Tobacco",
    20 : "Physical",
    21 : "Bone",
    22 : "Cardiac",
    30 : 'Blood',
    32 : 'Cogn',
    51 : 'Mental_Health'
}
# Options
catn = 32
MAIN_FOLDER = os.path.join('data/interim', '20230810_regression_pred_change_agesextime_wnoise_Demean_ONLY')

USE_PC_COMPS = False
dfX_eval = dfX[['Female', 'Age_v1', 'Waittime']].copy()
dfX_eval = dfX_org.copy() # Original
dfX_eval = dfX.copy() # Demean
dfX_eval = dfX_perm.copy() # Demean

for catn in cat_dict.keys():
    print(cat_dict[catn].upper())
    # Extract Columns associated with catn
    # outcomes_pre = ukbb_y.columns[0:10]
    # outcomes_pre = corr_org[corr_org['catid'] == catn]['phesid'].values
    if catn == 31: catn = 21
    outcomes = []
    pheno_one_value = []
    for ii in tqdm(df_pheno_change.columns):
        if ii == 'userID':
            continue
        # Currect Cat doesn't match desired cate
        if y_cat_dict.loc[man_plot.translate(ii, y_cat_dict)]['Cat_ID'][0] != catn:
            continue

        if len(np.unique(ukbb_y_idx.loc[df_pca_chg_rate.index][ii])) == 1:
            pheno_one_value.append(ii)
            continue
        outcomes.append(ii)
    if len(outcomes) == 0:
        print("Empty Category")
        continue

    # High/low coef interval
    coef_out = pd.DataFrame(np.zeros((dfX_eval.shape[1]+2+1, len(outcomes))), columns=outcomes, index=['r2 score', 'Adjusted r2 score', 'intercept',*dfX_eval.columns])
    # A condensed version of coef out which only saves things that are outside zero bounds
    coef_sig = pd.DataFrame(np.zeros((dfX_eval.shape[1]+1, len(outcomes))), columns=outcomes, index= ['intercept', *dfX_eval.columns])

    n_bs = 100
    SAVE_FOLDER = os.path.join(MAIN_FOLDER, f'regression_change_{cat_dict[catn]}')

    for n_yout, y_outcome in tqdm(enumerate(outcomes)):
        if y_outcome == 'userID':
            continue #go to next y outcome
        multi_reg = Ridge(fit_intercept=True, random_state=0**2)

        # dfY = dfY_normed[y_outcome][mask_2vis].copy()
        dfY = dfY_normed[y_outcome].copy()
        print(y_outcome.upper(), f'{n_yout+1}/{len(outcomes)}')

        if len(np.unique(dfY)) == 1:
            print("This value has only one possible outcome")
            pheno_one_value.append(y_outcome)
            # continue

        n_coefs = len(dfX_eval.T)+1

        # multi_reg = MultiTaskElasticNet(l1_ratio=0, fit_intercept=False, normalize=False,
                                        # alpha=0.01, random_state=0)  # multi-output Ridge regression
        multi_reg.fit(dfX_eval, dfY)
        Y_pred = multi_reg.predict(dfX_eval)
        r2_per_y_col = r2_score(dfY, Y_pred)
        print(list(zip([y_outcome], [r2_per_y_col] )))
        r2_mine,  interaction_AdR = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)
        print(interaction_AdR)
        assert np.isclose(r2_mine, r2_per_y_col), "\a Failed assesment"

        # blah
        #Boot strap (i.e. repeat analysis but with mixed up data input) to recreate the model fit
        # this creates some variance in the coefs we get, and shows how much is due to the input and how much is retained regardless of input

        # perform bagging to build confidence in the regression coefficients
        bs_coefs = []
        bs_coefs_nopc = []
        r2_vals = []
        r2_adj_vals = []
        for i_bs in range(n_bs):
            np.random.seed(i_bs**2)
            bs_sample_inds = np.random.randint(0, len(dfX_eval), len(dfX_eval)) # we are now scrambling, and thus not taking the full set of 40k pts anymore
            multi_reg.fit(dfX_eval.iloc[bs_sample_inds], dfY.iloc[bs_sample_inds])
            bs_coefs.append([multi_reg.intercept_, *multi_reg.coef_])
            # bs_coefs.append([multi_reg.intercept_, *multi_reg.coef_@dfX_pc])
            bs_coefs_nopc.append([multi_reg.intercept_, *multi_reg.coef_])

            Y_pred = multi_reg.predict(dfX_eval.iloc[bs_sample_inds])
            r2_per_y_col = r2_score(dfY.iloc[bs_sample_inds], Y_pred)
            r2_vals.append(r2_per_y_col)
            r2_mine,  interaction_AdR = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)
            assert np.isclose(r2_mine, r2_per_y_col), "\a Failed assesment"
            r2_adj_vals.append(interaction_AdR)
        bs_coefs_array = np.array(bs_coefs)
        bs_cnopc_array = np.array(bs_coefs_nopc)
        r2_array = np.array(r2_vals)
        r2_adj_vals = np.array(r2_adj_vals)
        # np.save(CURR_FOLDER + '/bs_coefs_array_200713', bs_coefs_array)
        #-------------------------------------------------


        # derive a nice human-readable output with coefficients and their BS intervals
        # from scipy.stats import scoreatpercentile
        #NB according to scipy docs, its better to use np.percentile, as its faster and scoreatpercentile is not going to be supported in the future

        th = 5 #percentile of interest: we take 5% and 95%
        #bs_coefs_array_old = np.load('ana_sMRI_SES/bs_coefs_array.npy')  # HACK !!! what we have in result niftis as coefficients
        # bs_coefs_array = np.load(CURR_FOLDER + '/bs_coefs_array_200713.npy')  # HACK !!!
        mean_CI = bs_coefs_array.mean(0)
        upper_CI = np.abs(scoreatpercentile(bs_coefs_array, 100 - th, axis=0) - bs_coefs_array.mean(0))
        lower_CI = np.abs(scoreatpercentile(bs_coefs_array, th, axis=0) - bs_coefs_array.mean(0))

        lower_CI_org = mean_CI - lower_CI
        upper_CI_org = mean_CI + upper_CI


        mean_CIr2 = r2_array.mean(0)
        upper_CIr2 = np.abs(scoreatpercentile(r2_array, 100 - th, axis=0) - r2_array.mean(0))
        lower_CIr2 = np.abs(scoreatpercentile(r2_array, th, axis=0) - r2_array.mean(0))

        mean_CIr2_adj = r2_adj_vals.mean(0)
        upper_CIr2_adj = np.abs(scoreatpercentile(r2_adj_vals, 100 - th, axis=0) - r2_adj_vals.mean(0))
        lower_CIr2_adj = np.abs(scoreatpercentile(r2_adj_vals, th, axis=0) - r2_adj_vals.mean(0))

        lower_CI = [mean_CIr2 - lower_CIr2, mean_CIr2_adj - lower_CIr2_adj, *lower_CI_org]
        upper_CI = [mean_CIr2 + upper_CIr2, mean_CIr2_adj + upper_CIr2_adj, *upper_CI_org]

        #add upper and lower bounds into the coeff matrix
        icol = n_yout
        iref = icol + 2*icol
        coef_out[y_outcome] = [mean_CIr2, mean_CIr2_adj, *mean_CI]
        coef_out.insert(iref + 1, coef_out.columns[iref] + '_5%', lower_CI)
        coef_out.insert(iref + 2, coef_out.columns[iref] + '_95%', upper_CI)

        # Pick out significant components
        means  =[]
        if USE_PC_COMPS:
            mean_CI = bs_cnopc_array.mean(0)
            upper_CI = np.abs(scoreatpercentile(bs_cnopc_array, 100 - th, axis=0) - bs_cnopc_array.mean(0))
            lower_CI = np.abs(scoreatpercentile(bs_cnopc_array, th, axis=0) - bs_cnopc_array.mean(0))

            lower_CI_org = mean_CI - lower_CI
            upper_CI_org = mean_CI + upper_CI

        for lk in range(len(lower_CI_org)):
            if (lower_CI_org[lk] < 0 and upper_CI_org[lk] > 0):
                means.append(0)
            else:
                means.append(mean_CI[lk])

        coef_sig[y_outcome] = means
    print('\a')

    from pathlib import Path
    Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Now saving")
    coef_out.to_csv(SAVE_FOLDER + f'/effect_sizes_and_CI_{n_bs}bs.csv', float_format="%1.3f")
    coef_sig.to_csv(SAVE_FOLDER + f'/sig_comps_{n_bs}bs.csv', float_format="%1.3f")
    # np.savetxt(SAVE_FOLDER + f'/one_val_pheno_{cat_dict[catn]}.txt', pheno_one_value, delimiter = ',', fmt='%s')

dfX_eval.std().to_csv(os.path.join(MAIN_FOLDER, 'dfX_reg_stds.csv'))
print('\a\a')

plt.figure(); sns.heatmap(coef_sig, cmap = 'bwr', center = 0)
plt.title("Sig coef from linear prediction of visit 2 value based on visit 1 value")
plt.tight_layout()

f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]}, figsize = (15,10))
sns.heatmap(coef_sig, cmap = 'coolwarm', center = 0, linecolor ='k', linewidth = 0.1, yticklabels = coef_sig.index, ax = a0)
sns.heatmap(coef_out.loc['r2 score'][outcomes].to_frame().T, annot = True, cbar = False, cmap = 'viridis', linecolor ='k', linewidth = 0.1, square=True,  ax = a1)
off_diag_mask = coef_sig == 0 #-1 gives lower diagonal
sns.heatmap(coef_sig, mask=~off_diag_mask, cbar=False, annot = False, cmap = 'Greys', vmin=0, vmax=1,  linecolor ='k', linewidth = 0.1, ax = a0)
plt.title(f"{cat_dict[catn]} Sig coef of change")
f.tight_layout()

#%%
##################################################################
# Summary: Load in r2 values
# Plot them manhattan plot style
keys = ['pheno', 'r2 score', 'Adj r2 score', 'catn', 'cat name', 'i', 'top_comp', 'top_comp_val', 'top_comp_group']
dict_top_comp = {key : [] for key in keys}

USE_STDS = True
runtot = 1 # start index from 1
for catn in tqdm(cat_dict.keys()):
    if catn  == 3 or catn == 30:
        continue
    # Load in Data
    SAVE_FOLDER = os.path.join('data/interim', 'regression_33_ptrn_v2', f'regression_change_{cat_dict[catn]}')
    SAVE_FOLDER = os.path.join('data/interim', 'regression_no_interaction_v2', f'regression_change_{cat_dict[catn]}')
    SAVE_FOLDER = os.path.join('data/interim', 'regression_w_rate', f'regression_change_{cat_dict[catn]}')
    # SAVE_FOLDER = os.path.join('data/interim', 'regression_rate_PCAed', f'regression_change_{cat_dict[catn]}')
    SAVE_FOLDER = os.path.join('data/interim', 'regression_rate_age_sex', f'regression_change_{cat_dict[catn]}')
    SAVE_FOLDER = os.path.join('data/interim', '20230623_regression_LBAC_MBAC_agev2_Demean_ONLY', f'regression_change_{cat_dict[catn]}')
    SAVE_FOLDER = os.path.join('data/interim', '20230810_regression_pred_change_agesextime_Demean_ONLY', f'regression_change_{cat_dict[catn]}')
    SAVE_FOLDER = os.path.join('data/interim', '20230810_regression_pred_change_agesextime_wnoise_Demean_ONLY', f'regression_change_{cat_dict[catn]}')
    # SAVE_FOLDER = os.path.join('data/interim', '20230810_regression_pred_change_Demean_ONLY', f'regression_change_{cat_dict[catn]}')
    # SAVE_FOLDER = os.path.join('data/interim', '20230623_regression_sex_waittime_agev2_Demean_ONLY', f'regression_change_{cat_dict[catn]}')
    # SAVE_FOLDER = os.path.join('data/interim', '20230623_regression_sex_waittime_agev2_perm_Demean_ONLY', f'regression_change_{cat_dict[catn]}')
    # SAVE_FOLDER = os.path.join('data/interim', 'regression_after_new_deconf_1st', f'regression_change_{cat_dict[catn]}')
    coefs_all = pd.read_csv(os.path.join(SAVE_FOLDER, 'effect_sizes_and_CI_100bs.csv'), index_col= 0)
    if USE_PC_COMPS:
        coefs_all_new = (coefs_all.iloc[3::].T@dfX_pc).T
        coefs_all = pd.concat([coefs_all.iloc[0:3], coefs_all_new])
    phenos = [ii for ii in coefs_all.columns if '%' not in ii]

    if USE_STDS:
        dfX_stds = pd.read_csv(os.path.join(*SAVE_FOLDER.split('/')[:-1], 'dfX_reg_stds.csv'), index_col= 0)
        coefs_comp = coefs_all[3::].multiply(dfX_stds.squeeze()*2, axis='index')
        coefs_comp.loc['intecept'] = coefs_all.loc['intercept']
    else:
        coefs_comp = coefs_all[2::]

    # We want to keep the r2 score and the ID of the strongest coef
    for ip, pheno in enumerate(phenos):
        dict_top_comp['pheno'].append(pheno)
        dict_top_comp['catn'].append(catn if catn != 21 else 31)
        dict_top_comp['i'].append(runtot + ip)
        dict_top_comp['cat name'].append(cat_dict[catn])
        dict_top_comp['r2 score'].append(coefs_all[pheno]['r2 score'])
        dict_top_comp['Adj r2 score'].append(coefs_all[pheno]['Adjusted r2 score'])
        dict_top_comp['top_comp_val'].append(coefs_comp[pheno].abs().max())

        top_comp = coefs_comp.index[coefs_comp[pheno].abs().argmax()]
        dict_top_comp['top_comp'].append(top_comp)
        if top_comp == 'intercept': grp = 'intercept'
        elif 'Wait*' in top_comp: grp = 'Wait*Pattern'
        elif '/Wait' in top_comp: grp = 'Pattern Change Rate'
        elif 'Amount' in top_comp: grp = 'Amount of Change'
        elif 'Pattern' in top_comp: grp = 'Pattern Change'
        else: grp = 'Wait/Sex/Age'
        dict_top_comp['top_comp_group'].append(grp)
    runtot = runtot + len(phenos)

# df_phen_comp = pd.DataFrame(dict_top_comp) # Baseline comparison
df_phen_reg = pd.DataFrame(dict_top_comp) # Main models

# Draw Plot


################################
# Accessory Work
################################
# Shape Defined by top_comp_group
# new_cat_name = man_plot.cat_name(corr_org, y_cat_dict)
new_cat_name = ['early life factors', 'lifestyle - general',
       'lifestyle -\nexercise & work', 'lifestyle - alcohol',
       'lifestyle - tobacco', 'physical - general', 'physical - cardiac',
       'blood assays', 'physical - bone\ndensity & sizes',
       'cognitive\nphenotypes', 'mental health\nself-report']

colors = sns.color_palette('bright', n_colors=len(np.unique(df_phen_reg['catn'])))
colors = sns.color_palette('bright', n_colors=11)

cat_color_map = {'early life factors' : colors[0],
                 'lifestyle - general' : colors[1],
                 'lifestyle - exercise & work' : colors[2],
                 'lifestyle - alcohol' : colors[3],
                 'lifestyle - tobacco' : colors[4],
                 'physical - general' : colors[7],
                 'physical - cardiac' : colors[6],
                 'blood assays' : colors[8],
                 'physical - bone density & sizes' : colors[5],
                 'cognitive phenotypes' : colors[9],
                 'mental health self-report' : colors[10]}

cat_short = {'blood assays': 'blood',
             'cognitive phenotypes': 'Cogn',
             'early life factors': 'early_life',
             'lifestyle and environment - alcohol': 'alcohol',
             'lifestyle and environment - exercise and work': 'exercise',
             'lifestyle and environment - general': 'lifestyle_general',
             'lifestyle and environment - tobacco': 'tobacco',
             'lifestyle - alcohol': 'alcohol',
             'lifestyle - exercise & work': 'exercise',
             'lifestyle - general': 'lifestyle_gen',
             'lifestyle - tobacco': 'tobacco',
             'mental health self-report': 'mental_health',
             'physical measures - bone density and sizes': 'bone',
             'physical measures - cardiac & blood vessels': 'cardiac',
             'physical measures - general': 'physical',
             'physical - bone density & sizes': 'bone',
             'physical - cardiac': 'cardiac',
             'physical - general': 'physical',
             'Small': 'age_sex_wait_only'}

# Turn from long form into group id
cat_dict_rev = {cat_dict[k].lower():k for k in cat_dict}
cat_dict_rev['bone'] = 31
cat_color_map_catid = { cat_dict_rev[cat_short[cat].lower()]: cat_color_map[cat] for cat in cat_color_map}


# marker_lbl = [ii.get_text() for ii in plot._legend.get_texts()[-4::]]
################################
# Draw Plot
################################

df_phen_reg['r2_score'] = df_phen_reg['Adj r2 score']

# Subtract full model results from the results of the smaller model
# Unless the smaller model yields a sub-zero r2
df_phen_reg['r2_score'] = df_phen_reg['Adj r2 score'] - df_phen_comp['Adj r2 score']
df_phen_reg['r2_score'][df_phen_comp['Adj r2 score']<0] = df_phen_reg['Adj r2 score'][df_phen_comp['Adj r2 score']<0]


# Move around NO2 pollution point bc if i don't it completely overlaps
df_phen_reg.at[184, 'i'] = 181
df_phen_reg.at[180, 'i'] = 185

new_cat_name_change = new_cat_name[1::]
new_cat_name_change.pop(6)
markers = ['*', 'o', '^']

marker_lbl = ['Age/Sex/Wait', 'Pattern', 'Pattern * Wait']

markers = ['*', 'o', '^']
markers = ['*', '^', 'o']
marker_lbl = ['Age/Sex/Wait',  'Pattern * Wait', 'Pattern']

markers = [ 'o', '^', '*']
# markers = [ '*', '^', 'o']

marker_lbl = ['Pattern Change', 'Rate of Pattern Change', 'Wait/Sex/Age']
marker_lbl = ['Wait/Sex/Age', 'Pattern Change', 'Amount of Pattern Change']
marker_lbl = ['Amount of Pattern Change', 'Pattern Change', 'Wait/Sex/Age']

plot = sns.relplot(data=df_phen_reg, x='i', y=f'r2_score',  edgecolor='k',
                    aspect=1.3, height=7, hue='catn', style='top_comp_group',
                    palette=cat_color_map_catid, legend=False, markers= markers)
t_df = df_phen_reg.groupby('catn')['i'].median()
t_dfm = df_phen_reg.groupby('catn')['i'].max()[:-1]
# Shift exercise and work and bone density
# a bit to the right for enhanced readibility
# t_df[31] = t_df[31] + 20  # Blood Assays
# t_df[11] = t_df[11] + 20  # Exercise + Work
# t_df[11] = t_df[11] + 20  # Exercise + Work

plot.ax.set_ylabel('Adjusted $R^2$ score [Gain over age/sex/wait model]')
plot.ax.set_xlabel('')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plot.ax.set_xticks(t_df)
    plot.ax.set_xticklabels(new_cat_name_change, rotation=90, ha='right')

cat_color_map_quick = {i:cat_color_map_catid[i] for i in cat_color_map_catid if i not in [3, 30]}
for xtick, color in zip(plot.ax.get_xticklabels(), cat_color_map_quick):
    xtick.set_color(cat_color_map_quick[color])
    # xtick.set_color(color)

plt.tick_params(axis='x', bottom=False)
plot.fig.suptitle(f'Manhattan plot of {lk}')
[plt.axvline(x=xc, color='k', linestyle='--') for xc in t_dfm];
plt.axhline(y=0, color='k', linestyle='-')

import matplotlib.lines as mlines
lines = []
for ii, marker in enumerate(markers):
    lines.append(mlines.Line2D([], [], color='grey', marker=marker, linewidth=0, markeredgecolor='k',
                            label=marker_lbl[ii]))
plt.legend(handles=lines)

# if ylim:
#     plot.fig.tight_layout()
#     plot.set(ylim=ylim)
#     plot.fig.tight_layout()
#     locs, labels = plt.yticks()
#     plt.yticks([*locs, -np.log10(thresBon), -np.log10(thresFDR)],
#                 [*labels, 'BON', "FDR"])
# else:
#     locs, labels = plt.yticks()
#     plt.yticks([*locs, -np.log10(thresBon), -np.log10(thresFDR)],
#                 [*labels, 'BON', "FDR"])
plt.grid(axis='y', which='both')
plot.fig.tight_layout()

VIS_FOLDER = 'reports/Project2_Longitudional/Long_Fig_Drafts/Fig2_Phenome'

plt.savefig(os.path.join(VIS_FOLDER, "brain2behavchange_null_removed.pdf"))

################################
# Evaluate
################################
'''
{3: 'Early_Life',
 10: 'Lifestyle_Gen',
 11: 'Exercise',
 13: 'Alcohol',
 14: 'Tobacco',
 20: 'Physical',
 21: 'Bone',
 22: 'Cardiac',
 30: 'Blood',
 32: 'Cogn',
 51: 'Mental_Health'}
 '''
df_phen_reg.query('catn==30 and r2_score>0.075')
df_phen_reg.query('r2_score>0.03')[['pheno', 'top_comp', 'cat name', 'i', 'r2_score']]