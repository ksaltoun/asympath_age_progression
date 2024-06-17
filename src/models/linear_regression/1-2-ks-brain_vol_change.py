
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from tqdm import tqdm

'''
Predict total brain volume (change) on the basis of either
    - time between visits
    - time between visits + demographics (age, sex, etc)
    - time between visits + baseline asymmetry
    - time between visits + asymmetry change (LBAC)
    - time between visits + asymmetry change (MBAC)

Visualization + permutation also in code
'''

def adjusted_R2(y_pred, y_true, d, n=None):
    # n =number variables
    # d = number of parameters
    # y_true; y_pred = true and predicted y value

    assert len(y_pred) == len(y_true), 'Expected estimates and true y to be equal length'

    RSS = ((y_true-y_pred)**2).sum()
    TSS = ((y_true-y_true.mean())**2).sum()
    if n is None:
        n=len(y_pred)

    R2 = 1-RSS/TSS
    Adj_R2 = 1-RSS/TSS*(n-1)/(n-d-1)
    return Adj_R2.values

def adjusted_R2_v2(r2, d, n):
    # n =number variables
    # d = number of parameters
    # y_true; y_pred = true and predicted y value

    RSS_P = 1 - r2
    R2 = r2
    Adj_R2 = 1-RSS_P*(n-1)/(n-d-1)
    return R2, Adj_R2

def BIC(y_pred, y_true, d, n=None):
    assert len(y_pred) == len(y_true), 'Expected estimates and true y to be equal length'
    if n is None:
        n=len(y_pred)

    RSS = ((y_true-y_pred)**2).sum()
    BIC = -2 * np.log(RSS/n) + d *np.log(n)
    return BIC.values
def AIC(y_pred, y_true, d, n=None):
    assert len(y_pred) == len(y_true), 'Expected estimates and true y to be equal length'
    if n is None:
        n=len(y_pred)

    RSS = ((y_true-y_pred)**2).sum()
    AIC = -2/n * np.log(RSS/n) + 2* d /n
    return AIC
    aic_score = n*np.log(rss/n) + 2*p
# Expt
# Can PCA Changes be used to predict brain volume change?

# Is brain towards/away from asymm related to change in total brain volume?
vols = ukbb[['eid', '25009-2.0', '25009-3.0']] # Grey + white matter (norm for head size)
vols = ukbb[['eid', '25005-2.0', '25005-3.0']] # Grey matter (norm for head size)
vols = ukbb[['eid', '25009-2.0', '25009-3.0', '25005-2.0', '25005-3.0', '25007-2.0', '25007-3.0']] # Grey matter (norm for head size)
vols = vols.set_index('eid').sort_index()

eids = df_pca_chg_rate.index.values
vol_change = pd.DataFrame()
vol_change['BV Change (G+W)'] = 100*(vols.loc[eids]['25009-3.0'] - vols.loc[eids]['25009-2.0'])/vols.loc[eids]['25009-2.0']
vol_change['BV Change (G)'] = 100*(vols.loc[eids]['25005-3.0'] - vols.loc[eids]['25005-2.0'])/vols.loc[eids]['25005-2.0']
vol_change['BV Change (W)'] = 100*(vols.loc[eids]['25007-3.0'] - vols.loc[eids]['25007-2.0'])/vols.loc[eids]['25005-2.0']
vol_change['BV (G)'] = StandardScaler().fit_transform(vols.loc[eids]['25005-2.0'].values.reshape(-1, 1))
vol_change['BV (G+W)'] = StandardScaler().fit_transform(vols.loc[eids]['25009-2.0'].values.reshape(-1, 1))
vol_change['BV (W)'] = StandardScaler().fit_transform(vols.loc[eids]['25007-2.0'].values.reshape(-1, 1))


N_DAYS_PER_YEAR = 365
vol_change['BV Change (G+W) (rate)'] = N_DAYS_PER_YEAR*vol_change['BV Change (G+W)']/waittime_orgd
vol_change['BV Change (G) (rate)'] = N_DAYS_PER_YEAR*vol_change['BV Change (G)']/waittime_orgd
vol_change['BV Change (W) (rate)'] = N_DAYS_PER_YEAR*vol_change['BV Change (W)']/waittime_orgd


# Load in Data
DATA_FOLDER = 'models/Asymm_Patterns/Changes/2023_02_post_deconf'
df_pca_chg_rate = pd.read_csv(os.path.join(DATA_FOLDER, 'Rate_Adjusted', 'Rate_Asymm_Change.csv'), index_col=0)
df_amt_chg_rate = pd.read_csv(os.path.join(DATA_FOLDER, 'Rate_Adjusted', 'Rate_Amount_Change.csv'), index_col=0)

dfX_new_2vis = pd.read_csv('data/Project_1/Processed/df_Pattern_Transform_All.csv', index_col=0)
dfX_new_2vis = dfX_new_2vis.set_index('eid').sort_index()
dfX_new_2vis_v1 = dfX_new_2vis.loc[df_amt_chg_rate.index.values]
dfX_new_2vis_v1.columns = dfX_new_2vis_v1.columns.map(lambda x: x.replace('Comp ', 'Pattern '))

meta_df = pd.read_csv('data/Project_1/Processed/df_meta.csv', index_col=0)
meta_df_idx = meta_df.set_index('eid').sort_index()

ageraw = ukbb['21003-2.0'].values[:, np.newaxis]  # Age at recruitment

##############################################################################################
# Different Models Predicting %BV_Change
##############################################################################################


var_plot = labels[0:33]

# Set up X (i.e. visit 1 values)
n_comp = len(var_plot)
dfXvc = df_pca_chg_rate[[ *var_plot]].copy()
dfXvc['Waittime'] = waittime_days
dfXvc = dfXvc.join(meta_df_idx[['sex_F', 'age_v2']])
dfXvc= dfXvc.rename(columns={"sex_F": "Female", "age_v2": "Age_v1"})
dfXvc['Age2'] = dfXvc['Age_v1'] **2
dfXvc['Age2_Sex'] = dfXvc['Age_v1'] **2 * dfXvc['Female']
dfXvc['Age_Sex'] = dfXvc['Age_v1'] * dfXvc['Female']
# dfXvc = dfXvc.drop(columns = var_plot)

waitcolTerm = []
for col in var_plot:
    dfXvc[f'{col}_Org'] = dfX_new_2vis_v1[col]
    waitcolTerm.append(f'{col}_Org')

amountcolTerm = []
for col in var_plot:
    dfXvc[f'Amount_{col}'] = df_amt_chg_rate[col]
    amountcolTerm.append(f'Amount_{col}')

extras = ['Age_v1', 'Female', 'Age2', 'Age2_Sex', 'Age_Sex']
models = {"Baseline Asymm": ['Waittime', *waitcolTerm],
        "Asymm Change": ['Waittime', *var_plot],
        "Amount Asymm Change": ['Waittime', *amountcolTerm],
        # "Asymm Change + Age/Sex": np.concatenate([['Waittime'], var_plot, extras]),
        "Age/Sex": np.concatenate([['Waittime'], extras]),
        "Waittime": ['Waittime'] }

# High/low coef interval
outcomes = ['%BV_change']
outcomes = vol_change.columns
outcomes = [c for c in outcomes if ('(rate)' in c) or ('Change' not in c)]
# For now we will only look at R2 scores
# and therefore ignore coefs
# coef_out = pd.DataFrame(np.zeros((dfXvc.shape[1]+2, len(outcomes) )), columns=outcomes, index=['r2 score', 'intercept',*dfXvc.columns])
# # A condensed version of coef out which only saves things that are outside zero bounds
# coef_sig = pd.DataFrame(np.zeros((dfXvc.shape[1]+1, len(outcomes) )), columns=outcomes, index= ['intercept', *dfXvc.columns])


n_bs = 100

r2_dict = {}
r2_full_dict = {}
coef_full_dict = {}
coef_pe_dict = {}
for y_outcome in tqdm(outcomes):
    # Remove outcome from input
    dfY = vol_change[y_outcome].copy().to_frame()
    dfY = dfY - dfY.mean()
    r2_dict[y_outcome] = {}
    r2_full_dict[y_outcome] = {}
    coef_full_dict[y_outcome] = {}
    coef_pe_dict[y_outcome] = {}
    for coefs_list in models:
        multi_reg = Ridge(fit_intercept=True, random_state=0**2)

        dfX_eval = dfXvc[models[coefs_list]].copy()

        # multi_reg = MultiTaskElasticNet(l1_ratio=0, fit_intercept=False, normalize=False,
                                        # alpha=0.01, random_state=0)  # multi-output Ridge regression
        multi_reg.fit(dfX_eval, dfY)
        Y_pred = multi_reg.predict(dfX_eval)
        r2_per_y_col = r2_score(dfY, Y_pred, multioutput='raw_values')[0]
        # print(list(zip([y_outcome], r2_per_y_col )))

        n_coefs = len(multi_reg.coef_.flatten()) + 1 # for intercept
        r2_orig_pe,  interaction_AdR_pe = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)

        print(y_outcome.upper(), coefs_list, interaction_AdR_pe)

        # Capture Coefs
        coefs_main = [*multi_reg.intercept_]
        coefs_main.extend(*multi_reg.coef_)

        bs_coefs = []
        r2_vals = []
        r2_adj_vals = []
        for i_bs in tqdm(range(n_bs)):
            np.random.seed(i_bs**2)
            bs_sample_inds = np.random.randint(0, len(dfX_eval), len(dfX_eval)) # we are now scrambling, and thus not taking the full set of 40k pts anymore
            multi_reg.fit(dfX_eval.iloc[bs_sample_inds], dfY.iloc[bs_sample_inds])
            # coefs = np.asarray([multi_reg.intercept_, *multi_reg.coef_.flatten()])
            coefs = [*multi_reg.intercept_]
            # # coefs = [multi_reg.intercept_]
            coefs.extend(*multi_reg.coef_)
            bs_coefs.append(coefs)


            Y_pred = multi_reg.predict(dfX_eval.iloc[bs_sample_inds])
            r2_per_y_col = r2_score(dfY.iloc[bs_sample_inds], Y_pred)
            # interaction_AdR = adjusted_R2(Y_pred, dfY.iloc[bs_sample_inds], n_coefs)
            r2_orig,  interaction_AdR = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)

            r2_vals.append(r2_per_y_col)
            r2_adj_vals.append(interaction_AdR)
        bs_coefs_array = np.array(bs_coefs)
        r2_array = np.array(r2_vals)
        r2_adj_array = np.array(r2_adj_vals)

        bs_coefs_df = pd.DataFrame(bs_coefs_array, columns = ['intercept', *models[coefs_list]])
        coef_full_dict[y_outcome][coefs_list] = bs_coefs_df
        coef_pe_dict[y_outcome][coefs_list] = {b: a for a, b in zip(coefs_main, ['intercept', *models[coefs_list]])}
        # derive a nice human-readable output with coefficients and their BS intervals
        from scipy.stats import scoreatpercentile
        #NB according to scipy docs, its better to use np.percentile, as its faster and scoreatpercentile is not going to be supported in the future

        th = 2.5 #percentile of interest: we take 5% and 95%
        # # Code To keep coefs
        # mean_CI = bs_coefs_array.mean(0)
        # upper_CI = np.abs(scoreatpercentile(bs_coefs_array, 100 - th, axis=0) - bs_coefs_array.mean(0))
        # lower_CI = np.abs(scoreatpercentile(bs_coefs_array, th, axis=0) - bs_coefs_array.mean(0))

        # lower_CI = mean_CI - lower_CI
        # upper_CI = mean_CI + upper_CI

        # means  =[]
        # for lk in range(len(lower_CI)):
        #     if (lower_CI[lk] < 0 and upper_CI[lk] > 0):
        #         means.append(0)
        #     else:
        #         means.append(mean_CI[lk])
        # # coef_sig[cat] = means
        r2_orig_pe,  interaction_AdR_pe = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)

        mean_CIr2 = r2_array.mean(0)
        mean_r2 = r2_orig_pe
        upper_CIr2 = np.abs(scoreatpercentile(r2_array, 100 - th, axis=0) - r2_array.mean(0))
        lower_CIr2 = np.abs(scoreatpercentile(r2_array, th, axis=0) - r2_array.mean(0))
        mean_CIr2ad = r2_adj_array.mean(0)
        mean_r2ad = interaction_AdR_pe
        upper_CIr2ad = np.abs(scoreatpercentile(r2_adj_array, 100 - th, axis=0) - r2_adj_array.mean(0))
        lower_CIr2ad = np.abs(scoreatpercentile(r2_adj_array, th, axis=0) - r2_adj_array.mean(0))

        cat_r2 = {'Mean_CI': mean_CIr2,
                'Mean': mean_r2,
                f'Upper_{100-th}': upper_CIr2,
                f'Lower_{th}': lower_CIr2,
                'Mean_CI_Adj': mean_CIr2ad,
                'Mean_Adj': mean_r2ad,
                f'Upper_{100-th}_Adj': upper_CIr2ad,
                f'Lower_{th}_Adj': lower_CIr2ad}

        # # Code to keep Coefs
        # lower_CI = [mean_CIr2 - lower_CIr2, *lower_CI]
        # upper_CI = [mean_CIr2 + upper_CIr2, *upper_CI]

        # coef_dict[cat_short[cat]+'_Change'] = [mean_CIr2, *mean_CI]
        # coef_dict[cat_short[cat]+'_Change'+'_2.5%'] = lower_CI
        # coef_dict[cat_short[cat]+'_Change'+'_97.5%'] = upper_CI

        r2_dict[y_outcome][coefs_list] = cat_r2
        r2_full_dict[y_outcome][coefs_list+"_Adj"] = r2_adj_array.flatten()
        r2_full_dict[y_outcome][coefs_list+"_Org"] = r2_array.flatten()


#convert into dataframe
r2_full_df ={}
r2_df ={}
for y_outcome in tqdm(outcomes):
    r2_full_df[y_outcome] = pd.DataFrame.from_dict(r2_full_dict[y_outcome])
    r2_df[y_outcome] = pd.DataFrame.from_dict(r2_dict[y_outcome])
    # Fix an oopsie
    r2_df[y_outcome].loc['Mean_Adj'] = r2_df[y_outcome].loc['Mean_Adj']#.map(lambda x: x[0])
    r2_df[y_outcome].loc[f'Lower_{th}_Adj'] = r2_df[y_outcome].loc[f'Lower_{th}_Adj']#.map(lambda x: x[0])
    r2_df[y_outcome].loc[f'Upper_{100-th}_Adj'] = r2_df[y_outcome].loc[f'Upper_{100-th}_Adj']#.map(lambda x: x[0])

TBV_FOLDER = 'data/interim/Predict_TBV'
for y_outcome in tqdm(outcomes):
    yout = ['delBV' if 'Change' in y_outcome else 'BV', '_',
            'G' if '(G' in y_outcome else '',
            'W' if 'W)' in y_outcome else '']
    r2_full_df[y_outcome].to_csv(os.path.join(TBV_FOLDER, f'{yout}_fullR2.csv'))
    r2_df[y_outcome].to_csv(os.path.join(TBV_FOLDER, f'{yout}_slctR2.csv'))
    coef_pt_est = pd.DataFrame.from_dict(coef_pe_dict[y_outcome])
    coef_pt_est.to_csv(os.path.join(TBV_FOLDER, f'{yout}_coef_pt_est.csv'))
    for model_type in coef_full_dict[y_outcome]:
        modsn = model_type.replace('/', '').replace(' ','_')
        coef_full_dict[y_outcome][model_type].to_csv(os.path.join(TBV_FOLDER, f'{yout}_{modsn}_bs_coef.csv'))

####################################################################################################
# Examine Coefs
thres = 2.5
modsn = 'Asymm Change'
for y_outcome in outcomes:
    print(y_outcome.upper())
    ru = r2_full_df[y_outcome][modsn+'_Adj'].quantile(thres/100)
    rl = r2_full_df[y_outcome][modsn+'_Adj'].quantile(1-thres/100)
    print(f"R2 Score: {r2_df[y_outcome][modsn]['Mean_Adj']:0.3f} [{rl:0.3f}, {ru:0.3f}]")
    lower = coef_full_dict[y_outcome][modsn].quantile(thres/100)
    upper = coef_full_dict[y_outcome][modsn].quantile(1 - thres/100)
    coef_pt_est = pd.DataFrame.from_dict(coef_pe_dict[y_outcome][modsn], orient='index')
    coefs = coef_pt_est.merge(lower, left_index=True, right_index=True)
    coefs = coefs.merge(upper, left_index=True, right_index=True)
    print(coefs[~((lower <0) & (upper > 0))])
####################################################################################################
# Visualize the Distribubtion of R2
# One outcome class (white/grey/both); all models

V_measure_type = ['(G)', '(W)', '(G+W)']
V_measure_outcome = {v: [i for i in outcomes if i.endswith(v)] for v in V_measure_type}
mpick = V_measure_type[1]
r2_melt_list = []
for oc in V_measure_outcome[mpick]:
    r2_melt = r2_full_df[oc].melt()
    r2_melt['Adj'] = r2_melt['variable'].map(lambda x:x.split('_')[-1])
    r2_melt['variable'] = r2_melt['variable'].map(lambda x:x.split('_')[0])
    r2_melt['Outcome'] = oc
    r2_melt_list.append(r2_melt)
r2_melt = pd.concat(r2_melt_list)

fig, ax = plt.subplots()
sns.violinplot(data=r2_melt[r2_melt['Adj']=='Adj'],
               hue="variable", y='value', x='Outcome',
               scale='width', palette='spring')
# plt.xticks(rotation=45, ha='right')
# plt.legend(loc='upper left')
ax.set_ylim(-0.05, 0.53)
ax.minorticks_on()
plt.grid(axis='y', which='both')
# ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
ax.axhspan(ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], xmin=0,  xmax=0.5, alpha=0.7, facecolor='lightgrey', zorder=0 ) # Q2
plt.xlabel('')
plt.ylabel('Adjusted $R^2$  Score')
plt.axhline(0, c='k')
plt.tight_layout()
ax.legend(title='Predictor Variables')


####################################################################################################
# Visualize the Distribubtion of R2
# All outcome class (white/grey/both); Change or original only

T_measure_type = ['BV (', '(rate)', 'Change']
T_measure_outcome = {v: [i for i in outcomes if v in i] for v in T_measure_type}

mpick = T_measure_type[0]
r2_melt_list = []
for oc in T_measure_outcome[mpick]:
    r2_melt = r2_full_df[oc].melt()
    r2_melt['Adj'] = r2_melt['variable'].map(lambda x:x.split('_')[-1])
    r2_melt['variable'] = r2_melt['variable'].map(lambda x:x.split('_')[0])
    r2_melt['Outcome'] = oc
    r2_melt_list.append(r2_melt)
r2_melt = pd.concat(r2_melt_list)

fig, ax = plt.subplots()
sns.violinplot(data=r2_melt[r2_melt['Adj']=='Adj'],
               hue="variable", y='value', x='Outcome',
               inner = None,
               scale='width', palette='spring')
# plt.xticks(rotation=45, ha='right')
# plt.legend(loc='upper left')
# # Add in the Mean from point estimate
# for oc in T_measure_outcome[mpick]:
#     r2_df

ax.set_ylim(-0.05, 0.3)
ax.minorticks_on()
plt.grid(axis='y', which='both')
# ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
ax.axhspan(ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], xmin=0.33,  xmax=0.66, alpha=0.7, facecolor='lightgrey', zorder=0 ) # Q2
plt.xlabel('')
plt.ylabel('Adjusted $R^2$ Score ($\pm$ Bootstrap st. dev.)')
plt.axhline(0, c='k')
plt.tight_layout()
ax.legend(title='Predictor Variables')

plt.savefig(os.path.join(VIS_FOLDER, "TBV_fromLBAC_chng.svg"))
plt.savefig(os.path.join(VIS_FOLDER, "TBV_fromLBAC_chng_rate.svg"))
plt.savefig(os.path.join(VIS_FOLDER, "TBV_fromLBAC_orig.svg"))


##############################################################################################
# Permutation Test
##############################################################################################

from scipy.stats import ttest_ind

ttest_ind(a, b, equal_var=False)

# Permutation Test
r2_perm_dict = {}
r2_perm_full_dict = {}
for y_outcome in tqdm(outcomes):
    # Remove outcome from input
    dfY = vol_change[y_outcome].copy().to_frame()
    r2_perm_dict[y_outcome] = {}
    r2_perm_full_dict[y_outcome] = {}
    for coefs_list in models:
        multi_reg = Ridge(fit_intercept=True, random_state=0**2)

        dfX_eval = dfXvc[models[coefs_list]].copy()


        y_inds_perm = np.arange(0, len(dfY))
        np.random.shuffle(y_inds_perm)
        dfY_permuted = dfY.iloc[y_inds_perm]

        # multi_reg = MultiTaskElasticNet(l1_ratio=0, fit_intercept=False, normalize=False,
                                        # alpha=0.01, random_state=0)  # multi-output Ridge regression
        multi_reg.fit(dfX_eval, dfY_permuted)
        Y_pred = multi_reg.predict(dfX_eval)
        r2_per_y_col = r2_score(dfY_permuted, Y_pred, multioutput='raw_values')[0]
        # print(list(zip([y_outcome], r2_per_y_col )))

        n_coefs = len(multi_reg.coef_.flatten()) + 1 # for intercept
        interaction_AdR = adjusted_R2(Y_pred, dfY_permuted, n_coefs)[0]

        print(y_outcome.upper(), coefs_list, interaction_AdR)
        # bs_coefs = []
        r2_vals = []
        r2_adj_vals = []
        for i_bs in tqdm(range(n_bs)):
            np.random.seed(i_bs**2)
            y_inds_perm = np.arange(0, len(dfY))
            np.random.shuffle(y_inds_perm)
            dfY_permuted = dfY.iloc[y_inds_perm]
            multi_reg.fit(dfX_eval, dfY_permuted)

            Y_pred = multi_reg.predict(dfX_eval)
            r2_per_y_col = r2_score(dfY_permuted, Y_pred)
            interaction_AdR = adjusted_R2(Y_pred, dfY_permuted, n_coefs)

            r2_vals.append(r2_per_y_col)
            r2_adj_vals.append(interaction_AdR)
        # bs_coefs_array = np.array(bs_coefs)
        r2_array = np.array(r2_vals)
        r2_adj_array = np.array(r2_adj_vals)

        # derive a nice human-readable output with coefficients and their BS intervals
        from scipy.stats import scoreatpercentile
        #NB according to scipy docs, its better to use np.percentile, as its faster and scoreatpercentile is not going to be supported in the future

        th = 2.5 #percentile of interest: we take 5% and 95%

        mean_CIr2 = r2_array.mean(0)
        upper_CIr2 = np.abs(scoreatpercentile(r2_array, 100 - th, axis=0) - r2_array.mean(0))
        lower_CIr2 = np.abs(scoreatpercentile(r2_array, th, axis=0) - r2_array.mean(0))
        mean_CIr2ad = r2_adj_array.mean(0)
        upper_CIr2ad = np.abs(scoreatpercentile(r2_adj_array, 100 - th, axis=0) - r2_adj_array.mean(0))
        lower_CIr2ad = np.abs(scoreatpercentile(r2_adj_array, th, axis=0) - r2_adj_array.mean(0))

        cat_r2 = {'Mean': mean_CIr2,
                f'Upper_{100-th}': upper_CIr2,
                f'Lower_{th}': lower_CIr2,
                'Mean_Adj': mean_CIr2ad,
                f'Upper_{100-th}_Adj': upper_CIr2ad,
                f'Lower_{th}_Adj': lower_CIr2ad}

        # # Code to keep Coefs
        # lower_CI = [mean_CIr2 - lower_CIr2, *lower_CI]
        # upper_CI = [mean_CIr2 + upper_CIr2, *upper_CI]

        r2_perm_dict[y_outcome][coefs_list] = cat_r2
        r2_perm_full_dict[y_outcome][coefs_list+"_Adj"] = r2_adj_array.flatten()
        r2_perm_full_dict[y_outcome][coefs_list+"_Org"] = r2_array.flatten()


#convert into dataframe
r2_perm_full_df ={}
r2_perm_df ={}
for y_outcome in tqdm(outcomes):
    r2_perm_full_df[y_outcome] = pd.DataFrame.from_dict(r2_perm_full_dict[y_outcome])
    r2_perm_df[y_outcome] = pd.DataFrame.from_dict(r2_perm_dict[y_outcome])
    # Fix an oopsie
    r2_perm_df[y_outcome].loc['Mean_Adj'] = r2_perm_df[y_outcome].loc['Mean_Adj'].map(lambda x: x[0])
    r2_perm_df[y_outcome].loc[f'Lower_{th}_Adj'] = r2_perm_df[y_outcome].loc[f'Lower_{th}_Adj'].map(lambda x: x[0])
    r2_perm_df[y_outcome].loc[f'Upper_{100-th}_Adj'] = r2_perm_df[y_outcome].loc[f'Upper_{100-th}_Adj'].map(lambda x: x[0])
