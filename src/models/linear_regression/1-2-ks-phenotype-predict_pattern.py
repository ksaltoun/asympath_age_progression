from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

'''
Predict LBACs (or MBACs) from a phenotypic domain

THe idea is such:
  Linear Regression predicts pattern X change
  Given variables from a certain class (cognitive, mental health, etc.)
  Plot R2 score (K-fold validation average) in radar plot

Baseline and change domains are separate
demographic vars (age, sex, etc, time between visits) in all models
model with only demographic variables (no phenotypes) also included

Visualization + PCA of coefs also in code
'''

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)

def adjusted_R2_v2(r2, d, n):
    # n =number variables
    # d = number of parameters
    # y_true; y_pred = true and predicted y value

    RSS_P = 1 - r2
    R2 = 1-RSS_P
    Adj_R2 = 1-RSS_P*(n-1)/(n-d-1)
    return R2, Adj_R2

# Load in Data
DATA_FOLDER = 'models/Asymm_Patterns/Changes/2023_02_post_deconf'
df_pca_chg_rate = pd.read_csv(os.path.join(DATA_FOLDER, 'Rate_Adjusted', 'Rate_Asymm_Change.csv'), index_col=0)
df_amt_chg_rate = pd.read_csv(os.path.join(DATA_FOLDER, 'Rate_Adjusted', 'Rate_Amount_Change.csv'), index_col=0)


###########################################################################################
# UKBB - Original, Time Point 1
###########################################################################################
import processing.manhattan_plot_util as man_plot

print('Loading Phenome')
ukbb_y, y_desc_dict, y_cat_dict = man_plot.load_phenom(BASE_FOLDER = 'data/processed/')

meta_df = pd.read_csv('data/Project_1/Processed/df_meta.csv', index_col=0)

meta_df_idx = meta_df.set_index('eid').sort_index()

df_all_phen = ukbb_y.merge(meta_df_idx, right_index=True, left_on='userID')


import datetime
date_format = "%Y-%m-%dT%H:%M:%S"
v1datestr = ukbb.set_index('eid')['21862-2.0'].loc[df_pca_chg_rate.index].values
v2datestr = ukbb.set_index('eid')['21862-3.0'].loc[df_pca_chg_rate.index].values

v1date = [datetime.datetime.strptime(t, date_format) for t in v1datestr]
v2date = [datetime.datetime.strptime(t, date_format) for t in v2datestr]

waittime_days_org = np.asarray([(v2date[ii] - v1date[ii]).days for ii in range(df_pca_chg_rate.shape[0])])
waittime_days_org_df = pd.DataFrame(waittime_days_org, index = df_pca_chg_rate.index, columns=['Waittime'])
df_all_phen = df_all_phen.merge(waittime_days_org_df, right_index=True, left_on='userID')

df_all_phen['age_sex'] = df_all_phen['age_v2'] * df_all_phen['sex_F']
df_all_phen['age_v2_sq'] = df_all_phen['age_v2'] **2
df_all_phen['age2_sex'] = df_all_phen['age_v2_sq'] * df_all_phen['sex_F']
dmographic_phen = ['age_v2', 'sex_F', 'Waittime', 'age_sex', 'age2_sex', 'age_v2_sq']

def find_phenome(col):
    col = str(col)
    col_v = col.split('_')[0].split('#')[0].split('-')[0]
    col_v = [ii for ii in y_cat_dict.index
                if ii.startswith(col_v + '-')][0]
    return(col_v)

print('Extracting Columns by Category')
# group all columns by category
cat_cols_pheno = {}
all_cats = np.unique(y_cat_dict['Cat_Name'])
all_cats = np.delete(all_cats, 0) # remove the age/sex one
for cat in tqdm(all_cats):
    cat_cols_pheno[cat] = [col for col in ukbb_y.drop(columns='userID').columns
                     if y_cat_dict.loc[find_phenome(col)]['Cat_Name'] == cat]


from pathlib import Path
ptrn = 'Pattern 2'
SAVE_MODEL = True
SAVE_FOLDER = 'data/interim/231118_pred_change_from_phenome'
TEST_ABS = False
Path(SAVE_FOLDER+f'/{ptrn}').mkdir(parents=True, exist_ok=True)

from datetime import datetime
TODAY = datetime.today().strftime('%Y%m%d')
# High/low coef interval
outcomes = [ptrn]
n_bs = 100
N_PPL = 1425

cat_short = {'blood assays': 'blood',
             'cognitive phenotypes': 'cognitive',
             'early life factors': 'early_life',
             'lifestyle and environment - alcohol': 'alcohol',
             'lifestyle and environment - exercise and work': 'exercise',
             'lifestyle and environment - general': 'lifestyle_general',
             'lifestyle and environment - tobacco': 'tobacco',
             'mental health self-report': 'mental_health',
             'physical measures - bone density and sizes': 'bone',
             'physical measures - cardiac & blood vessels': 'cardiac',
             'physical measures - general': 'physical',
             'Small': 'age_sex_wait_only'}

for cp in tqdm(range(33)):
    ptrn = f'Pattern {cp+1}'
    coef_dict = {}
    r2_dict = {}
    r2_full_dict = {}
    Path(SAVE_FOLDER+f'/{ptrn}').mkdir(parents=True, exist_ok=True)

    for  n_yout,  cat in enumerate([*all_cats, 'Small']):

        # dfXvc = ukbb_y[['userID', *cat_col]]
        if cat == 'Small':
            dfXvc = df_all_phen[['userID', *dmographic_phen]]
        else:
            cat_col = cat_cols_pheno[cat]
            dfXvc = df_all_phen[['userID', *cat_col, *dmographic_phen]]

        if TEST_ABS:
            dfXY = dfXvc.merge(df_pca_chg_rate[ptrn], right_index=True, left_on='userID')
        else:
            dfXY = dfXvc.merge(df_amt_chg_rate[ptrn], right_index=True, left_on='userID')
        # dfXY = dfXY.merge(meta_df[['sex_F', 'age', 'eid']], left_on='userID', right_on='eid')

        multi_reg = Ridge(fit_intercept=True, random_state=0**2)

        dfX_eval = dfXY.copy()[dfXvc.columns]
        dfX_eval = dfX_eval.drop(columns='userID')
        # dfX_eval = dfXY.copy()[[*cat_col, 'age', 'sex_F']]

        # Remove outcome from input
        # dfY = vol_change.copy().to_frame()
        dfY = dfXY[ptrn].copy().to_frame()
        dfY = dfY - dfY.mean()

        # multi_reg = MultiTaskElasticNet(l1_ratio=0, fit_intercept=False, normalize=False,
                                        # alpha=0.01, random_state=0)  # multi-output Ridge regression
        multi_reg.fit(dfX_eval, dfY)
        Y_pred = multi_reg.predict(dfX_eval)
        r2_per_y_col = r2_score(dfY, Y_pred, multioutput='raw_values')

        n_coefs = len(multi_reg.coef_.flatten()) #+ 1 # for intercept
        # interaction_AIC = AIC(Y_pred, dfY, n_coefs)
        # interaction_BIC = BIC(Y_pred, dfY, n_coefs)
        # interaction_AdR = adjusted_R2(Y_pred, dfY, n_coefs)
        r2_mine,  interaction_AdR = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)

        print([ptrn, cat, r2_per_y_col, interaction_AdR ])

        # blah
        #Boot strap (i.e. repeat analysis but with mixed up data input) to recreate the model fit
        # this creates some variance in the coefs we get, and shows how much is due to the input and how much is retained regardless of input

        # perform bagging to build confidence in the regression coefficients
        bs_coefs = []
        r2_vals = []
        r2_adj_vals = []
        for i_bs in tqdm(range(n_bs)):
            np.random.seed(i_bs**2)
            bs_sample_inds = np.random.randint(0, len(dfX_eval), len(dfX_eval)) # we are now scrambling, and thus not taking the full set of 40k pts anymore
            multi_reg.fit(dfX_eval.iloc[bs_sample_inds], dfY.iloc[bs_sample_inds])
            # coefs = np.asarray([multi_reg.intercept_, *multi_reg.coef_.flatten()])
            coefs = [*multi_reg.intercept_]
            # coefs = [multi_reg.intercept_]
            coefs.extend(*multi_reg.coef_)
            bs_coefs.append(coefs)

            Y_pred = multi_reg.predict(dfX_eval.iloc[bs_sample_inds])
            r2_per_y_col = r2_score(dfY.iloc[bs_sample_inds], Y_pred)
            # interaction_AdR = adjusted_R2(Y_pred, dfY.iloc[bs_sample_inds], n_coefs)
            r2_mine,  interaction_AdR = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)


            r2_vals.append(r2_per_y_col)
            r2_adj_vals.append(interaction_AdR)
        bs_coefs_array = np.array(bs_coefs)
        r2_array = np.array(r2_vals)
        r2_adj_array = np.array(r2_adj_vals)
        if SAVE_MODEL == True:
            np.save(SAVE_FOLDER + f'/{ptrn}/bs_coefs_array_{ptrn}_{cat_short[cat]}_{TODAY}', bs_coefs_array)
            cur_coef = pd.DataFrame(bs_coefs_array, columns=['intercept', *dfX_eval.columns], index=[f'Perm_{i+1:03}' for i in range(n_bs)])
            cur_coef.to_csv(os.path.join(SAVE_FOLDER, ptrn, f'coefs_all_{cat_short[cat]}.csv'))
        #-------------------------------------------------


        # derive a nice human-readable output with coefficients and their BS intervals
        from scipy.stats import scoreatpercentile
        #NB according to scipy docs, its better to use np.percentile, as its faster and scoreatpercentile is not going to be supported in the future

        th = 2.5 #percentile of interest: we take 5% and 95%
        #bs_coefs_array_old = np.load('ana_sMRI_SES/bs_coefs_array.npy')  # HACK !!! what we have in result niftis as coefficients
        # bs_coefs_array = np.load(CURR_FOLDER + '/bs_coefs_array_200713.npy')  # HACK !!!
        mean_CI = bs_coefs_array.mean(0)
        upper_CI = np.abs(scoreatpercentile(bs_coefs_array, 100 - th, axis=0) - bs_coefs_array.mean(0))
        lower_CI = np.abs(scoreatpercentile(bs_coefs_array, th, axis=0) - bs_coefs_array.mean(0))

        lower_CI = mean_CI - lower_CI
        upper_CI = mean_CI + upper_CI

        means  =[]
        for lk in range(len(lower_CI)):
            if (lower_CI[lk] < 0 and upper_CI[lk] > 0):
                means.append(0)
            else:
                means.append(mean_CI[lk])
        # coef_sig[cat] = means

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

        lower_CI = [mean_CIr2 - lower_CIr2, *lower_CI]
        upper_CI = [mean_CIr2 + upper_CIr2, *upper_CI]

        coef_dict[cat_short[cat]] = [mean_CIr2, *mean_CI]
        coef_dict[cat_short[cat]+'_2.5%'] = lower_CI
        coef_dict[cat_short[cat]+'_97.5%'] = upper_CI

        if SAVE_MODEL:
            coef_bounds = pd.DataFrame.from_dict({k:coef_dict.get(k) \
                                    for k in [cat_short[cat], cat_short[cat]+'_97.5%', cat_short[cat]+'_2.5%']})
            coef_bounds.index = ['r2', 'intercept', *dfX_eval.columns]
            coef_bounds.to_csv(os.path.join(SAVE_FOLDER, ptrn, f'coefs_bounds_{cat_short[cat]}.csv'))

        r2_dict[cat_short[cat]] = cat_r2
        r2_full_dict[cat_short[cat]+"_Adj"] = r2_adj_array.flatten()
        r2_full_dict[cat_short[cat]+"_Org"] = r2_array.flatten()


    #blah
    r2_full_df = pd.DataFrame.from_dict(r2_full_dict)
    r2_df = pd.DataFrame.from_dict(r2_dict)
    # # Fix an oopsie
    # r2_df.loc['Mean_Adj'] = r2_df.loc['Mean_Adj'].map(lambda x: x[0])
    # r2_df.loc[f'Lower_{th}_Adj'] = r2_df.loc[f'Lower_{th}_Adj'].map(lambda x: x[0])
    # r2_df.loc[f'Upper_{100-th}_Adj'] = r2_df.loc[f'Upper_{100-th}_Adj'].map(lambda x: x[0])
    if SAVE_MODEL:
        r2_df.T.to_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_bounds_{ptrn}.csv'))
        r2_full_df.to_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_all_{ptrn}.csv'))
#  for outputing the upper/lower bounds
# coef_out.loc['r2 score'][np.asarray([(i+'_5%', i+'_95%') for i in cat_all]).flatten()].to_frame()


# ###############################################################
# # Here I want to pull out the R2 scores to compare

# r2_all_ptrn = {}
# for fldn in os.listdir(SAVE_FOLDER):
#     r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'r2_bounds_{fldn}.csv'), index_col=0)
#     r2_curr = r2_curr.T
#     r2_all_ptrn[fldn] = r2_curr.T['Mean_Adj'].to_dict()
# r2_all_ptrn = pd.DataFrame.from_dict(r2_all_ptrn)


###########################################################################################
# UKBB - Original, Change
###########################################################################################
import processing.manhattan_plot_util as man_plot

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


    print('Extracting Columns by Category')
    # group all columns by category
    cat_cols_chg = {}
    all_cats_chg = np.unique(y_cat_dict['Cat_Name'])
    for cat in tqdm(all_cats_chg):
        curr_col = [col for col in df_pheno_change.drop(columns='userID').columns
                        if y_cat_dict.loc[find_phenome(col)]['Cat_Name'] == cat]
        if curr_col !=[]:
            cat_cols_chg[cat] = curr_col


    import datetime
    date_format = "%Y-%m-%dT%H:%M:%S"
    v1datestr = ukbb.set_index('eid')['21862-2.0'].loc[df_pca_chg_rate.index].values
    v2datestr = ukbb.set_index('eid')['21862-3.0'].loc[df_pca_chg_rate.index].values

    v1date = [datetime.datetime.strptime(t, date_format) for t in v1datestr]
    v2date = [datetime.datetime.strptime(t, date_format) for t in v2datestr]

    waittime_days_org = np.asarray([(v2date[ii] - v1date[ii]).days for ii in range(df_pca_chg_rate.shape[0])])
    waittime_days_org_df = pd.DataFrame(waittime_days_org, index = df_pca_chg_rate.index, columns=['Waittime'])

    df_all_phen_cng = df_pheno_change.merge(meta_df_idx, right_index=True, left_on='userID')
    df_all_phen_cng = df_all_phen_cng.merge(waittime_days_org_df, right_index=True, left_on='userID')

    df_all_phen_cng['age_sex'] = df_all_phen_cng['age_v2'] * df_all_phen_cng['sex_F']
    df_all_phen_cng['age_v2_sq'] = df_all_phen_cng['age_v2'] **2
    df_all_phen_cng['age2_sex'] = df_all_phen_cng['age_v2_sq'] * df_all_phen_cng['sex_F']
    dmographic_phen = ['age_v2', 'sex_F', 'Waittime', 'age_sex', 'age2_sex', 'age_v2_sq']

from pathlib import Path
ptrn = 'Pattern 2'
SAVE_MODEL = True
SAVE_FOLDER = 'data/interim/231118_pred_change_from_phenome'
TEST_ABS = False
Path(SAVE_FOLDER+f'/{ptrn}').mkdir(parents=True, exist_ok=True)

from datetime import datetime
TODAY = datetime.today().strftime('%Y%m%d')
# High/low coef interval
outcomes = [ptrn]
n_bs = 100

cat_short = {'blood assays': 'blood',
             'cognitive phenotypes': 'cognitive',
             'early life factors': 'early_life',
             'lifestyle and environment - alcohol': 'alcohol',
             'lifestyle and environment - exercise and work': 'exercise',
             'lifestyle and environment - general': 'lifestyle_general',
             'lifestyle and environment - tobacco': 'tobacco',
             'mental health self-report': 'mental_health',
             'physical measures - bone density and sizes': 'bone',
             'physical measures - cardiac & blood vessels': 'cardiac',
             'physical measures - general': 'physical',
             'Small': 'age_sex_wait_only'}

for cp in tqdm(range(33)):
    # if cp + 1 in [1,2, 6]:
    #     continue
    ptrn = f'Pattern {cp+1}'
    coef_dict = {}
    r2_dict = {}
    r2_full_dict = {}
    Path(SAVE_FOLDER+f'/{ptrn}').mkdir(parents=True, exist_ok=True)

    for  n_yout,  cat in enumerate(cat_cols_chg.keys()):

        if cat == 'Small':
            dfXvc = df_all_phen_cng[['userID', *dmographic_phen]]
        else:
            cat_col = cat_cols_chg[cat]
            dfXvc = df_all_phen_cng[['userID', *cat_col, *dmographic_phen]]

        if TEST_ABS:
            dfXY = dfXvc.merge(df_pca_chg_rate[ptrn], right_index=True, left_on='userID')
        else:
            dfXY = dfXvc.merge(df_amt_chg_rate[ptrn], right_index=True, left_on='userID')

        multi_reg = Ridge(fit_intercept=True, random_state=0**2)

        dfX_eval = dfXY.copy()[dfXvc.columns]
        dfX_eval = dfX_eval.drop(columns='userID')
        # dfX_eval = dfXY.copy()[[*cat_col, 'age', 'sex_F']]

        # Remove outcome from input
        # dfY = vol_change.copy().to_frame()
        dfY = dfXY[ptrn].copy().to_frame()
        dfY = dfY - dfY.mean()

        # multi_reg = MultiTaskElasticNet(l1_ratio=0, fit_intercept=False, normalize=False,
                                        # alpha=0.01, random_state=0)  # multi-output Ridge regression
        multi_reg.fit(dfX_eval, dfY)
        Y_pred = multi_reg.predict(dfX_eval)
        r2_per_y_col = r2_score(dfY, Y_pred, multioutput='raw_values')

        n_coefs = len(multi_reg.coef_.flatten()) + 1 # for intercept
        # interaction_AIC = AIC(Y_pred, dfY, n_coefs)
        # interaction_BIC = BIC(Y_pred, dfY, n_coefs)
        # interaction_AdR = adjusted_R2(Y_pred, dfY, n_coefs)
        r2_mine,  interaction_AdR = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)
        print([ptrn, cat, r2_per_y_col, interaction_AdR ])

        # blah
        #Boot strap (i.e. repeat analysis but with mixed up data input) to recreate the model fit
        # this creates some variance in the coefs we get, and shows how much is due to the input and how much is retained regardless of input

        # perform bagging to build confidence in the regression coefficients
        bs_coefs = []
        r2_vals = []
        r2_adj_vals = []
        for i_bs in tqdm(range(n_bs)):
            np.random.seed(i_bs**2)
            bs_sample_inds = np.random.randint(0, len(dfX_eval), len(dfX_eval)) # we are now scrambling, and thus not taking the full set of 40k pts anymore
            multi_reg.fit(dfX_eval.iloc[bs_sample_inds], dfY.iloc[bs_sample_inds])
            # coefs = np.asarray([multi_reg.intercept_, *multi_reg.coef_.flatten()])
            coefs = [*multi_reg.intercept_]
            # coefs = [multi_reg.intercept_]
            coefs.extend(*multi_reg.coef_)
            bs_coefs.append(coefs)

            Y_pred = multi_reg.predict(dfX_eval.iloc[bs_sample_inds])
            r2_per_y_col = r2_score(dfY.iloc[bs_sample_inds], Y_pred)
            # interaction_AdR = adjusted_R2(Y_pred, dfY.iloc[bs_sample_inds], n_coefs)
            r2_mine,  interaction_AdR = adjusted_R2_v2(r2_per_y_col, n_coefs, N_PPL)


            r2_vals.append(r2_per_y_col)
            r2_adj_vals.append(interaction_AdR)
        bs_coefs_array = np.array(bs_coefs)
        r2_array = np.array(r2_vals)
        r2_adj_array = np.array(r2_adj_vals)
        if SAVE_MODEL == True:
            np.save(SAVE_FOLDER + f'/{ptrn}/bs_coefs_array_{ptrn}_{cat_short[cat]}_Change_{TODAY}', bs_coefs_array)
            cur_coef = pd.DataFrame(bs_coefs_array, columns=['intercept', *dfX_eval.columns], index=[f'Perm_{i+1:03}' for i in range(n_bs)])
            cur_coef.to_csv(os.path.join(SAVE_FOLDER, ptrn, f'coefs_all_{cat_short[cat]}_Change.csv'))
        #-------------------------------------------------


        # derive a nice human-readable output with coefficients and their BS intervals
        from scipy.stats import scoreatpercentile
        #NB according to scipy docs, its better to use np.percentile, as its faster and scoreatpercentile is not going to be supported in the future

        th = 2.5 #percentile of interest: we take 5% and 95%
        #bs_coefs_array_old = np.load('ana_sMRI_SES/bs_coefs_array.npy')  # HACK !!! what we have in result niftis as coefficients
        # bs_coefs_array = np.load(CURR_FOLDER + '/bs_coefs_array_200713.npy')  # HACK !!!
        mean_CI = bs_coefs_array.mean(0)
        upper_CI = np.abs(scoreatpercentile(bs_coefs_array, 100 - th, axis=0) - bs_coefs_array.mean(0))
        lower_CI = np.abs(scoreatpercentile(bs_coefs_array, th, axis=0) - bs_coefs_array.mean(0))

        lower_CI = mean_CI - lower_CI
        upper_CI = mean_CI + upper_CI

        means  =[]
        for lk in range(len(lower_CI)):
            if (lower_CI[lk] < 0 and upper_CI[lk] > 0):
                means.append(0)
            else:
                means.append(mean_CI[lk])
        # coef_sig[cat] = means

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

        lower_CI = [mean_CIr2 - lower_CIr2, *lower_CI]
        upper_CI = [mean_CIr2 + upper_CIr2, *upper_CI]

        coef_dict[cat_short[cat]+'_Change'] = [mean_CIr2, *mean_CI]
        coef_dict[cat_short[cat]+'_Change'+'_2.5%'] = lower_CI
        coef_dict[cat_short[cat]+'_Change'+'_97.5%'] = upper_CI

        if SAVE_MODEL:
            coef_bounds = pd.DataFrame.from_dict({k:coef_dict.get(k) \
                                    for k in [cat_short[cat]+'_Change', cat_short[cat]+'_Change'+'_97.5%', cat_short[cat]+'_Change'+'_2.5%']})
            coef_bounds.index = ['r2', 'intercept', *dfX_eval.columns]
            coef_bounds.to_csv(os.path.join(SAVE_FOLDER, ptrn, f'coefs_bounds_{cat_short[cat]}_change.csv'))

        r2_dict[cat_short[cat]+'_Change'] = cat_r2
        r2_full_dict[cat_short[cat]+'_Change'+"_Adj"] = r2_adj_array.flatten()
        r2_full_dict[cat_short[cat]+'_Change'+"_Org"] = r2_array.flatten()


    #blah
    r2_full_df = pd.DataFrame.from_dict(r2_full_dict)
    r2_df = pd.DataFrame.from_dict(r2_dict)
    # # Fix an oopsie
    # r2_df.loc['Mean_Adj'] = r2_df.loc['Mean_Adj'].map(lambda x: x[0])
    # r2_df.loc[f'Lower_{th}_Adj'] = r2_df.loc[f'Lower_{th}_Adj'].map(lambda x: x[0])
    # r2_df.loc[f'Upper_{100-th}_Adj'] = r2_df.loc[f'Upper_{100-th}_Adj'].map(lambda x: x[0])
    if SAVE_MODEL:
        r2_df.T.to_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_bounds_{ptrn}_change.csv'))
        r2_full_df.to_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_all_{ptrn}_change.csv'))


###############################################################
# Here I want to pull out the R2 scores to compare
SAVE_FOLDER = 'data/interim/230322_pred_change_from_phenome'

SAVE_FOLDER = 'data/interim/230626_pred_abs_change_from_phenome'
SAVE_FOLDER = 'data/interim/230918_pred_change_from_phenome'
SAVE_FOLDER = 'data/interim/231118_pred_change_from_phenome'


r2_all_ptrn = {}
for fldn in os.listdir(SAVE_FOLDER):
    if fldn.startswith('.'):
        continue
    r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'r2_bounds_{fldn}.csv'), index_col=0)
    d1 = r2_curr['Mean_Adj'].to_dict()
    r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'r2_bounds_{fldn}_change.csv'), index_col=0)
    d2 = r2_curr['Mean_Adj'].to_dict()
    r2_all_ptrn[fldn] = {**d1, **d2}
r2_all_ptrn = pd.DataFrame.from_dict(r2_all_ptrn)
r2_all_ptrn_abs = r2_all_ptrn.copy()

SAVE_FOLDER = 'data/interim/230626_pred_change_from_phenome'
SAVE_FOLDER = 'data/interim/230918_pred_abs_change_from_phenome'
SAVE_FOLDER = 'data/interim/231118_pred_abs_change_from_phenome'

r2_all_ptrn = {}
for fldn in os.listdir(SAVE_FOLDER):
    if fldn.startswith('.'):
        continue
    r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'r2_bounds_{fldn}.csv'), index_col=0)
    d1 = r2_curr['Mean_Adj'].to_dict()
    r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'r2_bounds_{fldn}_change.csv'), index_col=0)
    d2 = r2_curr['Mean_Adj'].to_dict()
    r2_all_ptrn[fldn] = {**d1, **d2}
r2_all_ptrn = pd.DataFrame.from_dict(r2_all_ptrn)

r2_all_ptrn_all = r2_all_ptrn.join(r2_all_ptrn_abs, lsuffix='_LBAC', rsuffix='_MBAC')

plt.figure()
sns.heatmap(r2_all_ptrn[labels[0:33]].corr(), square=True, linecolor='k', linewidth=0.1)
plt.title("Correlation of R2 Scores")
plt.tight_layout()

plt.figure()
sns.heatmap(r2_all_ptrn_all.corr(), square=True, linecolor='k', linewidth=0.1,cmap='coolwarm', center=0)
plt.title("Correlation of R2 Scores")
plt.tight_layout()

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=15)
svd.fit(r2_all_ptrn_all.T)

###############################################################
# Circular Bar Chart of R2 scores
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap

#  Some layout stuff -----------------

cat_names = np.asarray(['early life factors', 'lifestyle - general',
                        'lifestyle -\nexercise & work', 'lifestyle - alcohol',
                        'lifestyle - tobacco', 'physical - general', 'physical - cardiac',
                        'blood assays', 'physical - bone\ndensity & sizes',
                        'cognitive\nphenotypes', 'mental health\nself-report'])

colors = sns.color_palette('bright', n_colors=11)

cat_short_rev = {'blood': 'blood assays',
                'cognitive': 'cognitive\nphenotypes',
                'early_life': 'early life factors',
                'alcohol': 'lifestyle - alcohol',
                'exercise': 'lifestyle -\nexercise & work',
                'lifestyle_general': 'lifestyle - general',
                'tobacco': 'lifestyle - tobacco',
                'mental_health': 'mental health\nself-report',
                'bone': 'physical - bone\ndensity & sizes',
                'cardiac': 'physical - cardiac',
                'physical': 'physical - general',
                'age_sex_wait_only': 'Age/Sex &\n Time between Visits'
                }
# Move Grey to correct area
c = colors[5]
colors[5] = colors[7]
colors[7] = c

# Switch brown and yellow
c = colors[7]
colors[7] = colors[8]
colors[8] = c
colors = {cat_names[i]:c for i, c in enumerate(colors)}
colors['Age/Sex &\n Time between Visits'] = 'olive'

VIS_FOLDER = 'reports/Project2_Longitudional/Long_Fig_Drafts/Fig2_Phenome/REDONE_PHENO_0923'

USE_ABS = True; SAVE_FOLDER = 'data/interim/231118_pred_abs_change_from_phenome'
USE_ABS = False; SAVE_FOLDER = 'data/interim/231118_pred_change_from_phenome'


ptrn = 'Pattern 6'
for ptrn_num in tqdm(range(1, 33+1)):
    ptrn = f'Pattern {ptrn_num}'
    r2_curr1 = pd.read_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_bounds_{ptrn}_change.csv'), index_col=0)
    r2_curr2 = pd.read_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_bounds_{ptrn}.csv'), index_col=0)
    r2_ptrn = pd.concat([r2_curr1, r2_curr2])

    r2_ptrn = r2_ptrn.sort_index()
    r2_ptrn['color'] = r2_ptrn.index.map(lambda x: colors[cat_short_rev[x.split('_Change')[0]]])
    r2_ptrn['hatch'] = r2_ptrn.index.map(lambda x: 'xxx' if '_Change' in x else None)

    # Decide when to switch colours
    lns = r2_ptrn.index.map(lambda x: x.split('_Change')[0])
    lns = [lns[i] != lns[i-1] for i in range(1,len(r2_ptrn))]
    lns = [True, *lns]
    # -----------------------------
    # Initialize layout in polar coordinates
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={"projection": "polar"})

    # Set background color to white, both axis and figure.
    ANGLES = np.linspace(0.03, 2 * np.pi - 0.03, len(r2_ptrn), endpoint=False)
    # r2_all_ptrn.index.map(lambda x: x.split('_Change')[0])
    Y_LIM = 0.25
    theta_offset = 0.8* np.pi / 2
    ax.set_theta_offset(theta_offset)
    ax.set_ylim(-0.1, Y_LIM)

    # Add geometries to the plot -------------------------------------
    # See the zorder to manipulate which geometries are on top

    # # Add bars to represent the cumulative track lengths
    # ax.bar(ANGLES, r2_all_ptrn[ptrn], color=r2_all_ptrn['color'],
    #         edgecolor='k', hatch=r2_all_ptrn['hatch'], alpha=0.9, width=0.22, zorder=10)
    ax.bar(ANGLES, r2_ptrn['Mean_Adj'], color=r2_ptrn['color'],
            edgecolor='k', hatch=r2_ptrn['hatch'], alpha=0.9, width=0.22, zorder=10,
            capsize=0, yerr= r2_ptrn[[f'Lower_{th}_Adj', f'Upper_{100-th}_Adj']].T.values)

    # Add dashed vertical lines. These are just references
    ax.vlines(ANGLES[lns]-0.15, -0.1, Y_LIM, color='k', ls=(0, (4, 4)), zorder=11)

    # Fix Labels
    clbl = r2_ptrn.index.map(lambda x: cat_short_rev[x.split('_Change')[0]])[lns]
    for  angle, label, color in zip( ANGLES[lns], clbl, r2_ptrn['color'].values[lns]):
        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle + theta_offset)

        # Flip some labels upside down
        alignment = ""
        if angle  <= np.pi:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle - 0.02,
            y=Y_LIM-0.05,
            s=label,
            ha=alignment,
            va='center',
            color = color,
            rotation=rotation,
            size=10,
            rotation_mode="anchor")

    # Add Legend
    import matplotlib.patches as mpatches

    a_val = 0.6
    circ1 = mpatches.Patch( facecolor='grey',edgecolor='k', alpha=a_val,hatch=None,label='Baseline')
    circ2= mpatches.Patch( facecolor='grey',edgecolor='k', alpha=a_val,hatch='xxx',label='Change')


    ax.legend(handles = [circ1,circ2],loc=2, fontsize=10)

    ax.set_xticks([])
    ax.xaxis.grid(False)
    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")

    # Add Labels for R2 Score
    PAD = 0.0040
    locs, lbls = plt.yticks()
    for lc, lb in zip(locs, lbls):
        ax.text(-0.2 * np.pi / 2, lc + PAD, f'{lc:.2f}', ha="center", size=8)
    ax.set_yticklabels([])
    ax.text(-0.2 * np.pi / 2+0.14, Y_LIM/2, "Adjusted R2", ha="left",
            rotation = np.rad2deg(-0.2 * np.pi / 2+theta_offset), size=10)

    ax.text(0, -0.1, *[ptrn+'\nMBAC' if USE_ABS else ptrn+'\nLBAC'], ha="center", size=12, zorder = 12,
            bbox=dict(facecolor='white', alpha=0.8, linewidth=0))

    # Add line at y=0
    fig = plt.gcf()
    max_wind_circle = plt.Circle((0, 0), 0.1, transform=ax.transData._b,
                    fill=False, edgecolor='k', linewidth=2, alpha=1, zorder=9)
    fig.gca().add_artist(max_wind_circle)

    plt.savefig(os.path.join(VIS_FOLDER, f"behav2brain_{'MBAC' if USE_ABS else 'LBAC'}", f"behav2brain_ptrn{ptrn_num}_{'MBAC' if USE_ABS else 'LBAC'}_v3_age_sex.pdf"))
    plt.close()
############################################################################################

# I want to get the areas that are strongly driving a particular pattern
# Using the coefs whose beta do not cross zero in the 2.5/97.5 threshold
ptrn = 'Pattern 2'
SAVE_FOLDER = 'data/interim/230626_pred_abs_change_from_phenome'
SAVE_FOLDER = 'data/interim/230626_pred_change_from_phenome'

USE_ABS = True; SAVE_FOLDER = 'data/interim/231118_pred_abs_change_from_phenome'
USE_ABS = False; SAVE_FOLDER = 'data/interim/231118_pred_change_from_phenome'
coef_bounds = pd.read_csv(os.path.join(SAVE_FOLDER, ptrn, f'coefs_bounds_{cat_short[cat]}.csv'), index_col =0)

# Find the signs of the upper/lower bound
# keep only coefs where the upper/lower bound have the same sign (i.e. s1 * s2 = +1)
th = 2.5
coef_mask = np.sign(coef_bounds[f'{cat_short[cat]}_{th}%'])*np.sign(coef_bounds[f'{cat_short[cat]}_{100-th}%'])
coef_mask = coef_mask>0

sig_coefs = coef_bounds[coef_mask]
print(ptrn, ' ', cat_short[cat].upper())
print(*[(k, y_desc_dict[man_plot.translate(k, y_cat_dict)[0]], sig_coefs.loc[k][f'{cat_short[cat]}']) for k in sig_coefs.index if k not in ['r2', 'intercept', *baseline]], sep='\n')
print(*[(k, sig_coefs.loc[k][f'{cat_short[cat]}']) for k in sig_coefs.index if k in ['r2', 'intercept', *baseline]], sep='\n')

coef_bounds = pd.read_csv(os.path.join(SAVE_FOLDER, ptrn, f'coefs_bounds_{cat_short[cat]}_change.csv'), index_col =0)

# Find the signs of the upper/lower bound
# keep only coefs where the upper/lower bound have the same sign (i.e. s1 * s2 = +1)
th = 2.5
coef_mask = np.sign(coef_bounds[f'{cat_short[cat]}_Change_{th}%'])*np.sign(coef_bounds[f'{cat_short[cat]}_Change_{100-th}%'])
coef_mask = coef_mask>0

sig_coefs = coef_bounds[coef_mask]
print('\n\n', ptrn, ' Change in ', cat_short[cat].upper())
print(*[(k, y_desc_dict[man_plot.translate(k, y_cat_dict)[0]], sig_coefs.loc[k][f'{cat_short[cat]}_Change']) for k in sig_coefs.index if k not in ['r2', 'intercept', 'Waittime', 'sex_F', 'age_v2']], sep='\n')
print(*[(k, sig_coefs.loc[k][f'{cat_short[cat]}_Change']) for k in sig_coefs.index if k in ['r2', 'intercept', 'Waittitme', 'sex_F', 'age_v2']], sep='\n')


# For all patterns
# Is 6142#2 relevant?

# Change version

COEF_2_FIND = '6142#2'
COEF_2_FIND = '20111#10'
COEF_2_FIND = '6141#4'
COEF_2_FIND = '1031'
cat = all_cats[5] # Lifestyle - General

SAVE_FOLDER_ABS = 'data/interim/230626_pred_abs_change_from_phenome'
SAVE_FOLDER_PCA = 'data/interim/230626_pred_change_from_phenome'

SAVE_FOLDER_ABS = 'data/interim/231118_pred_abs_change_from_phenome'
SAVE_FOLDER_PCA = 'data/interim/231118_pred_change_from_phenome'
rtr_rel_all = {}
for ptrn in labels[0:33]:
    coef_bounds = pd.read_csv(os.path.join(SAVE_FOLDER_PCA, ptrn, f'coefs_bounds_{cat_short[cat]}_change.csv'), index_col =0)
    rtr_rel = coef_bounds.loc[COEF_2_FIND]
    rtr_rel_pca = np.sign(rtr_rel[f'{cat_short[cat]}_Change_{th}%'])*np.sign(rtr_rel[f'{cat_short[cat]}_Change_{100-th}%'])

    coef_bounds = pd.read_csv(os.path.join(SAVE_FOLDER_ABS, ptrn, f'coefs_bounds_{cat_short[cat]}_change.csv'), index_col =0)
    rtr_rel = coef_bounds.loc[COEF_2_FIND]
    rtr_rel_abs = np.sign(rtr_rel[f'{cat_short[cat]}_Change_{th}%'])*np.sign(rtr_rel[f'{cat_short[cat]}_Change_{100-th}%'])

    rtr_rel_all[ptrn] = {'MBAC': rtr_rel_abs>0, 'LBAC': rtr_rel_pca>0}
rtr_rel_all = pd.DataFrame().from_dict(rtr_rel_all).T

# General (one timepoint)
COEF_2_FIND = '6142#2'
COEF_2_FIND = '20111#10'
COEF_2_FIND = '6141#1'
COEF_2_FIND = '709'
cat = all_cats[5] # Lifestyle - General

SAVE_FOLDER_ABS = 'data/interim/230626_pred_abs_change_from_phenome'
SAVE_FOLDER_PCA = 'data/interim/230626_pred_change_from_phenome'
rtr_rel_all = {}
for ptrn in labels[0:33]:
    coef_bounds = pd.read_csv(os.path.join(SAVE_FOLDER_PCA, ptrn, f'coefs_bounds_{cat_short[cat]}.csv'), index_col =0)
    rtr_rel = coef_bounds.loc[COEF_2_FIND]
    rtr_rel_pca = np.sign(rtr_rel[f'{cat_short[cat]}_{th}%'])*np.sign(rtr_rel[f'{cat_short[cat]}_{100-th}%'])

    coef_bounds = pd.read_csv(os.path.join(SAVE_FOLDER_ABS, ptrn, f'coefs_bounds_{cat_short[cat]}.csv'), index_col =0)
    rtr_rel = coef_bounds.loc[COEF_2_FIND]
    rtr_rel_abs = np.sign(rtr_rel[f'{cat_short[cat]}_{th}%'])*np.sign(rtr_rel[f'{cat_short[cat]}_{100-th}%'])

    rtr_rel_all[ptrn] = {'MBAC': rtr_rel_abs>0, 'LBAC': rtr_rel_pca>0}
rtr_rel_all = pd.DataFrame().from_dict(rtr_rel_all).T
############################################################################################
# PCA of coefs
SAVE_FOLDER = 'data/interim/pred_change_from_phenome'
SAVE_FOLDER = 'data/interim/230324_pred_abs_change_from_phenome'
SAVE_FOLDER = 'data/interim/230322_pred_change_from_phenome'
SAVE_FOLDER = 'data/interim/230626_pred_change_from_phenome'

SAVE_FOLDER_ABS = 'data/interim/230626_pred_abs_change_from_phenome'
SAVE_FOLDER_PCA = 'data/interim/230626_pred_change_from_phenome'

SAVE_FOLDER_ABS = 'data/interim/230918_pred_abs_change_from_phenome'
SAVE_FOLDER_PCA = 'data/interim/230918_pred_change_from_phenome'

SAVE_FOLDER_ABS = 'data/interim/231118_pred_abs_change_from_phenome'
SAVE_FOLDER_PCA = 'data/interim/231118_pred_change_from_phenome'

SAVE_FOLDER = SAVE_FOLDER_PCA

cat = all_cats[2] # ELF
cat = all_cats[1] # Cognition
cat = all_cats[0] # Blood
cat = all_cats[7] # Mental Health
cat = all_cats[5] # Lifestyle - General
cat = all_cats[3] # Alcohol
cat = all_cats[10] # Physical - General

# Extract Baseline Coefs
coef_all_ptrn = {}
for fldn in os.listdir(SAVE_FOLDER):
    if fldn.startswith('.'):
        continue
    r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'coefs_bounds_{cat_short[cat]}.csv'), index_col=0)
    coef_all_ptrn[fldn] =  r2_curr[cat_short[cat]].to_dict()
    # r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'r2_bounds_{fldn}_change.csv'), index_col=0)
    # d2 = r2_curr['Mean_Adj'].to_dict()
    # # r2_all_ptrn[fldn] = {**d1, **d2}
    # if coef_all_ptrn is not None:
    #     coef_all_ptrn = pd.concat([coef_all_ptrn, r2_curr])
    # else:
    #     coef_all_ptrn = r2_curr
coef_all_ptrn = pd.DataFrame.from_dict(coef_all_ptrn)
# coef_all_ptrn.index = ['r2', 'intercept', *cat_cols_pheno[cat]]

# Extract Change Coefs
coef_all_ptrn = {}
for fldn in os.listdir(SAVE_FOLDER):
    if fldn.startswith('.'):
        continue
    r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'coefs_bounds_{cat_short[cat]}_change.csv'), index_col=0)
    coef_all_ptrn[fldn] =  r2_curr[cat_short[cat]+'_Change'].to_dict()
    # r2_curr = pd.read_csv(os.path.join(SAVE_FOLDER, fldn, f'r2_bounds_{fldn}_change.csv'), index_col=0)
    # d2 = r2_curr['Mean_Adj'].to_dict()
    # # r2_all_ptrn[fldn] = {**d1, **d2}
    # if coef_all_ptrn is not None:
    #     coef_all_ptrn = pd.concat([coef_all_ptrn, r2_curr])
    # else:
    #     coef_all_ptrn = r2_curr
coef_all_ptrn = pd.DataFrame.from_dict(coef_all_ptrn)
# coef_all_ptrn.index = ['r2', 'intercept', *cat_cols_chg[cat]]

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

sq_size = 32
# pca_sq = TruncatedSVD(n_components=sq_size)
pca_sq = PCA(n_components=sq_size)
X_t = pca_sq.fit_transform(coef_all_ptrn.iloc[2:, :].T)

pccomp = pd.DataFrame(pca_sq.components_, columns = coef_all_ptrn.index[2:]).T

print('\n\n', 'Change in ' if 'Change' in r2_curr.columns[0] else '',
      cat.upper(), 'MBAC' if 'abs' in SAVE_FOLDER else 'LBAC')
for ipc in range(4):
    print("PC Comp ", ipc+1)
    print("\t Explained Var:", f'{100*pca_sq.explained_variance_ratio_[ipc]:.3f}')

    for k in pccomp[ipc].abs().nlargest().items():
        if k[0] not in ['age_v2', 'sex_F', 'Waittime']:
            print((k[0], y_desc_dict[man_plot.translate(k[0], y_cat_dict)[0]],
            pccomp[ipc].loc[k[0]]))
        else:
            print(k[0], k[1])

    print('\n')

pca_sq.explained_variance_ratio_[pca_sq.explained_variance_ratio_.argsort()]




pca_sq.explained_variance_ratio_
pccomp[0].abs().nlargest()


############################################################################################
# Editors wanted violin plots instead of bar charts


VIS_FOLDER = 'reports/Project2_Longitudional/Long_Fig_Drafts/Fig2_Phenome/Violin_Plot_0325'

USE_ABS = True; SAVE_FOLDER = 'data/interim/231118_pred_abs_change_from_phenome'
USE_ABS = False; SAVE_FOLDER = 'data/interim/231118_pred_change_from_phenome'

ptrn = 'Pattern 6'
for ptrn_num in tqdm(range(1, 33+1)):
    ptrn = f'Pattern {ptrn_num}'
    r2_curr1 = pd.read_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_bounds_{ptrn}_change.csv'), index_col=0)
    r2_curr2 = pd.read_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_bounds_{ptrn}.csv'), index_col=0)
    r2_ptrn = pd.concat([r2_curr1, r2_curr2])

    r2_curr1 = pd.read_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_all_{ptrn}_change.csv'), index_col=0)
    r2_curr2 = pd.read_csv(os.path.join(SAVE_FOLDER, ptrn, f'r2_all_{ptrn}.csv'), index_col=0)
    r2_ptrn_all = r2_curr1.merge(r2_curr2, left_index=True, right_index=True)

    r2_ptrn = r2_ptrn.sort_index()
    r2_ptrn['color'] = r2_ptrn.index.map(lambda x: cat_short_rev[x.split('_Change')[0]])
    r2_ptrn['color'] = r2_ptrn['color'].map(lambda x: colors[x])
    r2_ptrn['hatch'] = r2_ptrn.index.map(lambda x: 'xxx' if '_Change' in x else None)

    # Decide when to switch colours
    lns = r2_ptrn.index.map(lambda x: x.split('_Change')[0])
    lns = [lns[i] != lns[i-1] for i in range(1,len(r2_ptrn))]
    lns = [True, *lns]

    # Make adjustments to the r2_ptrn_all
    # To fit the order of what we have
    # And to select only adjusted
    r2style = '_Adj' # Or _Org
    r2_ptrn_all_cut = r2_ptrn_all.filter(like=r2style, axis=1)
    r2_ptrn_all_cut.columns = [c.split(r2style)[0] for c in r2_ptrn_all_cut.columns]
    r2_ptrn_all_cut = r2_ptrn_all_cut[r2_ptrn.index]

    # -----------------------------
    # Initialize layout in polar coordinates
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={"projection": "polar"})

    # Set background color to white, both axis and figure.
    ANGLES = np.linspace(0.03, 2 * np.pi - 0.03, len(r2_ptrn), endpoint=False)
    # r2_all_ptrn.index.map(lambda x: x.split('_Change')[0])
    Y_LIM = 0.3
    theta_offset = 0.8* np.pi / 2
    ax.set_theta_offset(theta_offset)
    ax.set_ylim(-0.1, Y_LIM)

    # Add geometries to the plot -------------------------------------
    # See the zorder to manipulate which geometries are on top

    # # Add bars to represent the cumulative track lengths
    # ax.bar(ANGLES, r2_all_ptrn[ptrn], color=r2_all_ptrn['color'],
    #         edgecolor='k', hatch=r2_all_ptrn['hatch'], alpha=0.9, width=0.22, zorder=10)
    # ax.bar(ANGLES, r2_ptrn['Mean_Adj'], color=r2_ptrn['color'],
    #         edgecolor='k', hatch=r2_ptrn['hatch'], alpha=0.9, width=0.22, zorder=10,
    #         capsize=0, yerr= r2_ptrn[[f'Lower_{th}_Adj', f'Upper_{100-th}_Adj']].T.values)



    # Plot violin plots
    violin_parts = ax.violinplot(r2_ptrn_all_cut, positions=ANGLES, widths=0.22, showmeans=True, showextrema=True, showmedians=False)

    # Customize each violin with color and hatch pattern
    for i, pc in enumerate(violin_parts['bodies']):


        # Apply the hatch pattern
        pc.set_hatch(r2_ptrn['hatch'][i])
        # Apply the color
        pc.set_facecolor(r2_ptrn['color'][i])
        pc.set_alpha(0.9)
        pc.set_edgecolor('k')  # Optional, edge color


    # Adjust the appearance of the lines (violins) specifically if needed
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        vp = violin_parts[partname]
        vp.set_edgecolor((0, 0, 0, 0.5))  # Set the edges of these elements to softened black

    # Add dashed vertical lines. These are just references
    ax.vlines(ANGLES[lns]-0.15, -0.1, Y_LIM, color='k', ls=(0, (4, 4)), zorder=11)

    # Fix Labels
    clbl = r2_ptrn.index.map(lambda x: cat_short_rev[x.split('_Change')[0]])[lns]
    for  angle, label, color in zip( ANGLES[lns], clbl, r2_ptrn['color'].values[lns]):
        # Move some labels
        if any(map(label.__contains__, ['ognitive', 'ental', 'eneral'])): # remove first letter bc case-sensitive
            angle = angle + np.diff(ANGLES)[0]

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle + theta_offset)

        # Flip some labels upside down
        alignment = ""
        if angle  <= np.pi:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle - 0.02,
            y=Y_LIM-0.05,
            s=label,
            ha=alignment,
            va='center',
            color = color,
            rotation=rotation,
            size=10,
            rotation_mode="anchor")

    # Add Legend
    import matplotlib.patches as mpatches

    a_val = 0.6
    circ1 = mpatches.Patch( facecolor='grey',edgecolor='k', alpha=a_val,hatch=None,label='Baseline')
    circ2= mpatches.Patch( facecolor='grey',edgecolor='k', alpha=a_val,hatch='xxx',label='Change')


    ax.legend(handles = [circ1,circ2],loc=7, fontsize=10)

    ax.set_xticks([])
    ax.xaxis.grid(False)
    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")

    # Add Labels for R2 Score
    PAD = 0.0040
    locs, lbls = plt.yticks()
    for lc, lb in zip(locs, lbls):
        ax.text(-0.2 * np.pi / 2, lc + PAD, f'{lc:.2f}', ha="center", size=8)
    ax.set_yticklabels([])
    ax.text(-0.2 * np.pi / 2+0.14, Y_LIM/2, "Adjusted R2", ha="left",
            rotation = np.rad2deg(-0.2 * np.pi / 2+theta_offset), size=10)

    ax.text(0, -0.1, *[ptrn+' MBAC' if USE_ABS else ptrn+' LBAC'], ha="center", size=12, zorder = 12,
            bbox=dict(facecolor='white', alpha=0.8, linewidth=0))

    # Add line at y=0
    fig = plt.gcf()
    max_wind_circle = plt.Circle((0, 0), 0.1, transform=ax.transData._b,
                    fill=False, edgecolor='k', linewidth=2, alpha=1, zorder=9)
    fig.gca().add_artist(max_wind_circle)

    plt.savefig(os.path.join(VIS_FOLDER, f"behav2brain_{'MBAC' if USE_ABS else 'LBAC'}", f"behav2brain_ptrn{ptrn_num}_{'MBAC' if USE_ABS else 'LBAC'}_violin.pdf"))
    plt.close()
