# We now have data that has been PCA'ed

# I may be missing import statements, so add if you find something I missed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)

DATA_FOLDER = 'data/interim/icd10_work/full_data'
IN_FOLDER = 'data/processed/ICD_work/PCA_Pipeline/PCA_transformed2'

phecodes_idx = pd.read_csv(os.path.join(DATA_FOLDER, 'phecodes.csv'), index_col = 0)
ICD_DDR_small = pd.read_csv(os.path.join('models/ICD_Main_PCA', 'ICD_DDR_all_brain_cohort_GOOD.csv'), index_col = 0)

# DATA_FOLD = 'data/interim/icd10_work'

# # Load in
# ppl_xcldphe = pd.read_csv(os.path.join(DATA_FOLD, 'proc', 'exclusion.csv'), index_col = 0)
# ppl_phecodes_small = pd.read_csv(os.path.join(DATA_FOLD, 'proc', 'inclusion.csv'), index_col = 0)


df_pca_chg_rate = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Asymm_Change.csv'), index_col=0)
df_amt_chg_rate = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Amount_Change.csv'), index_col=0)

# df_pca_change = pd.read_csv(os.path.join('models/Asymm_Patterns/Changes/Asym_pca_changes.csv'), index_col=0)

# Dictionary to translate cat code to original value
n_code=4
all_cats = np.unique(phecodes_idx['group'])
dict_cat_code = {curr_cat[0:n_code].upper(): curr_cat for curr_cat in all_cats}
groupid = phecodes_idx.groupby(['group', 'groupid']).size().reset_index()[['group', 'groupid']].values
groupid = {g:c for (g, c) in groupid}

def ICD_correlat(comp_nlz, ICD_PCA):
    """
    Most important function imo
    Conducts correlations between brain and all medical codes
    Will likely take the longest to run of all functions

    INPUT NOTES:
    comp_nlz
    First dataframe is brain related phenotypes
    (or generally what is being compared against phenotypes)
    It is expected to have 'eid's as the INDEX
    to connect it to the medical diagnoses file

    ppl_phecodes, ppl_xcldphe
    Second two dataframe are medical code phenotypes
    First is positive cases
    Second is people who should be excluded from the control group
    obtained through the load ICD function
    Should have INDEX columns of eid
    to connect it to the brain data

    phecodes_idx
    This dataframe has the definitions of all phecodes we have
    obtained through the load ICD function
    This will be used to innclude human readable labels
    and groupings


    OUTPUT NOTES:
    Returns a dataframe containing 1447 rows x n columns
    There are 2 columns which describe the phenotype
    (biobank designation, category)
    one column is an index ('i')
    The index is multiple index and includes the group id and phecode
    and 3 x m columns, where m is the number of brain-related
    columns to compare behaviour to
    r value, p value, and -log10(p) is retained
    for each behaviour/brain related combo

    If you don't like the multiindex use the reset_index() command
    But note this may interfere with the plotting portion of the coding
    """
    cols = ICD_PCA.columns

    keys = np.concatenate([
        ['groupid', 'group', 'phecode', 'type'], # At some point adding a description may be needed
        [f"-logp_{c}" for c in comp_nlz],
        [f"r_{c}" for c in comp_nlz],
        [f"p_{c}" for c in comp_nlz]
    ])

    print("Merging Data...")
    new_use = pd.merge(ICD_PCA, comp_nlz, on='eid')
    # in original code, we had something to deal with the binary nature of ICD data
    # where if there were fewer than min_hits positive cases we drop the disease altogether
    # With condensed, summarized data, we don't need this

    # We also do not need to account for exclusions

    mnhtn_data = {key: [] for key in keys}
    not_enough = []
    print("Running Analyses")
    for col in tqdm(cols):
        col = str(col)
        cat_code = col.split('-')[0]
        gr = dict_cat_code[cat_code]

        # First record column and category
        mnhtn_data['phecode'].append(col)
        mnhtn_data['type'].append(col.split('_')[-1])
        mnhtn_data['groupid'].append(groupid[gr])
        mnhtn_data['group'].append(gr)

        # Now compute pearson correlation
        for comp in comp_nlz:

            # we need to deal with nan values
            keep = ~np.logical_or(np.isnan(new_use[comp]),
                                    np.isnan(new_use[col]))

            if len(np.unique(new_use[col][~np.isnan(new_use[col])])) == 1:
                # Check that y output has at least two unique points
                # print(f'{col} triggered < 2 unique y values')
                not_enough.append(col)
                mnhtn_data[f"r_{comp}"].append(np.nan)
                mnhtn_data[f"p_{comp}"].append(np.nan)
                mnhtn_data[f"-logp_{comp}"].append(np.nan)
            elif keep.sum() >= 2:
                r, p = pearsonr(new_use[comp][keep], new_use[col][keep])
                mnhtn_data[f"r_{comp}"].append(r)
                mnhtn_data[f"p_{comp}"].append(p)
                mnhtn_data[f"-logp_{comp}"].append(-np.log10(p))
            else:
                # print(f"Fewer than 2 non-nan pts exist for {col} / {comp} combo")
                not_enough.append(col)
                mnhtn_data[f"r_{comp}"].append(np.nan)
                mnhtn_data[f"p_{comp}"].append(np.nan)
                mnhtn_data[f"-logp_{comp}"].append(np.nan)

    # Organize by Groups
    clreddf = pd.DataFrame(mnhtn_data)
    clreddf = clreddf.sort_values(by=['groupid'])
    clreddf = clreddf.reset_index(drop=True)
    clreddf['i'] = clreddf.index

    multi = clreddf.set_index(['groupid', 'phecode']).sort_index()
    multi['i'] = np.arange(len(multi))

    return multi.copy()


# corr_asm = ICD_correlat(df_pca_change, ICD_DDR_small)
# corr_asm = ICD_correlat(df_pca_change, icd_trans_new_df)
# corr_sym = ICD_correlat(df_symm_shift, icd_trans_new_df)
# corr_org = ICD_correlat(pain_score_test, ICD_DDR_small)
# corr_new = ICD_correlat(pain_score_test, icd_trans_new_df)

df_pca_chg_rate.index.name = 'eid'
df_amt_chg_rate.index.name = 'eid'

corr_chg = ICD_correlat(df_pca_chg_rate[labels[0:33]].reset_index(), ICD_DDR_small)
corr_amt = ICD_correlat(df_amt_chg_rate[labels[0:33]].reset_index(), ICD_DDR_small)

corr_chg = corr_chg.drop(columns=['-logp_eid', 'p_eid', 'r_eid'])
corr_amt = corr_amt.drop(columns=['-logp_eid', 'p_eid', 'r_eid'])


# corr_chg2 = ICD_correlat(df_pca_rate_SS[labels[0:33]], icd_trans_new_df)
# corr_amt2 = ICD_correlat(df_amt_rate_SS[labels[0:33]], icd_trans_new_df)


import processing.ks_1_0_MeDiWAS_fncs as MeDiWAS

# Unravel
look_at = corr_amt
corr = look_at.reset_index()
corr = corr.melt(id_vars=['groupid', 'phecode', 'group','type'])
tt = [ii.startswith('-logp_') for ii in corr['variable']]
corr_log = corr[tt]
corr_log = corr_log.rename(columns={'value':'-logp_Pattern'})
tt = [ii.startswith('p_') for ii in corr['variable']]
corr_log['p_Pattern'] = corr[tt]['value'].values
corr_log['i'] = np.arange(corr_log.shape[0])
corr_log['Pattern'] =  corr_log['variable'].map(lambda x: x.split('-logp_')[1])
corr_log =corr_log.set_index(['groupid', 'phecode']).sort_index()
corr_log['id'] =  corr_log.index.get_level_values(1).map(lambda x: x.split('-')[-1].split('_')[0])

lbl_thres=0.05*corr_log.shape[0]/(corr.shape[0]/3)
plot = MeDiWAS.manhattan_plot(corr_log, 'Pattern', label=None, n_t=58)
lk='Pattern'

plot.ax.axhline(-np.log10(0.05/58))

texts = [plot.ax.text(p[1]['i'], p[1][f'-logp_{lk}'],
                    '\n'.join([p[1]['Pattern'], p[1]['type']+' '+p[1]['id']]),
                    size = 'small', backgroundcolor='white', fontstretch='condensed',
                    bbox={'pad':0, 'ec':'white',  'alpha':0.1, 'color':'white'} )
        for p in  corr_log[corr_log[f'-logp_{lk}']>-np.log10(0.05/58)].iterrows()]


texts = [plot.ax.text(p[1]['i'], p[1][f'-logp_{lk}'],
                    '\n'.join([p[1]['Pattern'], p[1]['type']+' '+p[1]['id']]),
                    size = 'small', backgroundcolor='white', fontstretch='condensed',
                    bbox={'pad':0, 'ec':'white',  'alpha':0.1, 'color':'white'} )
        for p in  corr_log[corr_log[f'-logp_{lk}']>2.5].iterrows()]

# How many hits per pattern
thresBon = 0.05/icd_trans_new_df.shape[1]
thresBon = 0.05/(icd_trans_new_df.shape[1]/3)
n_Hits = {}
for lk in df_pca_change.columns:
    fdr = MeDiWAS.findFDR(corr_asm, lk, thresBon)
    n_Hits[lk] = (fdr, sum(corr_asm['p_'+lk] < fdr), sum(corr_asm['p_'+lk] < thresBon))
n_Hits = pd.DataFrame.from_dict(n_Hits, orient='index', columns=['FDR', 'Pass FDR', "Pass Bon"])

# Amount
# Make a plot with homemade Bon and FDR threshold
plot = MeDiWAS.manhattan_plot(corr_log_amt, 'Pattern', label=None, thres=None)
plt.title('Amount of Asymmetry Pattern Change')
plot.ax.axhline(-np.log10(0.05/corr_amt.shape[0]), c='k', ls='--')
FDR_list = []
for lbl in labels[0:33]:
    FDR_list.append(MeDiWAS.findFDR(corr_amt, lbl, 0.05/look_at.shape[0]))
FDR_list = np.asarray(FDR_list)

plot.ax.axhline(-np.log10(FDR_list.max()), c='k', ls='--')
# plt.savefig(os.path.join(VIS_FOLDER, "MEDI_Amount_Unlabelled.pdf"))

texts = [plot.ax.text(p[1]['i'], p[1][f'-logp_{lk}'],
                    '\n'.join([p[1]['Pattern'], p[1]['type']+' '+p[1]['id']]),
                    size = 'small', backgroundcolor='white', fontstretch='condensed',
                    bbox={'pad':0, 'ec':'white',  'alpha':0.1, 'color':'white'} )
        for p in  corr_log_amt[corr_log_amt[f'-logp_{lk}']>-np.log10(FDR_list.max())].iterrows()]
# plt.savefig(os.path.join(VIS_FOLDER, "MEDI_Amount_Labelled.pdf"))

# Change
# Make a plot with homemade Bon and FDR threshold
plot = MeDiWAS.manhattan_plot(corr_log_chg_v2, 'Pattern', label=None, thres=None)
plt.title('Amount of Asymmetry Pattern Change')
plot.ax.axhline(-np.log10(0.05/corr_chg.shape[0]), c='k', ls='--')
FDR_list = []
for lbl in labels[0:33]:
    FDR_list.append(MeDiWAS.findFDR(corr_chg, lbl, 0.05/look_at.shape[0]))
FDR_list = np.asarray(FDR_list)

plot.ax.axhline(-np.log10(FDR_list.max()), c='k', ls='--')
plt.savefig(os.path.join(VIS_FOLDER, "MEDI_Change_Unlabelled.pdf"))

texts = [plot.ax.text(p[1]['i'], p[1][f'-logp_{lk}'],
                    '\n'.join([p[1]['Pattern'], p[1]['type']+' '+p[1]['id']]),
                    size = 'small', backgroundcolor='white', fontstretch='condensed',
                    bbox={'pad':0, 'ec':'white',  'alpha':0.1, 'color':'white'} )
        for p in  corr_log_chg[corr_log_chg[f'-logp_{lk}']>-np.log10(FDR_list.max())].iterrows()]
plt.savefig(os.path.join(VIS_FOLDER, "MEDI_Change_Labelled.pdf"))


# Attempt a Miami Plot


look_at = corr_amt
corr = look_at.reset_index()
corr = corr.melt(id_vars=['groupid', 'phecode', 'group','type'])
tt = [ii.startswith('-logp_') for ii in corr['variable']]
corr_log = corr[tt].set_index(['groupid', 'phecode']).sort_index()
corr_log = corr_log.rename(columns={'value':'-logp_Pattern'})
tt = [ii.startswith('p_') for ii in corr['variable']]
corr_log['p_Pattern'] = corr[tt]['value'].values
tt = [ii.startswith('r_') for ii in corr['variable']]
corr_log['r_Pattern'] = corr[tt]['value'].values
corr_log['i'] = np.arange(corr_log.shape[0])
corr_log['id'] =  corr_log.index.get_level_values(1).map(lambda x: x.split('-')[-1].split('_')[0])
corr_log['Pattern'] =  corr_log['variable'].map(lambda x: x.split('-logp_')[1])
corr_log_amt = corr_log.copy()
corr_log_amt['Measure'] = "MBAC"
corr_log_amt = corr_log_amt.reset_index().set_index(['groupid', 'phecode', 'Measure'])

look_at = corr_chg
corr = look_at.reset_index()
corr = corr.melt(id_vars=['groupid', 'phecode', 'group','type'])
tt = [ii.startswith('-logp_') for ii in corr['variable']]
corr_log = corr[tt].set_index(['groupid', 'phecode']).sort_index()
corr_log = corr_log.rename(columns={'value':'-logp_Pattern'})
tt = [ii.startswith('p_') for ii in corr['variable']]
corr_log['p_Pattern'] = corr[tt]['value'].values
tt = [ii.startswith('r_') for ii in corr['variable']]
corr_log['r_Pattern'] = corr[tt]['value'].values
corr_log['i'] = np.arange(corr_log.shape[0])
corr_log['id'] =  corr_log.index.get_level_values(1).map(lambda x: x.split('-')[-1].split('_')[0])
corr_log['Pattern'] =  corr_log['variable'].map(lambda x: x.split('-logp_')[1])

corr_log_chg = corr_log.copy()
corr_log_chg['Measure'] = "LBAC"
corr_log_chg = corr_log_chg.reset_index().set_index(['groupid', 'phecode', 'Measure'])

#################################

look_at = corr_amt
corr = look_at.reset_index()
corr = corr.melt(id_vars=['groupid', 'phecode', 'group','type'])
corr = corr.set_index(['groupid', 'phecode', 'Pattern']).sort_index()
tt = [ii.startswith('-logp_') for ii in corr['variable']]
corr_log = corr[tt].set_index(['groupid', 'phecode', 'Pattern']).sort_index()
corr_log = corr_log.rename(columns={'value': '-logp_Pattern'})
corr_log = corr_log.merge(corr[tt]['value'], right_index = True, left_index= True )

tt = [ii.startswith('p_') for ii in corr['variable']]
corr_log['p_Pattern'] = corr[tt]['value'].values
tt = [ii.startswith('r_') for ii in corr['variable']]
corr_log['r_Pattern'] = corr[tt]['value'].values
corr_log['Pattern'] =  corr_log['variable'].map(lambda x: x.split('-logp_')[1])
corr_log =corr_log.set_index(['groupid', 'phecode']).sort_index()
corr_log['i'] = np.arange(corr_log.shape[0])
corr_log['id'] =  corr_log.index.get_level_values(1).map(lambda x: x.split('-')[-1].split('_')[0])
corr_log_amt = corr_log.copy()
corr_log_amt['Measure'] = "MBAC"
corr_log_amt = corr_log_amt.reset_index().set_index(['groupid', 'phecode', 'Measure'])


look_at = corr_chg
corr = look_at.reset_index()
corr = corr.melt(id_vars=['groupid', 'phecode', 'group','type'])
tt = [ii.startswith('-logp_') for ii in corr['variable']]
corr_log = corr[tt]
corr_log = corr_log.rename(columns={'value':'-logp_Pattern'})
tt = [ii.startswith('p_') for ii in corr['variable']]
corr_log['p_Pattern'] = corr[tt]['value'].values
tt = [ii.startswith('r_') for ii in corr['variable']]
corr_log['r_Pattern'] = corr[tt]['value'].values
corr_log['Pattern'] =  corr_log['variable'].map(lambda x: x.split('-logp_')[1])
corr_log = corr_log.set_index(['groupid', 'phecode']).sort_index()
corr_log['id'] =  corr_log.index.get_level_values(1).map(lambda x: x.split('-')[-1].split('_')[0])
corr_log['i'] = np.arange(corr_log.shape[0])
corr_log_chg = corr_log.copy()
corr_log_chg['Measure'] = "LBAC"
corr_log_chg = corr_log_chg.reset_index().set_index(['groupid', 'phecode', 'Measure'])

# Another attempt at FDR correction
# This time with harshest Bon, applied across both LBAC and MBAC associations
n_look = 33
thresBon = 0.05/(corr_amt.shape[0]*2*n_look) # 33 Patterns; 2 measures per
new_FDR = MeDiWAS.findFDR(pd.concat([corr_log_chg, corr_log_amt]), 'Pattern', thresBon)

n_corrections = 174 * 33 * 2
FDR_list = []
for lbl in labels[0:33]:
    FDR_list.append(MeDiWAS.findFDR(corr_chg, lbl, 0.05/n_corrections))
FDR_list_chg = np.asarray(FDR_list)

FDR_list = []
for lbl in labels[0:33]:
    FDR_list.append(MeDiWAS.findFDR(corr_amt, lbl, 0.05/n_corrections))
FDR_list_amt = np.asarray(FDR_list)

plot = MeDiWAS.miami_plot(corr_log_chg, corr_log_amt, 'Pattern', label=None, thres=None, n_t = n_corrections,
                          lbls=['Asymmetry Change', 'Amount of Asymmetry Change'],
                          thresFDRs = [FDR_list_chg.max(), FDR_list_amt.max()])
# plt.savefig(os.path.join(VIS_FOLDER, "MEDI_Miami_Change_Top.pdf"))

plot = MeDiWAS.miami_plot(corr_log_amt, corr_log_chg, 'Pattern', label=None, thres=None, n_t = n_corrections,
                          lbls=['Amount of Asymmetry Change', 'Asymmetry Change'],
                          thresFDRs = [FDR_list_amt.max(), FDR_list_chg.max()])

# plt.savefig(os.path.join(VIS_FOLDER, "MEDI_Miami_Abs_Top.pdf"))


# Aside
# Derm 4 is super strongly associated with pattern change.
# What is its phewas?
ICD_DDR_small['DERM-4_PCA']


import processing.manhattan_plot_util as man_plot

ukbb_y, y_desc_dict, y_cat_dict = man_plot.load_phenom(BASE_FOLDER = 'data/processed/')

disease_phewas = man_plot.phenom_correlat(ICD_DDR_small['DERM-4_PCA'].reset_index(), ukbb_y, y_desc_dict, y_cat_dict)

# We want to drop everything in mental health
cat_name = man_plot.cat_name(disease_phewas, y_cat_dict)

man_plot.manhattan_plot(disease_phewas, 'DERM-4_PCA', cat_name)

ppl_phecodes, ppl_xcldphe, phecodes_idx, groupid = MeDiWAS.load_ICD()
derm_mediwas = MeDiWAS.ICD_correlat(ICD_DDR_small['DERM-4_PCA'].reset_index(), ppl_phecodes, ppl_xcldphe, phecodes_idx)
MeDiWAS.manhattan_plot(derm_mediwas, 'DERM-4_PCA')
MeDiWAS.hits(derm_mediwas, 'DERM-4_PCA', sort=True)