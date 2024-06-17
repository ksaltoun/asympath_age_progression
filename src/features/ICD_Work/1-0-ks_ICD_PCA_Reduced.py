# We now have data that has been PCA'ed

# I may be missing import statements, so add if you find something I missed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)


def firstFlip(real, perm):
    """
    Finds FDR threshold in the standard method
    As used in the sklearn function, but modified to determine
    the p value threshold in real terms
    rather than chose which features to keep, which is what sklearn does
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
    """
    # # They should be sorted but if not
    # real = np.sort(real).flatten()
    # perm = np.sort(perm).flatten()

    real = real.flatten()
    perm = perm.flatten()

    # The magic part of FDR
    # Each p-val gets compared to the Bon thres of n - w tests
    # where w is the order (smallest to largest) of the p value of interest
    # and n is the total number of tests
    sel = real > perm
    # pass_thres is a binary array of if the ith p-value passes the FDR limit
    # We will now check that only the smallest (most significant) p values pass the FDR limit
    # Additionally, FDR is such that the first instance of when a p-value fails to pass the FDR limit is the FDR limit
    # So we will check for that too

    if sel[0] != True:
        # if the smallest p value doesn't pass the Thres then no values will pass thres
        n_pass = 1
    elif len(sel) > 0:
        # This ggets the index of the first value where the p values no longer pass the threshold
        first_switch = np.argwhere(np.diff(sel)).squeeze()
        if first_switch.shape  == ():
            # There is only one value where the passing criteria goes from true to false
            first_switch = first_switch[()]
        else:
            # There are many areas where the passing threshold gets crossed
            # We only select the firstion one
            first_switch = first_switch[0]
        # thresFDR = (sv <= sel.max()).sum() * thresBon
        # Plus 1 accounts for zero index
        n_pass = (first_switch +1)
    return(n_pass)


DATA_FOLDER = 'data/interim/icd10_work/full_data'
IN_FOLDER = 'data/processed/ICD_work/PCA_Pipeline/PCA_transformed2'

phecodes_idx = pd.read_csv(os.path.join(DATA_FOLDER, 'phecodes.csv'), index_col = 0)

n_code = 4

ICD_Transformed = pd.read_csv(os.path.join(IN_FOLDER, 'FULL_Transformed_Sel.csv'), index_col = 0)

cat_codes = [ii[0:n_code].upper() for ii in np.unique(phecodes_idx['group'])]
cat_codes_dict = {ii[0:n_code].upper():ii for ii in np.unique(phecodes_idx['group'])}
cat_codes_dict = {ii[0:n_code].upper():ii for ii in np.unique(phecodes_idx['group'])}
groupid = phecodes_idx.groupby(['group', 'groupid']).size().reset_index()[['group', 'groupid']].values
groupid = {g:c for (g, c) in groupid}

############################################################################################
# Load in
pc_comps = {}
pc_exvar = {}
pc_xvper = {}
pc_cpall = {}

for cat_code in cat_codes:
    pc_comps[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, f'{cat_code}_PC_Components_Sel.csv'), index_col = 0)
    pc_cpall[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, f'{cat_code}_PC_Components_All.csv'), index_col = 0)
    pc_exvar[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, f'{cat_code}_explained_var.csv'), index_col = 0)
    pc_xvper[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, 'Permutation', f'PERM_{cat_code}_explained_var.csv'), index_col = 0)
    pc_xvper[cat_code].columns = [cat_code+'-'+str(ii+1) for ii in range(pc_exvar[cat_code].shape[0])]

    # make it compatiable with ppl phecodes
    pc_comps[cat_code].columns = pc_comps[cat_code].columns.astype(float)
    pc_cpall[cat_code].columns = pc_cpall[cat_code].columns.astype(float)

############################################################################################
# For alternative way to calc significant components

n_flips = {cat_code:[] for cat_code in cat_codes}
for cat_code in cat_codes:
    for ii in range(5):
        n_f = firstFlip(pc_exvar[cat_code].values, pc_xvper[cat_code].loc[f'PERM_{ii+1}'].values)
        n_flips[cat_code].append(n_f)

############################################################################################
# Compare different selection criteria

var_thres_list = [40, 70, 50]
keys = ["code", "Group", 'N_Org', "N_Red", "Ex_Var", "N_Red_Alt", "Ex_Var_Alt"]
keys.extend([f"N_Red_{var_thres}" for var_thres in var_thres_list])
keys.extend([f"Ex_Var_{var_thres}" for var_thres in var_thres_list])

pc_summary = {key:[] for key in keys}
for cat_code in cat_codes:
    pc_summary["code"].append(cat_code)
    pc_summary["Group"].append(cat_codes_dict[cat_code])

    n_red =pc_comps[cat_code].shape[0]
    pc_summary["N_Red"].append(n_red)
    pc_summary["Ex_Var"].append(pc_exvar[cat_code][0:n_red].sum().values[0]*100)

    n_org = pc_comps[cat_code].shape[1]
    pc_summary["N_Org"].append(n_org)

    n_f_all = [firstFlip(pc_exvar[cat_code].values, pc_xvper[cat_code].loc[f'PERM_{ii+1}'].values) for ii in range(5)]
    n_f = np.ceil(np.mean(n_f_all)).astype(int)
    pc_summary["N_Red_Alt"].append(n_f)
    pc_summary["Ex_Var_Alt"].append(pc_exvar[cat_code][0:n_f].sum().values[0]*100)

    for var_thres in var_thres_list:
        n_f = np.arange(1, 1+n_org)[(pc_exvar[cat_code].cumsum() > var_thres/100).values.flatten()].min()
        pc_summary[f"N_Red_{var_thres}"].append(n_f)
        pc_summary[f"Ex_Var_{var_thres}"].append(pc_exvar[cat_code][0:n_f].sum().values[0]*100)


pc_summary = pd.DataFrame.from_dict(pc_summary)
pc_summary = pc_summary[keys].set_index('code')

############################################################################################

print('Extract top PCA contributors')
# Find top contributions for each PCA component
topn = 3
cat_contrib = {}
for cat in tqdm(cat_codes):
        n_pc = pc_summary.loc[cat]['N_Red_Alt']
        n_cols = pc_comps[cat].shape[1]
        for ii in range(n_pc):
            col = cat+f'-{ii+1}'
            cpca = pc_comps[cat].loc[col]
            t = np.argpartition(np.abs(cpca),-topn)[-topn:]
            t = np.bincount(t, minlength = n_cols).astype(bool)
            c = pc_comps[cat].columns[t].values
            cc = [phecodes_idx.loc[float(cc)]['description'] for cc in c]
            cat_contrib[col] = [*cc, *c, *cpca[t]]

cat_contrib = pd.DataFrame(cat_contrib,
                           index=[*[f'col {i+1}' for i in range(topn)],
                                  *[f'raw col {i+1}' for i in range(topn)],
                                  *[f'contrib {i+1}' for i in range(topn)]]).T

############################################################################################

# Extract only the PCs we are interested in
cross_cols = []
top50_cols = []
combo_cols = []
for cat in tqdm(cat_codes):
    cross_cols.extend([f'{cat}-{ii+1}' for ii in range(pc_summary['N_Red_Alt'].loc[cat])])
    top50_cols.extend([f'{cat}-{ii+1}' for ii in range(pc_summary['N_Red_50'].loc[cat])])
    combo_cols.extend([f'{cat}-{ii+1}' for ii in range(pc_summary[["N_Red_50", "N_Red_Alt"]].loc[cat].max())])

ICD_cross = ICD_Transformed[cross_cols]
ICD_top50 = ICD_Transformed[top50_cols]

ICD_cross.to_csv(os.path.join('data/processed/ICD_work/PCA_Pipeline', 'ICD_DDR_PCA.csv'))

############################################################################################
# Plot top contributors
import seaborn as sns
import matplotlib.pyplot as plt
import processing.ks_1_0_MeDiWAS_fncs as MeDiWAS
import textwrap

max_chars = 25

groupid, colorsdict = MeDiWAS.default_order()

# Make Pain Magn Contributions

out_fold = 'reports/ICD_Work/Domain_Reduced'
topn=6

from pathlib import Path

filename=os.path.join(out_fold, f'PCA')
Path(filename).mkdir(parents=True, exist_ok=True)

# [63:68] [72:81]
# for col in tqdm(np.asarray(combo_cols)[[30:40, 20]]):
for col in tqdm(cross_cols):
    cat = col.split('-')[0]


    filename=os.path.join(out_fold, f'PCA2', cat)
    Path(filename).mkdir(parents=True, exist_ok=True)

    # t = np.argpartition(np.abs(pc_cpall[cat].loc[col]),-topn)[-topn:]
    # small = pc_cpall[cat].loc[col].iloc[t.values].to_frame()

    small = pc_cpall[cat].loc[col].abs().nlargest(topn).to_frame()
    small[col] = pc_cpall[cat].loc[col].loc[small.index]
    small['color'] = phecodes_idx.loc[small.index.values.astype(float)]['group'].values
    small['color'] = small['color'].map(colorsdict)
    desc = phecodes_idx.loc[small.index.values.astype(float)]['description'].values
    desc = ['\n'.join(textwrap.wrap(d, width=max_chars, break_on_hyphens=False)) for d in desc]
    small['desc'] = desc

    plt.figure()
    plot = sns.barplot(data=small.reset_index(), x='desc', y=col, color=small['color'].iloc[0], errorbar=None, edgecolor='k')
    plt.xlabel('')
    plt.title(f"Top {topn} Absolute Largest {col} Loadings ({100*pc_exvar[cat].loc[col][0]:.1f}%)")
    plt.ylabel('Loading')
    for item in plot.get_xticklabels():
        item.set_rotation(45)
        item.set_horizontalalignment('right')
    plt.axhline(0, color='k')
    plt.legend(loc='upper left')
    plt.tight_layout()


    plt.savefig(os.path.join(filename, f'{col}_top{topn}.pdf'))
    plt.close()

