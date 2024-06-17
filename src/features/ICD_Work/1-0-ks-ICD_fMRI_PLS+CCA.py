
# We now have data that has been PLS'ed/CCA'ed with fMRI as the informing variable

# I may be missing import statements, so add if you find something I missed
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)

DATA_FOLDER = 'data/interim/icd10_work/full_data'
IN_FOLDER = 'data/processed/ICD_work/fMRI_informed'

phecodes_idx = pd.read_csv(os.path.join(DATA_FOLDER, 'phecodes.csv'), index_col = 0)

n_code = 4
############################################################################################
# ICD_Transformed = pd.read_csv(os.path.join(IN_FOLDER, 'FULL_Transformed_Sel.csv'), index_col = 0)
fMRI_pc = pd.read_csv(os.path.join(IN_FOLDER, 'General', 'fMRI_100_PC_Components.csv'), index_col = 0)
fMRI_ve = pd.read_csv(os.path.join(IN_FOLDER, 'General', 'fMRI_100_PC_explained_var.csv'), index_col = 0)

cat_codes = [ii[0:n_code].upper() for ii in np.unique(phecodes_idx['group'])]
cat_codes_dict = {ii[0:n_code].upper():ii for ii in np.unique(phecodes_idx['group'])}
cat_codes_dict = {ii[0:n_code].upper():ii for ii in np.unique(phecodes_idx['group'])}
groupid = phecodes_idx.groupby(['group', 'groupid']).size().reset_index()[['group', 'groupid']].values
groupid = {g:c for (g, c) in groupid}

############################################################################################
# Load in CCA + PLS
pls_fMRI = {}
pls_icds = {}
pls_irot = {}

cca_fMRI = {}
cca_icds = {}
cca_irot = {}

for cat_code in cat_codes:
    pls_fMRI[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, 'PLS', f'{cat_code}_fMRI_loadings.csv'), index_col = 0)
    pls_icds[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, 'PLS', f'{cat_code}_ICD_loadings.csv'), index_col = 0)
    pls_irot[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, 'PLS', f'{cat_code}_ICD_rotation.csv'), index_col = 0)

    cca_fMRI[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, 'CCA', f'{cat_code}_fMRI_loadings.csv'), index_col = 0)
    cca_icds[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, 'CCA', f'{cat_code}_ICD_loadings.csv'), index_col = 0)
    cca_irot[cat_code] = pd.read_csv(os.path.join(IN_FOLDER, 'CCA', f'{cat_code}_ICD_rotation.csv'), index_col = 0)

cca_fcor = pd.read_csv(os.path.join(IN_FOLDER,'CCA_Correlation.csv'), index_col=0)
pls_fcor = pd.read_csv(os.path.join(IN_FOLDER,'PLS_Correlation.csv'), index_col=0)

############################################################################################################
# Bring in CCA/PLS and push out the brain subset of data

# this has 37441 individuals
brain_eid = pd.read_csv('data/Project_1/Processed/df_meta.csv')['eid']
ICD_PCA = pd.read_csv(os.path.join('models/ICD_Main_PCA', 'ICD_DDR_PCA.csv'), index_col=0)
keep_cols = ICD_PCA.columns
ICD_PLS = pd.read_csv('data/processed/ICD_work/fMRI_informed/PLS_Transformed.csv', index_col=0)
ICD_CCA = pd.read_csv('data/processed/ICD_work/fMRI_informed/CCA_Transformed.csv', index_col=0)
ICD_PLS = ICD_PLS[keep_cols]
ICD_CCA = ICD_CCA[keep_cols]
ICD_PCA.columns = [c+'_PCA' for c in keep_cols]
ICD_PLS.columns = [c+'_PLS' for c in keep_cols]
ICD_CCA.columns = [c+'_CCA' for c in keep_cols]

# Now we merge everything
ICD_DDR = pd.merge(ICD_CCA, ICD_PLS, left_index = True, right_index=True)
ICD_DDR = pd.merge(ICD_PCA, ICD_DDR, left_index = True, right_index=True)
# ICD_DDR.to_csv(os.path.join('models/ICD_Main_PCA', 'ICD_DDR_all.csv'))

# Now we extract people of interest
ICD_DDR_small = ICD_DDR.loc[brain_eid.values]
# ICD_DDR_small.to_csv(os.path.join('models/ICD_Main_PCA', 'ICD_DDR_all_brain_cohort.csv'))
############################################################################################################


print('Extract top PLS contributors')
# Find top contributions for each PCA component
topn = 3
pls_contrib = {}
for cat in tqdm(cat_codes):
        n_pc = pls_icds[cat].shape[1]
        n_cols = pls_icds[cat].shape[0]
        for ii in range(n_pc):
            col = cat+f'-{ii+1}' 
            cpca = pls_icds[cat][col]
            t = np.argpartition(np.abs(cpca),-topn).values[-topn:]
            t = np.bincount(t, minlength = n_cols).astype(bool)
            c = pls_icds[cat].index[t].values
            cc = [phecodes_idx.loc[float(cc)]['description'] for cc in c]
            pls_contrib[col] = [*cc, *c, *cpca[t]]

pls_contrib = pd.DataFrame(pls_contrib, 
                           index=[*[f'col {i+1}' for i in range(topn)], 
                                  *[f'raw col {i+1}' for i in range(topn)], 
                                  *[f'contrib {i+1}' for i in range(topn)]]).T

print('Extract top CCA contributors')
topn = 3
cca_contrib = {}
for cat in tqdm(cat_codes):
        n_pc = cca_icds[cat].shape[1]
        n_cols = cca_icds[cat].shape[0]
        for ii in range(n_pc):
            col = cat+f'-{ii+1}' 
            cpca = cca_icds[cat][col]
            t = np.argpartition(np.abs(cpca),-topn).values[-topn:]
            t = np.bincount(t, minlength = n_cols).astype(bool)
            c = cca_icds[cat].index[t].values
            cc = [phecodes_idx.loc[float(cc)]['description'] for cc in c]
            cca_contrib[col] = [*cc, *c, *cpca[t]]

cca_contrib = pd.DataFrame(cca_contrib, 
                           index=[*[f'col {i+1}' for i in range(topn)], 
                                  *[f'raw col {i+1}' for i in range(topn)], 
                                  *[f'contrib {i+1}' for i in range(topn)]]).T


############################################################################################################
# Push photos showing top contributions of each group
out_fold = 'reports/ICD_Work/Domain_Reduced'
topn=6

from pathlib import Path
for col in tqdm(cross_cols):
    cat = col.split('-')[0]


    filename=os.path.join(out_fold, f'CCA2', cat)
    Path(filename).mkdir(parents=True, exist_ok=True)

    small = cca_icds[cat][col].abs().nlargest(topn).to_frame()
    small[col] = cca_icds[cat][col].loc[small.index]
    small['color'] = phecodes_idx.loc[small.index.values.astype(float)]['group'].values
    small['color'] = small['color'].map(colorsdict)
    desc = phecodes_idx.loc[small.index.values.astype(float)]['description'].values
    desc = ['\n'.join(textwrap.wrap(d, width=max_chars, break_on_hyphens=False)) for d in desc]
    small['desc'] = desc

    plt.figure()
    plot = sns.barplot(data=small.reset_index(), x='desc', y=col, color=small['color'].iloc[0], errorbar=None, edgecolor='k')
    plt.xlabel('')
    plt.title(f"Top {topn} Absolute Largest {col} Loadings (CCA fMRI Corr {pls_fcor.loc[col]['r']:.3f})")
    plt.ylabel('Loading')
    for item in plot.get_xticklabels():
        item.set_rotation(45)
        item.set_horizontalalignment('right')
    plt.axhline(0, color='k')
    plt.legend(loc='upper left')
    plt.tight_layout()


    plt.savefig(os.path.join(filename, f'{col}_top{topn}.pdf'))
    plt.close()    