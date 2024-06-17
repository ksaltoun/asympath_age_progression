
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, scoreatpercentile

import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import SpectralClustering

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)

import processing.manhattan_plot_util as man_plot


'''
Determine retirement status of all indiviudals
    - Employed at both time points
    - Retired at both time points
    - Employed (T1) -> Retired (T2)
    - Other (none of above)
Examine asymmetry change in different retirement groups
'''

print('Loading Phenome')
ukbb_y, y_desc_dict, y_cat_dict = man_plot.load_phenom(BASE_FOLDER = 'data/processed/')

df_pheno_change = pd.read_csv('data/processed/ukbb_phenotype_change.csv')

# Classify people according to retirement status

df_retirement = pd.merge(df_pheno_change[['userID', '6142#2']].rename(columns={"6142#2": "Retire_Change"}), \
                ukbb_y[['userID', '6142#2', '6142#1']].rename(columns={"6142#2": "Retired", "6142#1": "Employed"}), \
                on='userID', suffixes=['_chg', '_org'])
df_retirement = df_retirement.set_index('userID').sort_index()
# Drop People who exited retirement
df_retirement['Other'] = df_retirement['Retire_Change'] == -1
df_retirement.loc[df_retirement['Retire_Change'] == -1, 'Retire_Change'] = 0

# For people who became retired
# Ignore their baseline status
df_retirement.loc[df_retirement['Retire_Change'] == 1, 'Retired'] = 0
df_retirement.loc[df_retirement['Retire_Change'] == 1, 'Employed'] = 0

# Add into the 'Other' Column people with alternative employment statuses
df_retirement['Other'] = df_retirement['Other'].astype(int).add(df_retirement[['Retire_Change', 'Retired', 'Employed']].sum(1) != 1)
df_retirement['Other'] = (df_retirement['Other'] > 0).astype(int)


# For people who in other category
# Ignore their baseline status
df_retirement.loc[df_retirement['Other'] == 1, 'Retired'] = 0
df_retirement.loc[df_retirement['Other'] == 1, 'Retire_Change'] = 0
df_retirement.loc[df_retirement['Other'] == 1, 'Employed'] = 0

assert np.all(df_retirement.sum(1) == 1)

df_rtr_status = df_retirement.idxmax(1)
df_rtr_status.name = 'Status'
###############################################################################################
# Incorporate the Change
###############################################################################################
retire_asym = df_pca_chg_rate.join(df_rtr_status).melt(id_vars='Status')
retire_ptrn = df_pca_chg_rate.join(df_retirement)
retire_ptrn = (pos_left_hemi_faster * df_pca_chg_rate[pos_left_hemi_faster.index]).join(df_retirement)
# retire_asym_amt = df_amt_chg_rate.join(df_rtr_status).melt(id_vars='Status')
# retire_asym['abs_change'] = retire_asym_amt['value']
retire_asym['abs_change'] = retire_asym['value'].abs()
# We can do the easy way since this is based on raw values


retire_ptrn['Status'] = df_rtr_status
df_pca_chg_meta

###############################################################################################
# Bar / point Chart with Standard error of mean
###############################################################################################

plt.figure()
sns.violinplot(data=retire_asym, hue='Status', x='variable', y='value')


# No sex Separation
rtr_grp = retire_asym.groupby([ 'variable', 'Status'])

# point Plot
fig, ax = plt.subplots(figsize = (5,5))
plt.axhline(y=0, color = 'k')
# plt.errorbar(x= rtr_grp.count().loc[:,:, ptrn][rtr_grp.count().loc[:,:,ptrn]>n_thres].dropna().index.levels[1],
plt.errorbar(x= rtr_grp.count().loc[ptrn][rtr_grp.count().loc[ptrn]>n_thres].dropna().index,
            y = rtr_grp.mean().loc[ptrn][rtr_grp.count().loc[ptrn]>n_thres].dropna()['value'],
            yerr = rtr_grp.std().loc[ptrn][rtr_grp.count().loc[ptrn]>n_thres].dropna()['value']\
                    /np.sqrt(rtr_grp.count().loc[ptrn][rtr_grp.count().loc[ptrn]>n_thres].dropna()['value']),
            fmt = 'o', mec = 'k', c = 'blue', label=ptrn)

plt.xlabel("Employment Status")
plt.ylabel(f"Change in Projection of {ptrn}")
plt.tight_layout()

# Bar Plot
fig, ax = plt.subplots(figsize = (5,5))
plt.axhline(y=0, color = 'k')
# plt.errorbar(x= rtr_grp.count().loc[:,:, ptrn][rtr_grp.count().loc[:,:,ptrn]>n_thres].dropna().index.levels[1],
plt.bar(x= rtr_grp.count()['abs_change'].loc[ptrn][rtr_grp.count()['abs_change'].loc[ptrn]>n_thres].dropna().index,
        height = rtr_grp.mean()['abs_change'].loc[ptrn][rtr_grp.count()['abs_change'].loc[ptrn]>n_thres].dropna()['abs_change'],
        yerr = rtr_grp.std()['abs_change'].loc[ptrn][rtr_grp.count()['abs_change'].loc[ptrn]>n_thres].dropna()['abs_change']\
                /np.sqrt(rtr_grp.count()['abs_change'].loc[ptrn][rtr_grp.count()['abs_change'].loc[ptrn]>n_thres].dropna()['abs_change']))

plt.xlabel("Employment Status")
plt.ylabel(f"Change in Projection of {ptrn}")
plt.tight_layout()


###############################################################################################
# Parallel Line Plot
###############################################################################################


from pandas.plotting import parallel_coordinates

# Take the iris dataset
import seaborn as sns
data = sns.load_dataset('iris')

# Make the plot
rtr_mean = rtr_grp.mean().reset_index().pivot(index='variable', columns='Status', values='value').drop(columns='Other')

plt.figure()
parallel_coordinates(rtr_mean.loc[labels[0:33]].reset_index(), 'variable', colormap=plt.get_cmap("viridis"))

plt.figure()
rtr_mean_sub = rtr_mean.sub(rtr_mean['Employed'], axis=0)
# rtr_mean_sub['Pos'] = np.sign(rtr_mean['Employed'])
parallel_coordinates(rtr_mean_sub.loc[labels[0:33]].reset_index(), 'variable', colormap=plt.get_cmap("Set2"))



fig, ax = plt.subplots(figsize = (9, 11))

rtr_mean_sub = rtr_mean.sub(rtr_mean['Employed'], axis=0)
rtr_mean_sub['Pos'] = np.sign(rtr_mean['Employed'])
rtr_mean_sub['Emp'] = rtr_mean['Employed']



import matplotlib
import matplotlib.colors as mplcolors

cmap = matplotlib.cm.get_cmap('coolwarm')
norm = mplcolors.CenteredNorm(halfrange=retire_colors.abs().max())

# Color by cohen's d?
# Color by largest cohens d
retire_colors = retired_d_abs.abs().max(1)* np.sign(rtr_mean['Employed'].loc[labels[0:n_show]])
# retire_colors = sns.color_palette("viridis", n_show)


fig, ax = plt.subplots(figsize = (5, 7))

# Adjust figure margins, this is going to be useful later.
fig.subplots_adjust(left=0.05, right=0.90, top=0.9, bottom=0.075)
# Iterate over ages, colors, and sizes, adding one line and pair of dots at a time
# Note the horizontal positions are fixed at 1 and 2.
rtr_mean_int = rtr_mean.loc[labels[0:n_show]]
for y0, y1, y2, c in zip(rtr_mean_int['Employed'], rtr_mean_int['Retire_Change'], \
                         rtr_mean_int['Retired'], cmap(norm(retire_colors))):
    ax.plot([1, 2, 3], [y0, y1, y2], c=c, lw=1)
    ax.scatter(1, y0, color = c, zorder=10, cmap='coolwarm')
    ax.scatter(2, y1, color = c, zorder=10, cmap='coolwarm')
    ax.scatter(3, y2, color = c,  zorder=10, cmap='coolwarm')
plt.xticks([1,2,3], ['Employed', 'Became\nRetired', 'Retired'])
plt.axhline(y=0, color='k', linestyle='-')
plt.ylabel('Mean Absolute Change')
plt.tight_layout()

plt.figure()
sns.heatmap(retire_colors.values.reshape(-1,1), vmax = retire_colors.abs().max(), vmin = -retire_colors.abs().max(),
            center = 0, cmap = 'coolwarm', linewidths=.01,
            cbar_kws=dict(label = "Cohen's d", pad = 0.02),
            yticklabels = retired_d.index, xticklabels ='',  square = True )
plt.tight_layout()
##############################################################################################
# Cohen's D
##############################################################################################

def pooled_std(series1, series2):
    std1 = (len(series1) - 1)*series1.std()**2
    std2 = (len(series2) - 1)*series2.std()**2
    pstd = np.sqrt((std1 + std2)/(len(series1) + len(series2) - 2))
    return pstd

def cohensd(series1, series2):
    pstd = pooled_std(series1, series2)
    d = (series1.mean() - series2.mean())/pstd
    return d

ptrn = 'Pattern 24'
cohensd(retire_ptrn[ptrn][retire_ptrn['Employed'] ==1],
        retire_ptrn[ptrn][retire_ptrn['Retired'] ==1])

retired_d = {}
for ptrn in labels[0:n_show]:
    retired_d[ptrn] = {
        "ER":cohensd(retire_ptrn[ptrn][retire_ptrn['Employed'] ==1],
                     retire_ptrn[ptrn][retire_ptrn['Retired'] ==1]),
        "EC":cohensd(retire_ptrn[ptrn][retire_ptrn['Employed'] ==1],
                     retire_ptrn[ptrn][retire_ptrn['Retire_Change'] ==1]),
        "CR":cohensd(retire_ptrn[ptrn][retire_ptrn['Retire_Change'] ==1],
                     retire_ptrn[ptrn][retire_ptrn['Retired'] ==1])
    }
retired_d = pd.DataFrame().from_dict(retired_d).T

retired_d_ttest = {}
for ptrn in labels[0:n_show]:
    retired_d_ttest[ptrn] = {
        "ER":ttest_ind(retire_ptrn[ptrn][retire_ptrn['Employed'] ==1],
                       retire_ptrn[ptrn][retire_ptrn['Retired'] ==1])[1],
        "EC":ttest_ind(retire_ptrn[ptrn][retire_ptrn['Employed'] ==1],
                       retire_ptrn[ptrn][retire_ptrn['Retire_Change'] ==1])[1],
        "CR":ttest_ind(retire_ptrn[ptrn][retire_ptrn['Retire_Change'] ==1],
                       retire_ptrn[ptrn][retire_ptrn['Retired'] ==1])[1]
    }
retired_d_ttest = pd.DataFrame().from_dict(retired_d_ttest).T


retired_d_abs = {}
for ptrn in labels[0:n_show]:
    retired_d_abs[ptrn] = {
        "ER":cohensd(retire_ptrn[ptrn][retire_ptrn['Employed'] ==1].abs(),
                     retire_ptrn[ptrn][retire_ptrn['Retired'] ==1].abs()),
        "EC":cohensd(retire_ptrn[ptrn][retire_ptrn['Employed'] ==1].abs(),
                     retire_ptrn[ptrn][retire_ptrn['Retire_Change'] ==1].abs()),
        "CR":cohensd(retire_ptrn[ptrn][retire_ptrn['Retire_Change'] ==1].abs(),
                     retire_ptrn[ptrn][retire_ptrn['Retired'] ==1].abs())
    }
retired_d_abs = pd.DataFrame().from_dict(retired_d_abs).T

SAVE_D_FOLDER = 'data/processed/Cohens_d_Longitudinal'
retired_d_abs.to_csv(os.path.join(SAVE_D_FOLDER, 'cohens_retirement_dir_MBAC.csv'))
retired_d.to_csv(os.path.join(SAVE_D_FOLDER, 'cohens_retirement_dir_LBAC.csv'))

retired_d_abs = pd.read_csv(os.path.join(SAVE_D_FOLDER, 'cohens_retirement_dir_MBAC.csv'), index_col = 0)
retired_d = pd.read_csv(os.path.join(SAVE_D_FOLDER, 'cohens_retirement_dir_LBAC.csv'), index_col = 0)

##############################################################################################
# Remove age effected
##############################################################################################
from nilearn.signal import clean
# X_clean_chg = clean((pos_left_hemi_faster * df_pca_chg_rate[pos_left_hemi_faster.index]).values, confounds=meta_df_idx.loc[df_pca_chg_rate.index].values, detrend=False, standardize=False)
# X_clean_amt = clean(df_amt_chg_rate.values, confounds=meta_df_idx.loc[df_pca_chg_rate.index].values, detrend=False, standardize=False)

X_clean_chg = clean((pos_left_hemi_faster * df_pca_chg_rate[pos_left_hemi_faster.index]).values, confounds=meta_df_idx['age'].loc[df_pca_chg_rate.index].values, detrend=False, standardize=False)
X_clean_amt = clean(df_amt_chg_rate.values, confounds=meta_df_idx['age'].loc[df_pca_chg_rate.index].values, detrend=False, standardize=False)

dfX_clean_chg = pd.DataFrame(X_clean_chg, index = df_pca_chg_rate.index, columns= pos_left_hemi_faster.index )
dfX_clean_amt = pd.DataFrame(X_clean_amt, index = df_amt_chg_rate.index, columns= df_amt_chg_rate.columns )

retire_ptrn_age_amt = (dfX_clean_amt).join(df_retirement).join(meta_df_idx['sex_F'])
retire_ptrn_age_chg = (dfX_clean_chg).join(df_retirement).join(meta_df_idx['sex_F'])

retired_d_age = {}
for ptrn in labels[0:n_show]:
    retired_d_age[ptrn] = {
        "ER":cohensd(retire_ptrn_age_chg[ptrn][retire_ptrn_age_chg['Employed'] ==1],
                     retire_ptrn_age_chg[ptrn][retire_ptrn_age_chg['Retired'] ==1]),
        "EC":cohensd(retire_ptrn_age_chg[ptrn][retire_ptrn_age_chg['Employed'] ==1],
                     retire_ptrn_age_chg[ptrn][retire_ptrn_age_chg['Retire_Change'] ==1]),
        "CR":cohensd(retire_ptrn_age_chg[ptrn][retire_ptrn_age_chg['Retire_Change'] ==1],
                     retire_ptrn_age_chg[ptrn][retire_ptrn_age_chg['Retired'] ==1]),
        "Sex":cohensd(retire_ptrn_age_chg[ptrn][retire_ptrn_age_chg['sex_F'] ==1],
                      retire_ptrn_age_chg[ptrn][retire_ptrn_age_chg['sex_F'] ==0])
    }
retired_d_age = pd.DataFrame().from_dict(retired_d_age).T

retired_d_age_abs = {}
for ptrn in labels[0:n_show]:
    retired_d_age_abs[ptrn] = {
        "ER":cohensd(retire_ptrn_age_amt[ptrn][retire_ptrn_age_amt['Employed'] ==1].abs(),
                     retire_ptrn_age_amt[ptrn][retire_ptrn_age_amt['Retired'] ==1].abs()),
        "EC":cohensd(retire_ptrn_age_amt[ptrn][retire_ptrn_age_amt['Employed'] ==1].abs(),
                     retire_ptrn_age_amt[ptrn][retire_ptrn_age_amt['Retire_Change'] ==1].abs()),
        "CR":cohensd(retire_ptrn_age_amt[ptrn][retire_ptrn_age_amt['Retire_Change'] ==1].abs(),
                     retire_ptrn_age_amt[ptrn][retire_ptrn_age_amt['Retired'] ==1].abs()),
        "Sex":cohensd(retire_ptrn_age_amt[ptrn][retire_ptrn_age_chg['sex_F'] ==1],
                      retire_ptrn_age_amt[ptrn][retire_ptrn_age_chg['sex_F'] ==0])
    }
retired_d_age_abs = pd.DataFrame().from_dict(retired_d_age_abs).T


SAVE_D_FOLDER = 'data/processed/Cohens_d_Longitudinal'
retired_d_age_abs.to_csv(os.path.join(SAVE_D_FOLDER, 'cohens_retirement_dir_age_deconf_GOOD_MBAC.csv'))
retired_d_age.to_csv(os.path.join(SAVE_D_FOLDER, 'cohens_retirement_dir_age_deconf_GOOD_LBAC.csv'))

retired_d_age_abs =pd.read_csv(os.path.join(SAVE_D_FOLDER, 'cohens_retirement_dir_age_deconf_GOOD_MBAC.csv'), index_col=0)
retired_d_age =pd.read_csv(os.path.join(SAVE_D_FOLDER, 'cohens_retirement_dir_age_deconf_GOOD_LBAC.csv'), index_col=0)

# Does including age make the direction change?
# Does the magnitude of the relationship change?
rtr_age_dir = (np.sign(retired_d_age) * np.sign(retired_d))>0 # We want this to be positive
##############################################################################################
# Visualization
# FIrst graph version:
# Last contrast encoded in ball size
##############################################################################################

sex_stats_pca = {}
for lk in tqdm(labels[0:n_show]):
        ttest_res = ttest_ind(df_pca_chg_meta.query('sex_F ==1')[lk], df_pca_chg_meta.query('sex_F ==0')[lk])
        # ttest_org = ttest_ind(df_org_ptn_meta.query('sex_F ==1')[lk], df_org_ptn_meta.query('sex_F ==0')[lk])
        cdres = cohensd(df_pca_chg_meta.query('sex_F ==1')[lk], df_pca_chg_meta.query('sex_F ==0')[lk])
        # cdres_org = cohensd(df_org_ptn_meta.query('sex_F ==1')[lk], df_org_ptn_meta.query('sex_F ==0')[lk])
        sex_stats_pca[lk] = {'cohens_d': cdres, 'ttest_T':ttest_res[0], 'ttest_p':ttest_res[1]}
                             #'ttest_T_og':ttest_org[0], 'ttest_p_og':ttest_org[1], 'cohens_d_org': cdres_org}
sex_stats_pca = pd.DataFrame().from_dict(sex_stats_pca).T

sex_stats_amt = {}
for lk in tqdm(labels[0:n_show]):
        ttest_res = ttest_ind(df_amt_chg_meta.query('sex_F ==1')[lk], df_amt_chg_meta.query('sex_F ==0')[lk])
        cdres = cohensd(df_amt_chg_meta.query('sex_F ==1')[lk], df_amt_chg_meta.query('sex_F ==0')[lk])
        sex_stats_amt[lk] = {'cohens_d': cdres, 'ttest_T':ttest_res[0], 'ttest_p':ttest_res[1]}
sex_stats_amt = pd.DataFrame().from_dict(sex_stats_amt).T

##############################################################################################
# Visualization
# FIrst graph version:
# Last contrast encoded in ball size
##############################################################################################

n_show = 33

vmax = np.abs(retired_d).max().max() *1.1

# extent = [vmax*-1, vmax, vmax*-1,vmax]
# arr = np.array([[1,0],[0,1]])

ptrn_cmap = sns.color_palette("viridis", n_show)

fig, ax = plt.subplots(1,1)

small_ball = 20; big_ball = 800
colors = sns.color_palette('bright', n_colors = 4)
ax = plt.subplot(111)

n_balls=4
points_list = np.asarray([[retired_d['ER'].mean(), retired_d['CR'].mean()] for _ in range(n_balls)]);

size_list = np.asarray([0, 0.05, 0.1, 0.15, 0.2, 0.25]);
size_list = np.asarray([0,  0.1, 0.2, 0.3]);
real_size_list = small_ball + big_ball * size_list
real_size_list = [int(r) for r in real_size_list]

temporaryPoints = []
for ii in range(n_balls):
    temporaryPoints.append(plt.scatter(points_list[ii,0], points_list[ii,1], marker='o', color='w', edgecolors='k',
                                       s = real_size_list[ii], label = f"{size_list[ii]:.2f}" ))
lgnd = plt.legend(loc = 'lower right', prop={'size': 8})
#for ii in range(4): lgnd.legendHandles[ii]._sizes = [int(big_ball/20)];
[tt.remove() for tt in temporaryPoints]


for i_p, ptrn in enumerate(labels[0:n_show]):
    ax.errorbar(x=retired_d.loc[ptrn]["ER"], y=retired_d.loc[ptrn]["EC"], fmt='o', capsize=0, \
            # yerr = diffMFerr[ptrn], \
            # xerr= (df_side_shift_SS.std()/np.sqrt(N_PPL))[ptrn],\
            label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p],\
            zorder=4, markeredgewidth=0 )#, markeredgecolor='k')
            # label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')
    ax.scatter(x=retired_d.loc[ptrn]["ER"], y=retired_d.loc[ptrn]["EC"], \
            # yerr = diffMFerr[ptrn], \
            # xerr= (df_side_shift_SS.std()/np.sqrt(N_PPL))[ptrn],\
            label=ptrn, color=ptrn_cmap[i_p], \
            zorder=2, alpha=0.4, s=small_ball+big_ball*retired_d.abs().loc[ptrn]["CR"])#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.grid(True)

plt.xlabel("Cohen's d (Employed vs Retired)")
plt.ylabel("Cohen's d (Employed vs Retiring)")

ax.set_ylim(-vmax, vmax)
ax.set_xlim(-vmax, vmax)
# ax.plot([0, 1], [0, 1], 'k--', transform=ax.transAxes)


n_rpl = 78
rpl_x2 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_rpl)
rpl_y2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n_rpl)

rpl_x = np.append(rpl_x2, np.zeros(n_rpl))
rpl_y = np.append(np.zeros(n_rpl), rpl_y2)

texts = [ax.text(x, y, str(n), zorder=5,
                        size = 'small', color='k', fontstretch='condensed')
        for n, x, y in zip(np.arange(1, n_show+1), \
            retired_d["ER"].loc[labels[0:n_show]], \
            retired_d["EC"].loc[labels[0:n_show]])]

adjust_text(texts, ax = ax, expand_text=(1.05, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))



big_sex_d = sex_stats_pca.abs()['cohens_d'].max()
rect1 = plt.Rectangle((-big_sex_d, -big_sex_d), 2*big_sex_d, 2*big_sex_d,
                      ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
ax.add_patch(rect1)

big_sex_d = age_partition_stats.abs()['cohens_d'].max()
rect1 = plt.Rectangle((-big_sex_d, -big_sex_d), 2*big_sex_d, 2*big_sex_d,
                      ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
ax.add_patch(rect1)

circle1 = plt.Circle((0, 0), sex_stats_pca.abs()['cohens_d'].max(),
                      ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
ax.add_patch(circle1)
circle2 = plt.Circle((0, 0), age_partition_stats.abs()['cohens_d'].max(),
                      ec='darkgrey', color='silver', zorder=1, alpha=0.4)
ax.add_patch(circle2)


# -------------------------------------------------------------------
# Graph without balls
# but instead we will have sex x one contrast
# and the other two contrasts
# -------------------------------------------------------------------

VIS_FOLDER = 'reports/Project2_Longitudional/Long_Fig_Drafts/Fig1_General_Change'
ptrn_cmap = sns.color_palette("viridis", n_show)

vmax = 1.1*max(np.abs(retired_d).max().max(),
               np.abs(retired_d_abs).max().max(),
               np.abs(sex_stats_pca['cohens_d']).max(),
               np.abs(sex_stats_amt['cohens_d']).max())

fig, ax = plt.subplots(1,1)
for i_p, ptrn in enumerate(labels[0:n_show]):
    ax.errorbar(x=retired_d.loc[ptrn]["EC"], y=retired_d.loc[ptrn]["CR"], fmt='o', capsize=0, \
            # yerr = diffMFerr[ptrn], \
            # xerr= (df_side_shift_SS.std()/np.sqrt(N_PPL))[ptrn],\
            label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p],\
            zorder=4, markeredgewidth=0 )#, markeredgecolor='k')
            # label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.grid(True)

plt.xlabel("Cohen's d (Employed vs Retiring)")
plt.ylabel("Cohen's d (Retired vs Retiring)")

ax.set_ylim(-vmax, vmax)
ax.set_xlim(-vmax, vmax)

n_rpl = 78
rpl_x2 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_rpl)
rpl_y2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n_rpl)

rpl_x = np.append(rpl_x2, np.zeros(n_rpl))
rpl_y = np.append(np.zeros(n_rpl), rpl_y2)

texts = [ax.text(x, y, str(n), zorder=5,
                        size = 'small', color='k', fontstretch='condensed')
        for n, x, y in zip(np.arange(1, n_show+1), \
            retired_d["EC"].loc[labels[0:n_show]], \
            retired_d["CR"].loc[labels[0:n_show]])]

adjust_text(texts, ax = ax, expand_text=(1.05, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))

big_sex_d = sex_stats_pca.abs()['cohens_d'].max()
rect1 = plt.Rectangle((-big_sex_d, -big_sex_d), 2*big_sex_d, 2*big_sex_d,
                      ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
ax.add_patch(rect1)
# circle2 = plt.Circle((0, 0), age_partition_stats.abs()['cohens_d'].max(),
#                       ec='darkgrey', color='silver', zorder=1, alpha=0.4)
# ax.add_patch(circle2)
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDER, 'retirement_age_deconf_GOOD_v_employed_dir.png'))
plt.savefig(os.path.join(VIS_FOLDER, 'retirement_age_deconf_GOOD_v_employed_dir.pdf'))
# -------------------------------------------------------------------
# Graph 2: Change vs retirement (one axis) and sex
fig, ax = plt.subplots(1,1)

for i_p, ptrn in enumerate(labels[0:n_show]):
    ax.errorbar(x=retired_d.loc[ptrn]["Sex"], y=retired_d.loc[ptrn]["ER"], fmt='o', capsize=0, \
    # ax.errorbar(x=sex_stats_pca.loc[ptrn]["cohens_d"], y=retired_d.loc[ptrn]["ER"], fmt='o', capsize=0, \
            # yerr = diffMFerr[ptrn], \
            # xerr= (df_side_shift_SS.std()/np.sqrt(N_PPL))[ptrn],\
            label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p],\
            zorder=4, markeredgewidth=0 )#, markeredgecolor='k')
            # label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.grid(True)

plt.xlabel("Cohen's d (Male vs Female)")
plt.ylabel("Cohen's d (Employed vs Retired)")

ax.set_ylim(-vmax, vmax)
ax.set_xlim(-vmax, vmax)

n_rpl = 78
rpl_x2 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_rpl)
rpl_y2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n_rpl)

rpl_x = np.append(rpl_x2, np.zeros(n_rpl))
rpl_y = np.append(np.zeros(n_rpl), rpl_y2)

texts = [ax.text(x, y, str(n), zorder=5,
                        size = 'small', color='k', fontstretch='condensed')
        for n, x, y in zip(np.arange(1, n_show+1), \
            retired_d["Sex"].loc[labels[0:n_show]], \
            # sex_stats_pca["cohens_d"].loc[labels[0:n_show]], \
            retired_d["ER"].loc[labels[0:n_show]])]

adjust_text(texts, ax = ax, expand_text=(1.05, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))

# circle1 = plt.Circle((0, 0), sex_stats.abs()['cohens_d'].max(),
#                       ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
# ax.add_patch(circle1)

big_sex_d = retired_d["Sex"].abs().max()
# big_sex_d = sex_stats_pca.abs()['cohens_d'].max()
rect1 = plt.Rectangle((-big_sex_d, -big_sex_d), 2*big_sex_d, 2*big_sex_d,
                      ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
ax.add_patch(rect1)

plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDER, 'retirement_age_deconf_GOOD_v_sex_dir.pdf'))
plt.savefig(os.path.join(VIS_FOLDER, 'retirement_age_deconf_GOOD_v_sex_dir.png'))

circle2 = plt.Circle((0, 0), age_partition_stats.abs()['cohens_d'].max(),
                      ec='darkgrey', color='silver', zorder=1, alpha=0.4)
ax.add_patch(circle2)


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# MBAC version
# -------------------------------------------------------------------
# -------------------------------------------------------------------

fig, ax = plt.subplots(1,1)
for i_p, ptrn in enumerate(labels[0:n_show]):
    ax.errorbar(x=retired_d_abs.loc[ptrn]["EC"], y=retired_d_abs.loc[ptrn]["CR"], fmt='o', capsize=0, \
            # yerr = diffMFerr[ptrn], \
            # xerr= (df_side_shift_SS.std()/np.sqrt(N_PPL))[ptrn],\
            label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p],\
            zorder=4, markeredgewidth=0 )#, markeredgecolor='k')
            # label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.grid(True)

plt.xlabel("Cohen's d (Employed vs Retiring)")
plt.ylabel("Cohen's d (Retired vs Retiring)")

ax.set_ylim(-vmax, vmax)
ax.set_xlim(-vmax, vmax)

n_rpl = 78
rpl_x2 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_rpl)
rpl_y2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n_rpl)

rpl_x = np.append(rpl_x2, np.zeros(n_rpl))
rpl_y = np.append(np.zeros(n_rpl), rpl_y2)

texts = [ax.text(x, y, str(n), zorder=5,
                        size = 'small', color='k', fontstretch='condensed')
        for n, x, y in zip(np.arange(1, n_show+1), \
            retired_d_abs["EC"].loc[labels[0:n_show]], \
            retired_d_abs["CR"].loc[labels[0:n_show]])]

adjust_text(texts, ax = ax, expand_text=(1.05, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))

big_sex_d = sex_stats_amt.abs()['cohens_d'].max()
rect1 = plt.Rectangle((-big_sex_d, -big_sex_d), 2*big_sex_d, 2*big_sex_d,
                      ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
ax.add_patch(rect1)
# circle2 = plt.Circle((0, 0), age_partition_stats.abs()['cohens_d'].max(),
#                       ec='darkgrey', color='silver', zorder=1, alpha=0.4)
# ax.add_patch(circle2)
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDER, 'retirement_age_deconf_GOOD_v_employed_MBAC.pdf'))
plt.savefig(os.path.join(VIS_FOLDER, 'retirement_age_deconf_GOOD_v_employed_MBAC.png'))
# -------------------------------------------------------------------
# Graph 2: Change vs retirement (one axis) and sex
fig, ax = plt.subplots(1,1)

for i_p, ptrn in enumerate(labels[0:n_show]):
    ax.errorbar(x=retired_d_abs.loc[ptrn]["Sex"], y=retired_d_abs.loc[ptrn]["ER"], fmt='o', capsize=0, \
    # ax.errorbar(x=sex_stats_amt.loc[ptrn]["cohens_d"], y=retired_d_abs.loc[ptrn]["ER"], fmt='o', capsize=0, \
            # yerr = diffMFerr[ptrn], \
            # xerr= (df_side_shift_SS.std()/np.sqrt(N_PPL))[ptrn],\
            label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p],\
            zorder=4, markeredgewidth=0 )#, markeredgecolor='k')
            # label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.grid(True)

plt.xlabel("Cohen's d (Male vs Female)")
plt.ylabel("Cohen's d (Employed vs Retired)")

ax.set_ylim(-vmax, vmax)
ax.set_xlim(-vmax, vmax)

n_rpl = 78
rpl_x2 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_rpl)
rpl_y2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n_rpl)

rpl_x = np.append(rpl_x2, np.zeros(n_rpl))
rpl_y = np.append(np.zeros(n_rpl), rpl_y2)

texts = [ax.text(x, y, str(n), zorder=5,
                        size = 'small', color='k', fontstretch='condensed')
        for n, x, y in zip(np.arange(1, n_show+1), \
            retired_d_abs["Sex"].loc[labels[0:n_show]], \
            # sex_stats_amt["cohens_d"].loc[labels[0:n_show]], \
            retired_d_abs["ER"].loc[labels[0:n_show]])]

adjust_text(texts, ax = ax, expand_text=(1.05, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))

# circle1 = plt.Circle((0, 0), sex_stats.abs()['cohens_d'].max(),
#                       ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
# ax.add_patch(circle1)

big_sex_d = retired_d_abs["Sex"].abs().max()
# big_sex_d = sex_stats_amt.abs()['cohens_d'].max()
rect1 = plt.Rectangle((-big_sex_d, -big_sex_d), 2*big_sex_d, 2*big_sex_d,
                      ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
ax.add_patch(rect1)

plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDER, 'retirement_age_deconf_GOOD_v_sex_MBAC.pdf'))
plt.savefig(os.path.join(VIS_FOLDER, 'retirement_age_deconf_GOOD_v_sex_MBAC.png'))

circle2 = plt.Circle((0, 0), age_partition_stats.abs()['cohens_d'].max(),
                      ec='darkgrey', color='silver', zorder=1, alpha=0.4)
ax.add_patch(circle2)