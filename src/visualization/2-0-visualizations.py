
##############################################################################################
# Load Data
##############################################################################################
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)

SAVE_FOLDER = 'models/Asymm_Patterns/Changes/2023_02_post_deconf'

# These will be used for comparison much later
df_pca_change = pd.read_csv(os.path.join(SAVE_FOLDER, 'Original', 'Asymm_Change.csv'), index_col=0)
df_amt_change = pd.read_csv(os.path.join(SAVE_FOLDER, 'Original', 'Amount_Change.csv'), index_col=0)
df_pca_chg_rate = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Asymm_Change.csv'), index_col=0)
df_amt_chg_rate = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Amount_Change.csv'), index_col=0)
pattern_change_rel = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Change_rel_deltaTBV.csv'), index_col=0)
pos_left_hemi_faster = np.sign(pattern_change_rel['r'])
fullcomps_1vis_df = pd.read_csv('data/Project_1/Processed/df_Pattern_Feature_Contrib_All.csv', index_col = 0)


SAVE_FOLDER = 'models/Asymm_Patterns/Permutations/2023_03_06_Permutations'
dfX_perm_all.to_csv(os.path.join(SAVE_FOLDER, "Perm_Asymm_Rate.csv"))
dfX_perm_abs_all.to_csv(os.path.join(SAVE_FOLDER, "Perm_Amount_Rate.csv"))
##############################################################################################
# Tree Map (Squares)
##############################################################################################
import squarify
from adjustText import adjust_text

labels = df_pca_rate_SS.columns.values
ptrn_cmap = sns.color_palette("viridis", n_show)

ptrn_cmap_dict = {ptrn: ptrn_cmap[i] for i, ptrn in enumerate(labels[0:n_show])}

# Turned Sideways to have more room
top_sq = 33
plt.figure()
squarify.plot(sizes=df_pca_chg_rate[labels[0:n_show]].mean().abs().nlargest(top_sq),
        pad = 0.1,
        label=df_pca_chg_rate[labels[0:n_show]].mean().abs().nlargest(top_sq).index.map(lambda x:x.split(' ')[1]),
        color=[ptrn_cmap_dict[i] for i in df_pca_chg_rate[labels[0:n_show]].mean().abs().nlargest(top_sq).index],
         alpha=.8, norm_x = 100, norm_y=50, text_kwargs={'rotation': 90})
plt.axis('off')
# plt.show()

# Bar Plot of asymmetry change (ordered by absolute value)
plt.figure()
df_pca_chg_rate[labels[0:n_show]].mean().abs().nlargest(n_show).plot(kind='bar')
plt.ylabel("Absolute Mean Asymmetry Change [a.u.; yearly]")
plt.tight_layout()
plt.show()

VIS_FOLDER = 'reports/Project2_Longitudional/Long_Fig_Drafts/Fig1_General_Change'

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


##############################################################################################
# Amount of Change vs Change
# Original Data
##############################################################################################

# Determine how many overlap with zero
bound_one = df_pca_chg_rate.mean() + (df_pca_chg_rate.std()/np.sqrt(N_PPL))
bound_two = df_pca_chg_rate.mean() - (df_pca_chg_rate.std()/np.sqrt(N_PPL))

pos_left_hemi_faster = np.sign(pattern_change_rel['r'])
n_show = 33
from adjustText import adjust_text

ptrn_cmap = sns.color_palette("viridis", n_show)
labels = df_pca_change.columns.values
ptrn_cmap_dict = {ptrn: ptrn_cmap[i] for i, ptrn in enumerate(labels[0:n_show])}


fig, ax = plt.subplots(1,1)
pts= []
for i_p, ptrn in enumerate(labels[0:n_show]):
#     ax.errorbar(x=real_cng[ptrn], y=diffMF[ptrn], fmt='none', capsize=3, yerr = diffMFerr[ptrn], \
#             xerr= ((chg_org.std()/np.sqrt(N_PPL) - perm_means_df.mean()) / perm_stdev_df.mean())[ptrn],\
#             # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
#             label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')
    pt = ax.errorbar(x=(pos_left_hemi_faster*df_pca_chg_rate)[ptrn].mean(), \
                y=df_amt_chg_rate.mean()[ptrn], \
                fmt='none', capsize=3, \
                yerr = (df_amt_chg_rate.std()/np.sqrt(N_PPL))[ptrn], \
                xerr= (df_pca_chg_rate.std()/np.sqrt(N_PPL))[ptrn],\
                 # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
                label=ptrn, ecolor=ptrn_cmap_dict[ptrn] )#, markeredgecolor='k')
    pts.append(pt)

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.grid(True)

# reset to be symmetric and color coordinates
QUADRANT_COLOR = True
if QUADRANT_COLOR:

    xlim = np.abs(ax.get_xlim()).max()
    ax.set_xlim(-xlim, xlim)
    ylim = np.abs(ax.get_ylim()).max()
    ax.set_ylim(0, ylim)

    ax.axhspan(ymin=0, ymax=ylim, xmin=0.5,  xmax=1, alpha=0.7, facecolor='lightgrey' ) # Q2
    # ax.axhspan(ymin=-ylim, ymax=0, xmin=0,  xmax=0.5, alpha=0.7, facecolor='lightgrey' ) # Q2

n_rpl = 78
rpl_x2 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_rpl)
rpl_y2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n_rpl)

rpl_x = np.append(rpl_x2, np.zeros(n_rpl))
rpl_y = np.append(np.zeros(n_rpl), rpl_y2)

texts = [ax.text(x, y, str(n),
                        size = 'small', color='k', fontstretch='condensed')
        for n, x, y in zip(np.arange(1, n_show+1), (pos_left_hemi_faster*df_pca_chg_rate).mean()[labels[0:n_show]], df_amt_chg_rate.mean()[labels[0:n_show]])]
print("Adjusting Text")
adjust_text(texts, ax = ax, expand_text=(1.05, 1.25),
            arrowprops=dict(arrowstyle='->', color='slategrey', relpos=(0.5, 1)))


# [plt.annotate(n, (x,y)) for n, x, y in zip(np.arange(1, n_show+1), df_side_shift_SS.mean()[labels[0:n_show]], diffMF[labels[0:n_show]])]
ax.set_xlabel('Mean Asymmetry Change [Arb. units; yearly rate]')
ax.set_ylabel('Mean Amount of Asymmetry Change  [Arb. units; yearly rate]')
plt.tight_layout()

plt.savefig(os.path.join(VIS_FOLDER, "Change_vs_Amount_Change_Orig_sign.pdf"))

##############################################################################################
# Male vs Female Changes
# Original Units
##############################################################################################


meta_df = pd.read_csv('data/Project_1/Processed/df_meta.csv', index_col=0)
meta_df_idx = meta_df.set_index('eid').sort_index()

df_side_shift_meta = df_pca_chg_rate.join(meta_df_idx[['age_v2', 'sex_F', 'IQ', 'R-Hand', 'L-Hand']])
df_side_shift_meta = df_side_shift_meta.join(df_pca_chg_rate, lsuffix='_chg', rsuffix='_abs')

df_side_shift_meta = (pos_left_hemi_faster*df_pca_chg_rate[pos_left_hemi_faster.index]).join(meta_df_idx[['age_v2', 'sex_F', 'IQ']])

n_show = 33

fml_asym = df_pca_chg_rate[(df_side_shift_meta['sex_F'] == 1).values].melt().dropna()
fml_asym['abs_change'] = df_amt_chg_rate[(df_side_shift_meta['sex_F'] == 1).values].melt().dropna()['value']
fml_asym['Side'] = f'Female'
mal_asym = df_pca_chg_rate[(df_side_shift_meta['sex_F'] == 0).values].melt().dropna()
mal_asym['abs_change'] = df_amt_chg_rate[(df_side_shift_meta['sex_F'] == 0).values].melt().dropna()['value']
mal_asym['Side'] = f'Male'

males_females = mal_asym.groupby('variable')['value'].mean().to_frame().rename(columns={'value':'Mean_Male'})
males_females['Mean_Female'] = fml_asym.groupby('variable')['value'].mean()
males_females['SE_Female'] = fml_asym.groupby('variable')['value'].std().abs() / np.sqrt(fml_asym.groupby('variable')['value'].count())
males_females['SE_Male'] = mal_asym.groupby('variable')['value'].std().abs() / np.sqrt(mal_asym.groupby('variable')['value'].count())

males_females['Abs_Male'] = mal_asym.groupby('variable')['abs_change'].mean()
males_females['Abs_Female'] = fml_asym.groupby('variable')['abs_change'].mean()
males_females['SE_Abs_Male'] = mal_asym.groupby('variable')['abs_change'].std().abs() / np.sqrt(mal_asym.groupby('variable')['abs_change'].count())
males_females['SE_Abs_Female'] = fml_asym.groupby('variable')['abs_change'].std().abs() / np.sqrt(fml_asym.groupby('variable')['abs_change'].count())

ptrn_cmap = sns.color_palette("viridis", n_show)

ptrn_cmap_dict = {ptrn: ptrn_cmap[i] for i, ptrn in enumerate(labels[0:n_show])}

fig, ax = plt.subplots(1,1)
for i_p, ptrn in enumerate(labels[0:n_show]):
#     ax.errorbar(x=real_cng[ptrn], y=diffMF[ptrn], fmt='none', capsize=3, yerr = diffMFerr[ptrn], \
#             xerr= ((chg_org.std()/np.sqrt(N_PPL) - perm_means_df.mean()) / perm_stdev_df.mean())[ptrn],\
#             # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
#             label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')
    ax.errorbar(x=males_females['Mean_Female'].loc[ptrn], \
                y=males_females['Mean_Male'].loc[ptrn], \
                xerr = males_females['SE_Female'].loc[ptrn], \
                yerr= males_females['SE_Male'].loc[ptrn],\
                fmt='none', capsize=3, \
            # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
                label=ptrn, ecolor=ptrn_cmap_dict[ptrn] )#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.axline((0, 0), slope=1, c='xkcd:slate grey', ls='--')
ax.grid(True)

# reset to be symmetric and color coordinates
QUADRANT_COLOR = True
if QUADRANT_COLOR:

    xlim = np.abs(ax.get_xlim()).max()
    ax.set_xlim(-xlim, xlim)
    ylim = np.abs(ax.get_ylim()).max()
    ax.set_ylim(-ylim, ylim)

    ax.axhspan(ymin=0, ymax=ylim, xmin=0.5,  xmax=1, alpha=0.7, facecolor='lightgrey' ) # Q2
    ax.axhspan(ymin=-ylim,  ymax=0, xmin=0,  xmax=0.5,  alpha=0.7, facecolor='lightgrey' ) #Q4

n_rpl = 78
rpl_x2 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_rpl)
rpl_y2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n_rpl)

rpl_x = np.append(rpl_x2, np.zeros(n_rpl))
rpl_y = np.append(np.zeros(n_rpl), rpl_y2)

rpl_x_err = []
rpl_y_err = []

rpl_y_err.extend((males_females['Mean_Male']-males_females['SE_Male']).values)
rpl_x_err.extend((males_females['Mean_Female']).values)
rpl_y_err.extend((males_females['Mean_Male']+males_females['SE_Male']).values)
rpl_x_err.extend((males_females['Mean_Female']).values)
rpl_y_err.extend((males_females['Mean_Male']).values)
rpl_x_err.extend((males_females['Mean_Female']-males_females['SE_Female']).values)
rpl_y_err.extend((males_females['Mean_Male']).values)
rpl_x_err.extend((males_females['Mean_Female']+males_females['SE_Female']).values)


rpl_x = np.append(rpl_x, rpl_x_err)
rpl_y = np.append(rpl_y, rpl_y_err)

# texts = [ax.text(x, y, str(n),
#                         size = 'small', color='k', fontstretch='condensed')
#         for n, x, y in zip(np.arange(1, n_show+1), males_females['Mean_Female'], males_females['Mean_Male'])]

texts = [ax.text(males_females.loc[r]['Mean_Female'],  males_females.loc[r]['Mean_Male'],
                r.split(' ')[1],
                size = 'small', color='k', fontstretch='condensed')
        for r in labels[0:n_show]]

print("Adjusting Text")
adjust_text(texts, rpl_x, rpl_y , ax = ax, expand_text=(1.3, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))


# [plt.annotate(n, (x,y)) for n, x, y in zip(np.arange(1, n_show+1), df_side_shift_SS.mean()[labels[0:n_show]], diffMF[labels[0:n_show]])]
ax.set_xlabel('Mean Asymmetry Change in Females [Arb. units; yearly rate]')
ax.set_ylabel('Mean Asymmetry Change in Males [Arb. units; yearly rate]')
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDER, "Change_M_v_F_Orig.pdf"))



# Mean Absolute Change

fig, ax = plt.subplots(1,1)
for i_p, ptrn in enumerate(labels[0:n_show]):
#     ax.errorbar(x=real_cng[ptrn], y=diffMF[ptrn], fmt='none', capsize=3, yerr = diffMFerr[ptrn], \
#             xerr= ((chg_org.std()/np.sqrt(N_PPL) - perm_means_df.mean()) / perm_stdev_df.mean())[ptrn],\
#             # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
#             label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')
    ax.errorbar(x=males_females['Abs_Female'].loc[ptrn], \
                y=males_females['Abs_Male'].loc[ptrn], \
                xerr = males_females['SE_Abs_Female'].loc[ptrn], \
                yerr= males_females['SE_Abs_Male'].loc[ptrn],\
                fmt='none', capsize=3, \
            # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
                label=ptrn, ecolor=ptrn_cmap_dict[ptrn] )#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.axline((0, 0), slope=1, c='xkcd:slate grey', ls='--')
ax.grid(True)

ax.set_ylim(0.1, 0.35)
ax.set_xlim(0.1, 0.35)


rpl_x_err = []
rpl_y_err = []

rpl_y_err.extend((males_females['Abs_Male']-males_females['SE_Abs_Male']).values)
rpl_x_err.extend((males_females['Abs_Female']).values)
rpl_y_err.extend((males_females['Abs_Male']+males_females['SE_Abs_Male']).values)
rpl_x_err.extend((males_females['Abs_Female']).values)
rpl_y_err.extend((males_females['Abs_Male']).values)
rpl_x_err.extend((males_females['Abs_Female']-males_females['SE_Abs_Female']).values)
rpl_y_err.extend((males_females['Abs_Male']).values)
rpl_x_err.extend((males_females['Abs_Female']+males_females['SE_Abs_Female']).values)

# texts = [ax.text(x, y, str(n),
#                         size = 'small', color='k', fontstretch='condensed')
#         for n, x, y in zip(np.arange(1, n_show+1), old_young['Mean_Female'], old_young['Mean_Male'])]

texts = [ax.text(males_females.loc[r]['Abs_Female'],  males_females.loc[r]['Abs_Male'],
                r.split(' ')[1],
                size = 'small', color='k', fontstretch='condensed')
        for r in labels[0:n_show]]

print("Adjusting Text")
adjust_text(texts, rpl_x_err, rpl_y_err , ax = ax, expand_text=(1.3, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))


# [plt.annotate(n, (x,y)) for n, x, y in zip(np.arange(1, n_show+1), df_side_shift_SS.mean()[labels[0:n_show]], diffMF[labels[0:n_show]])]
ax.set_ylabel(f'Mean Amount of Asymmetry Change in Males [Arb. units; yearly rate]')
ax.set_xlabel(f'Mean Amount of Asymmetry Change in Females [Arb. units; yearly rate]')
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDER, "Abs_Change_M_v_F_Orig_Zoom.pdf"))

from scipy.stats import ttest_ind


df_pca_chg_meta = (pos_left_hemi_faster*df_pca_chg_rate).join(meta_df_idx[['age_v2', 'sex_F', 'IQ']])
df_amt_chg_meta = df_amt_chg_rate.join(meta_df_idx[['age_v2', 'sex_F', 'IQ']])
df_org_ptn_meta = dfX_new_v1.join(meta_df_idx[['age_v2', 'sex_F', 'IQ']])



# Sex Descriptive Stats
# Cohen's D; ttest_ind
from scipy.stats import ttest_ind

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

SAVE_D_FOLDER = 'data/processed/Cohens_d_Longitudinal'
sex_stats_amt.to_csv(os.path.join(SAVE_D_FOLDER, 'cohens_sex_MBAC.csv'))
sex_stats_pca.to_csv(os.path.join(SAVE_D_FOLDER, 'cohens_sex_dir_corr_LBAC.csv'))

sex_stats_amt = pd.read_csv(os.path.join(SAVE_D_FOLDER, 'cohens_sex_MBAC.csv'), index_col = 0)
sex_stats_pca = pd.read_csv(os.path.join(SAVE_D_FOLDER, 'cohens_sex_dir_corr_LBAC.csv'), index_col = 0)


##############################################################################################
# Old vs Young Changes
# Original Values
##############################################################################################

th = 25 #percentile of interest: we take 5% and 95%
lower_age, upper_age = np.percentile(df_side_shift_meta['age_v2'], [th, 100-th], axis=0)

age_partition_stats_pca = {}
for lk in tqdm(labels[0:n_show]):
        ttest_res = ttest_ind(yng_asym.query(f"variable=='{lk}'")['abs_change'], old_asym.query(f"variable=='{lk}'")['abs_change'])
        cdres = cohensd(yng_asym.query(f"variable=='{lk}'")['abs_change'], old_asym.query(f"variable=='{lk}'")['abs_change'])
        age_partition_stats_pca[lk] = {'cohens_d': cdres, 'ttest_T':ttest_res[0], 'ttest_p':ttest_res[1]}
age_partition_stats_pca = pd.DataFrame().from_dict(age_partition_stats_pca).T

age_partition_stats_pca = {}
for lk in tqdm(labels[0:n_show]):
        ttest_res = ttest_ind(df_side_shift_meta.query(f"age_v2<{lower_age}")[lk],
                              df_side_shift_meta.query(f"age_v2>{upper_age}")[lk])
        cdres = cohensd(df_side_shift_meta.query(f"age_v2<{lower_age}")[lk],
                        df_side_shift_meta.query(f"age_v2>{upper_age}")[lk])
        age_partition_stats_pca[lk] = {'cohens_d': cdres, 'ttest_T':ttest_res[0], 'ttest_p':ttest_res[1]}
age_partition_stats_pca = pd.DataFrame().from_dict(age_partition_stats_pca).T

age_partition_stats_abs = {}
for lk in tqdm(labels[0:n_show]):
        ttest_res = ttest_ind(df_side_shift_meta.query(f"age_v2<{lower_age}")[lk].abs(),
                              df_side_shift_meta.query(f"age_v2>{upper_age}")[lk].abs())
        cdres = cohensd(df_side_shift_meta.query(f"age_v2<{lower_age}")[lk].abs(),
                        df_side_shift_meta.query(f"age_v2>{upper_age}")[lk].abs())
        age_partition_stats_abs[lk] = {'cohens_d': cdres, 'ttest_T':ttest_res[0], 'ttest_p':ttest_res[1]}
age_partition_stats_abs = pd.DataFrame().from_dict(age_partition_stats_abs).T

age_partition_stats_abs.to_csv(os.path.join(SAVE_D_FOLDER, 'cohens_age_MBAC.csv'))
age_partition_stats_pca.to_csv(os.path.join(SAVE_D_FOLDER, 'cohens_age_dir_corr_LBAC.csv'))

age_partition_stats_abs = pd.read_csv(os.path.join(SAVE_D_FOLDER, 'cohens_age_MBAC.csv'), index_col = 0)
age_partition_stats_pca = pd.read_csv(os.path.join(SAVE_D_FOLDER, 'cohens_age_dir_corr_LBAC.csv'), index_col = 0)

demo_cohens_d = pd.DataFrame()
demo_cohens_d['age_MBAC'] = age_partition_stats_abs['cohens_d']
demo_cohens_d['age_LBAC'] = age_partition_stats_pca['cohens_d']
demo_cohens_d['sex_LBAC'] = sex_stats_pca['cohens_d']
demo_cohens_d['sex_MBAC'] = sex_stats_amt['cohens_d']

df_pca_chg_meta = df_pca_chg_rate.join(meta_df_idx[['age_v2', 'sex_F', 'IQ']])
df_amt_chg_meta = df_amt_chg_rate.join(meta_df_idx[['age_v2', 'sex_F', 'IQ']])

age_partition_stats_pca = {}
for lk in tqdm(labels[0:n_show]):
        ttest_res = ttest_ind(df_pca_chg_meta.query(f'age_v2 >= {upper_age}')[lk], df_pca_chg_meta.query(f'age_v2 <= {lower_age}')[lk])
        cdres = cohensd(df_pca_chg_meta.query(f'age_v2 >= {upper_age}')[lk], df_pca_chg_meta.query(f'age_v2 <= {lower_age}')[lk])
        age_partition_stats_pca[lk] = {'cohens_d': cdres, 'ttest_T':ttest_res[0], 'ttest_p':ttest_res[1]}
age_partition_stats_pca = pd.DataFrame().from_dict(age_partition_stats_pca).T

age_partition_stats_amt = {}
for lk in tqdm(labels[0:n_show]):
        ttest_res = ttest_ind(df_amt_chg_meta.query(f'age_v2 >= {upper_age}')[lk], df_amt_chg_meta.query(f'age_v2 <= {lower_age}')[lk])
        cdres = cohensd(df_amt_chg_meta.query(f'age_v2 >= {upper_age}')[lk], df_amt_chg_meta.query(f'age_v2 <= {lower_age}')[lk])
        age_partition_stats_amt[lk] = {'cohens_d': cdres, 'ttest_T':ttest_res[0], 'ttest_p':ttest_res[1]}
age_partition_stats_amt = pd.DataFrame().from_dict(age_partition_stats_amt).T

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# MBAC version
# -------------------------------------------------------------------
# -------------------------------------------------------------------

vmax = 1.15*max(age_partition_stats_pca["cohens_d"].abs().max(),
                age_partition_stats_abs["cohens_d"].abs().max())
fig, ax = plt.subplots(1,1)
for i_p, ptrn in enumerate(labels[0:n_show]):
    ax.errorbar(x=age_partition_stats_pca.loc[ptrn]["cohens_d"], y=age_partition_stats_abs.loc[ptrn]["cohens_d"], fmt='o', capsize=0, \
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

plt.xlabel("Cohen's d (Younger vs Older LBAC)")
plt.ylabel("Cohen's d (Younger vs Older MBAC)")

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
            age_partition_stats_pca["cohens_d"].loc[labels[0:n_show]], \
            age_partition_stats_abs["cohens_d"].loc[labels[0:n_show]])]

adjust_text(texts, ax = ax, expand_text=(1.05, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))

big_sex_d = sex_stats_pca.abs()['cohens_d'].max()
big_sex_d_abs = sex_stats_amt.abs()['cohens_d'].max()
rect1 = plt.Rectangle((-big_sex_d, -big_sex_d_abs), 2*big_sex_d, 2*big_sex_d_abs,
                      ec='slategrey', color='lightgrey', zorder=1, alpha=0.4)
ax.add_patch(rect1)
# circle2 = plt.Circle((0, 0), age_partition_stats.abs()['cohens_d'].max(),
#                       ec='darkgrey', color='silver', zorder=1, alpha=0.4)
# ax.add_patch(circle2)
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDER, 'age_MBAC_v_age_LBAC.pdf'))
plt.savefig(os.path.join(VIS_FOLDER, 'age_MBAC_v_age_LBAC.png'))

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Graphs
# -------------------------------------------------------------------
# -------------------------------------------------------------------

# We have the people who are big or small
# Use them to
yng_mask = df_side_shift_meta['age_v2'].le(lower_age)
old_mask = df_side_shift_meta['age_v2'].ge(upper_age)

old_asym = df_pca_chg_rate[old_mask].melt().dropna()
yng_asym = df_pca_chg_rate[yng_mask].melt().dropna()
old_asym['abs_change'] = df_amt_chg_rate[old_mask].melt().dropna()['value']
yng_asym['abs_change'] = df_amt_chg_rate[yng_mask].melt().dropna()['value']


old_asym.groupby('variable')['value'].agg([('negative' , lambda x : x[x < 0].mean()) , ('positive' , lambda x : x[x > 0].mean())])

old_young = old_asym.groupby('variable')['value'].mean()\
                        .to_frame().rename(columns={'value':'Mean_Old'})
old_young['Mean_Young'] = yng_asym.groupby('variable')['value'].mean()
old_young['SE_Young'] = yng_asym.groupby('variable')['value'].std().abs() / np.sqrt(yng_asym.groupby('variable')['value'].count())
old_young['SE_Old'] = old_asym.groupby('variable')['value'].std().abs() / np.sqrt(old_asym.groupby('variable')['value'].count())

old_young['Abs_Old'] = old_asym.groupby('variable')['abs_change'].mean()
old_young['Abs_Young'] = yng_asym.groupby('variable')['abs_change'].mean()
old_young['SE_Abs_Young'] = yng_asym.groupby('variable')['abs_change'].std().abs() / np.sqrt(yng_asym.groupby('variable')['abs_change'].count())
old_young['SE_Abs_Old'] = old_asym.groupby('variable')['abs_change'].std().abs() / np.sqrt(old_asym.groupby('variable')['abs_change'].count())


ptrn_cmap = sns.color_palette("viridis", n_show)

ptrn_cmap_dict = {ptrn: ptrn_cmap[i] for i, ptrn in enumerate(labels[0:n_show])}

# Mean vs Mean

fig, ax = plt.subplots(1,1)
for i_p, ptrn in enumerate(labels[0:n_show]):
#     ax.errorbar(x=real_cng[ptrn], y=diffMF[ptrn], fmt='none', capsize=3, yerr = diffMFerr[ptrn], \
#             xerr= ((chg_org.std()/np.sqrt(N_PPL) - perm_means_df.mean()) / perm_stdev_df.mean())[ptrn],\
#             # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
#             label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')
    ax.errorbar(x=old_young['Mean_Young'].loc[ptrn], \
                y=old_young['Mean_Old'].loc[ptrn], \
                xerr = old_young['SE_Young'].loc[ptrn], \
                yerr= old_young['SE_Old'].loc[ptrn],\
                fmt='none', capsize=3, \
            # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
                label=ptrn, ecolor=ptrn_cmap_dict[ptrn] )#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.axline((0, 0), slope=1, c='xkcd:slate grey', ls='--')
ax.grid(True)

# reset to be symmetric and color coordinates
QUADRANT_COLOR = True
if QUADRANT_COLOR:

    xlim = np.abs(ax.get_xlim()).max()
    ax.set_xlim(-xlim, xlim)
    ylim = np.abs(ax.get_ylim()).max()
    ax.set_ylim(-ylim, ylim)

    ax.axhspan(ymin=0, ymax=ylim, xmin=0.5,  xmax=1, alpha=0.7, facecolor='lightgrey' ) # Q2
    ax.axhspan(ymin=-ylim,  ymax=0, xmin=0,  xmax=0.5,  alpha=0.7, facecolor='lightgrey' ) #Q4

n_rpl = 78
rpl_x2 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_rpl)
rpl_y2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n_rpl)

rpl_x = np.append(rpl_x2, np.zeros(n_rpl))
rpl_y = np.append(np.zeros(n_rpl), rpl_y2)

rpl_x_err = []
rpl_y_err = []

rpl_y_err.extend((old_young['Mean_Old']-old_young['SE_Old']).values)
rpl_x_err.extend((old_young['Mean_Young']).values)
rpl_y_err.extend((old_young['Mean_Old']+old_young['SE_Old']).values)
rpl_x_err.extend((old_young['Mean_Young']).values)
rpl_y_err.extend((old_young['Mean_Old']).values)
rpl_x_err.extend((old_young['Mean_Young']-old_young['SE_Young']).values)
rpl_y_err.extend((old_young['Mean_Old']).values)
rpl_x_err.extend((old_young['Mean_Young']+old_young['SE_Young']).values)


rpl_x = np.append(rpl_x, rpl_x_err)
rpl_y = np.append(rpl_y, rpl_y_err)

# texts = [ax.text(x, y, str(n),
#                         size = 'small', color='k', fontstretch='condensed')
#         for n, x, y in zip(np.arange(1, n_show+1), old_young['Mean_Female'], old_young['Mean_Male'])]

texts = [ax.text(old_young.loc[r]['Mean_Young'],  old_young.loc[r]['Mean_Old'],
                r.split(' ')[1],
                size = 'small', color='k', fontstretch='condensed')
        for r in labels[0:n_show]]

print("Adjusting Text")
adjust_text(texts, rpl_x, rpl_y , ax = ax, expand_text=(1.3, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))


# [plt.annotate(n, (x,y)) for n, x, y in zip(np.arange(1, n_show+1), df_side_shift_SS.mean()[labels[0:n_show]], diffMF[labels[0:n_show]])]
ax.set_ylabel(f'Mean Asymmetry Change of Elderly (>{upper_age:.0f} years) [Arb. units; yearly rate]')
ax.set_xlabel(f'Mean Asymmetry Change of Young (<{lower_age:.0f} years) [Arb. units; yearly rate]')
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDER, "Change_Old_v_Young_Orig.pdf"))

# Mean Absolute Change

fig, ax = plt.subplots(1,1)
for i_p, ptrn in enumerate(labels[0:n_show]):
#     ax.errorbar(x=real_cng[ptrn], y=diffMF[ptrn], fmt='none', capsize=3, yerr = diffMFerr[ptrn], \
#             xerr= ((chg_org.std()/np.sqrt(N_PPL) - perm_means_df.mean()) / perm_stdev_df.mean())[ptrn],\
#             # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
#             label=ptrn, ecolor=ptrn_cmap[i_p] )#, markeredgecolor='k')
    ax.errorbar(x=old_young['Abs_Young'].loc[ptrn], \
                y=old_young['Abs_Old'].loc[ptrn], \
                xerr = old_young['SE_Abs_Young'].loc[ptrn], \
                yerr= old_young['SE_Abs_Old'].loc[ptrn],\
                fmt='none', capsize=3, \
            # label=ptrn, ecolor=ptrn_cmap[i_p], markerfacecolor=ptrn_cmap[i_p], markeredgewidth=0 )#, markeredgecolor='k')
                label=ptrn, ecolor=ptrn_cmap_dict[ptrn] )#, markeredgecolor='k')

# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.50),
#           ncol=5, fancybox=True, shadow=True, prop={'size': 8})
plt.axvline(0, c='k')
plt.axhline(0, c='k')
ax.axline((0, 0), slope=1, c='xkcd:slate grey', ls='--')
ax.grid(True)

ax.set_ylim(0.1, 0.35)
ax.set_xlim(0.1, 0.3)

rpl_x_err = []
rpl_y_err = []

rpl_y_err.extend((old_young['Abs_Old']-old_young['SE_Abs_Old']).values)
rpl_x_err.extend((old_young['Abs_Young']).values)
rpl_y_err.extend((old_young['Abs_Old']+old_young['SE_Abs_Old']).values)
rpl_x_err.extend((old_young['Abs_Young']).values)
rpl_y_err.extend((old_young['Abs_Old']).values)
rpl_x_err.extend((old_young['Abs_Young']-old_young['SE_Abs_Young']).values)
rpl_y_err.extend((old_young['Abs_Old']).values)
rpl_x_err.extend((old_young['Abs_Young']+old_young['SE_Abs_Young']).values)

# texts = [ax.text(x, y, str(n),
#                         size = 'small', color='k', fontstretch='condensed')
#         for n, x, y in zip(np.arange(1, n_show+1), old_young['Mean_Female'], old_young['Mean_Male'])]

texts = [ax.text(old_young.loc[r]['Abs_Young'],  old_young.loc[r]['Abs_Old'],
                r.split(' ')[1],
                size = 'small', color='k', fontstretch='condensed')
        for r in labels[0:n_show]]

print("Adjusting Text")
adjust_text(texts, rpl_x_err, rpl_y_err , ax = ax, expand_text=(1.3, 1.25),
                arrowprops=dict(arrowstyle='->', color='darkgrey', relpos=(0.5, 1)))


# [plt.annotate(n, (x,y)) for n, x, y in zip(np.arange(1, n_show+1), df_side_shift_SS.mean()[labels[0:n_show]], diffMF[labels[0:n_show]])]
ax.set_ylabel(f'Mean Absolute Change in Asymmetry of Elderly (>{upper_age:.0f} years) [Arb. units; yearly rate]')
ax.set_xlabel(f'Mean Absolute Change in Asymmetry of Young (<{lower_age:.0f} years) [Arb. units; yearly rate]')
plt.tight_layout()

plt.savefig(os.path.join(VIS_FOLDER, "Abs_Change_Old_v_Young_Orig_Zoom.pdf"))


##############################################################################################
# Corrrelation Strucutre
##############################################################################################


# EXPT : Are PCA changes uncorrelated?

import seaborn as sns

#First Glance

comps_df = (pos_left_hemi_faster*df_pca_chg_rate.iloc[:, 0:33]).copy()
comps_df = comps_df.join(df_amt_chg_rate.iloc[:, 0:33], lsuffix='_chg', rsuffix='_amt')
corrMat = comps_df.corr()

from sklearn.cluster import SpectralClustering
n_cluster = 4
clustering = SpectralClustering(n_clusters=n_cluster)
clustering.fit(corrMat)

new_idxs = np.concatenate([np.where(clustering.labels_ == ii )[0] for ii in range(clustering.n_clusters)])
corrMat2 = corrMat.iloc[new_idxs, new_idxs]


plt.figure(); sns.heatmap(np.triu(corrMat2), cmap = 'bwr', center = 0,
                          xticklabels=corrMat.columns.values[new_idxs], yticklabels=corrMat.index.values[new_idxs])


plt.figure(); sns.heatmap(np.triu(corrMat, k=1), cmap = 'bwr', center = 0,  vmax=1, vmin=-1,
                          xticklabels=corrMat.columns.values, yticklabels=corrMat.index.values)

plt.figure(); sns.heatmap(corrMat, cmap = 'bwr', center = 0, vmax=1, vmin=-1, square=True,
                          cbar_kws={"shrink": 0.5},
                          xticklabels=corrMat.columns.values, yticklabels=corrMat.index.values)
# plt.savefig(os.path.join(VIS_FOLDER, "Correlation_Change_and_Abs_change_dir_fix_good.pdf"))


corrMat_melt = corrMat.where(np.triu(np.ones(corrMat.shape),k=1).astype(bool))
corrMat_melt = corrMat_melt.stack().reset_index()
corrMat_melt.columns = ['Row','Column','Value']

corrMat_melt_ss = corrMat_melt[(corrMat_melt['Row'].str.contains('_chg'))]
corrMat_melt_ss[(corrMat_melt_ss['Column'].str.contains('_amt'))]

vals = [(i, corrMat.loc[f'Pattern {i}_amt'][f'Pattern {i}_chg']) for i in range(1, 34)]
vals = pd.DataFrame(vals, columns=['Pattern', 'Correlation']).set_index('Pattern').squeeze(axis=0)
##############################################################################################
# Unused
##############################################################################################


comps_df = pd.DataFrame(fullcomps_1vis, index = var_plot).T
corrMat = comps_df.corr()
plt.figure(); sns.heatmap(corrMat2.upper(), cmap = 'bwr', center = 0,
                          xticklabels=var_plot, yticklabels=var_plot)
plt.title("Correlation of Component x  Brain Features")



plt.figure(); sns.heatmap(corrMat, cmap = 'bwr', center = 0,
                          xticklabels=var_plot, yticklabels=var_plot)
plt.title("Correlation of Component x  Component Expression at Visit 1")

t = np.zeros((n_comp, n_comp))
for i1 in range(n_comp):
    for i2 in range(n_comp):
        r, _ = pearsonr(fullcomps_1vis[i1], fullcomps_1vis[i2])
        t[i1, i2] = r
#%%
#with Bootstrap
df_pca_change = (dfX_new_2vis_v2 - dfX_new_2vis_v1).copy()

n_bs = 100
n_ind = mask_2vis.sum()
n_comp = n_comp #reminder that we have this already

#Boot strap (i.e. repeat analysis but with mixed up data input) to recreate the model fit
# this creates some variance in the coefs we get, and shows how much is due to the input and how much is retained regardless of input

# perform bagging to build confidence in the regression coefficients
bs_coefs = []
r2_vals = []
for i_bs in tqdm(range(n_bs)):
    np.random.seed(i_bs**2)
    bs_sample_inds = np.random.randint(0, n_ind, n_ind) # we are now scrambling, and thus not taking the full set of 40k pts anymore

    comps_df = (dfX_new_2vis_v2.iloc[bs_sample_inds] - dfX_new_2vis_v1.iloc[bs_sample_inds]).copy().drop('eid', axis = 1)
    corrMat = comps_df.corr()
    bs_coefs.append(corrMat.values)
bs_coefs_array = np.array(bs_coefs)
# np.save(CURR_FOLDER + '/bs_coefs_array_200713', bs_coefs_array)
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

sig_mask  = np.zeros(lower_CI.shape)
for i1 in range(n_comp):
    for i2 in range(n_comp):
        if not (lower_CI[i1, i2] < 0 and upper_CI[i1, i2] > 0):
            sig_mask[i1, i2] = mean_CI[i1, i2]


plt.figure(); sns.heatmap(sig_mask, cmap = 'bwr', center = 0,
                           xticklabels=corrMat.columns.values, yticklabels=corrMat.index.values)


from sklearn.cluster import SpectralClustering
n_cluster = 5
clustering = SpectralClustering(n_clusters=n_cluster)
clustering.fit(sig_mask)

new_idxs = np.concatenate([np.where(clustering.labels_ == ii )[0] for ii in range(clustering.n_clusters)])

plt.figure(); sns.heatmap(sig_mask[:, new_idxs][new_idxs], cmap = 'bwr', center = 0,
                           xticklabels=corrMat.columns.values[new_idxs], yticklabels=corrMat.index.values[new_idxs])
