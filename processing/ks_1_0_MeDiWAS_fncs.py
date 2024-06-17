import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
import textwrap
from adjustText import adjust_text

GREY50 = "#7F7F7F"
GREY30 = "#4d4d4d"

# Loading data should have para for different group orderings
def default_order():
    '''    Second Version Order
    groupid = {'congenital anomalies': 1,
    'respiratory': 2,
    'infectious diseases': 3,
    'mental disorders': 4,
    'circulatory system': 5,
    'pregnancy complications': 6,
    'neoplasms': 7,
    'sense organs': 8,
    'injuries & poisonings': 9,
    'endocrine/metabolic': 10,
    'symptoms': 11,
    'musculoskeletal': 12,
    'digestive': 13,
    'dermatologic': 14,
    'hematopoietic': 15,
    'neurological': 16,
    'genitourinary': 17}
    '''

    groupid = {'congenital anomalies': 8,
    'respiratory': 15,
    'infectious diseases': 3,
    'mental disorders': 17,
    'circulatory system': 7,
    'pregnancy complications': 6,
    'neoplasms': 16,
    'sense organs': 4,
    'injuries & poisonings': 10,
    'endocrine/metabolic': 12,
    'symptoms': 1,
    'musculoskeletal': 2,
    'digestive': 9,
    'dermatologic': 13,
    'hematopoietic': 11,
    'neurological': 14,
    'genitourinary': 5}

    ''' Origional Order one
    groupid = {'hematopoietic': 1,
    'congenital anomalies': 2,
    'endocrine/metabolic': 3,
    'sense organs': 4,
    'circulatory system': 5,
    'digestive': 6,
    'infectious diseases': 7,
    'respiratory': 8,
    'neurological': 9,
    'genitourinary': 10,
    'musculoskeletal': 11,
    'dermatologic': 12,
    'neoplasms': 13,
    'injuries & poisonings': 14,
    'symptoms': 15,
    'pregnancy complications': 16,
    'mental disorders': 17}
    '''

    # N.B. for karin 
    # H 0 - 360
    # C 60 - 100
    # L 0 - 70
    # Hard makes more different colors
    # https://medialab.github.io/iwanthue/

    colorsdict = {
    'hematopoietic': "#ff003a", # red
    'congenital anomalies': "#523be5", # blue
    'infectious diseases': "#13c534", # green
    'sense organs':"#ff5dff", # pink
    'circulatory system': "#bf0052", # dark red
    'digestive': "#8eb847", # yellow-green
    'endocrine/metabolic': "#7a47a9", # purple
    'symptoms': "#009d3f", # green
    'pregnancy complications': "#ca45f7", # pink
    'genitourinary' : "#bd7100", # orange
    'injuries & poisonings': "#89207c", # purple
    'respiratory': "#fe8d4a", # orange
    'neoplasms': "#ff2baa", # hot pink
    'musculoskeletal': "#51008d", # dark purple
    'dermatologic': "#ed83f3", # light pink
    'neurological': "#9b0099", # purple
    'mental disorders': "#0173f0" # blue
    }

    # Sorted by diff
    colors = [
    "#ff5dff", # pink
    "#13c534", # green
    "#51008d", # dark purple
    "#fe8d4a", # orange
    "#523be5", # blue
    "#009d3f", # dark green
    "#ff2baa", # hot pink
    "#8eb847", # puke green
    "#89207c", # dim purple
    "#bd7100", # dim orange
    "#0173f0", # blue
    "#bf0052", # dark red
    "#ca45f7", # pink
    "#ff003a", # bright red
    "#7a47a9", # blue-purple
    "#ed83f3", # light pink
    "#9b0099"] # red-purple

    return groupid, colorsdict

def load_ICD(DATA_FOLD = 'data/interim/icd10_work', useDefaults=True, groupid=None):

    # Load in 
    ppl_xcldphe = pd.read_csv(os.path.join(DATA_FOLD,'proc', 'exclusion.csv'), index_col = 0)
    ppl_phecodes = pd.read_csv(os.path.join(DATA_FOLD,'proc', 'inclusion.csv'), index_col = 0)

    # Sort index apparently makes pandas run faster
    ppl_xcldphe = ppl_xcldphe.sort_index()
    ppl_phecodes = ppl_phecodes.sort_index()

    # TODO Karin you need to fix this, probably most people will have this in the same folder as other stuff
    ICDMAP_FOLD = 'data/external/ICD9_10_Mapping'

    phecodes = pd.read_csv(os.path.join(ICDMAP_FOLD, 'UKB_Phecode_v1-2b1_ICD_Mapping.txt'), sep='\t')
    phecodes_idx = phecodes.set_index('phecode').sort_index()

    if useDefaults:
        groupid, _ = default_order()
    elif groupid is None:
        raise AttributeError("If you are not using the default disease class ordering, please supply a dictionary of disease class (key) and order (value) pairs (groupid)")
    elif type(groupid) != dict:
        raise AttributeError("please supply groupid as a dictionary of disease class (key) and order (value) pairs")

    phecodes_idx['groupid'] = phecodes_idx['group'].map(groupid)

    return ppl_phecodes, ppl_xcldphe, phecodes_idx, groupid


def ICD_correlat(comp_nlz, ppl_phecodes, ppl_xcldphe, phecodes_idx, min_hits=1):
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
    cols = ppl_phecodes.columns

    keys = np.concatenate([
        ['groupid', 'group', 'phecode', 'desc'], 
        [f"-logp_{c}" for c in comp_nlz],
        [f"r_{c}" for c in comp_nlz], 
        [f"p_{c}" for c in comp_nlz]
    ])

    print("Merging Data...")
    new_use = pd.merge(ppl_phecodes, comp_nlz, on='eid')
    enough_hits = cols[new_use[cols].sum()>=min_hits]
    if len(enough_hits) < len(ppl_phecodes.T):
        n_drop = len(ppl_phecodes.T) - len(enough_hits)
        print(f"Dropping {n_drop} diseases because they have fewer than {min_hits} positive cases in the data set")
    
    new_use = new_use[[*enough_hits, *comp_nlz.columns.values]]
    new_use = new_use.join(ppl_xcldphe[enough_hits], on='eid',  rsuffix='_drop')

    mnhtn_data = {key: [] for key in keys}
    not_enough = []

    for col in tqdm(enough_hits):
        col_int = float(col)
        col = str(col)

        # First record column and category
        mnhtn_data['phecode'].append(col)
        mnhtn_data['groupid'].append(phecodes_idx.loc[col_int]['groupid'])
        mnhtn_data['group'].append(phecodes_idx.loc[col_int]['group'])
        mnhtn_data['desc'].append(phecodes_idx.loc[col_int]['description'])

        # Now compute pearson correlation
        for comp in comp_nlz:

            # we need to deal with nan values
            keep = ~np.logical_or(np.isnan(new_use[comp]),
                                    np.isnan(new_use[col]))

            # now drop those in the exclude columns
            # keep = keep.subtract(ppl_xcldphe[col])
            keep = keep.subtract(new_use[f'{col}_drop'])
            keep = keep>0
            
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

    if (len(not_enough) > 0) | (len(enough_hits) < len(cols)):
        print("NOTE: the following disease codes were not addressed in your data")
        print("Either (a) due to not enough non-nan values in one of the input dataset")
        print(f"Or (b) because there were more than {min_hits} people with the disease in group")
        print('Note you can adjust the latter by changing the "min_hits" parameter')
        dropped = [ii for ii in cols if ii not in enough_hits]
        dropped = np.concatenate([dropped, list(set(not_enough))])
        dropped = list(set(dropped))
        print('\t'+"\n\t".join([f"{a}: {phecodes_idx.loc[float(a)]['description']}" 
                            for a in dropped]))

    return multi.copy()


def ICD_DDR_correlat(comp_nlz, ICD_PCA, phecodes_idx=None):
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

    ICD_PCA
    Second dataframe is domain reduced disease data case
    Should have INDEX columns of eid
    to connect it to the brain data
    Column names are as follows
    ABBR-#_Model
    Where 
        ABBR  : 4-letter code of disease class
        #     : Component number
        Model : Model type (PCA, CCA, PLS). Association models had rs-fMRI input as 'Y' loadings


    phecodes_idx
    This dataframe has the definitions of all phecodes we have
    obtained through the load ICD function
    This will be used to innclude human readable labels
    and groupings

    If phecodes_idx not provided, I will use default information


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
        ['groupid', 'group', 'phecode', 'type', 'desc'], # At some point adding a description may be needed
        [f"-logp_{c}" for c in comp_nlz],
        [f"r_{c}" for c in comp_nlz], 
        [f"p_{c}" for c in comp_nlz]
    ])

    # Map abbr to group
    n_code = 4
    if phecodes_idx is None:
        groupid, _ = default_order()
        all_cats = list(groupid.keys())
    else:
        all_cats = np.unique(phecodes_idx['group'])
        groupid = phecodes_idx.groupby(['group', 'groupid']).size().reset_index()[['group', 'groupid']].values
        groupid = {g:c for (g, c) in groupid}
    dict_cat_code = {curr_cat[0:n_code].upper(): curr_cat for curr_cat in all_cats}

    print("Merging Data...")
    new_use = pd.merge(ICD_PCA, comp_nlz, on='eid')
    # in original code, we had something to deal with the binary nature of ICD data
    # where if there were fewer than min_hits positive cases we drop the disease altogether
    # With condensed, summarized data, we don't need this

    # We also do not need to account for exclusions

    mnhtn_data = {key: [] for key in keys}
    not_enough = []

    for col in tqdm(cols):
        col = str(col)
        cat_code = col.split('-')[0]
        gr = dict_cat_code[cat_code]

        # First record column and category
        mnhtn_data['phecode'].append(col)
        mnhtn_data['desc'].append(col.replace('_', ' '))
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

def findFDR(clreddf, col, thresBon):
    """
    Finds FDR threshold in the standard method
    As used in the sklearn function, but modified to determine
    the p value threshold in real terms
    rather than chose which features to keep, which is what sklearn does
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
    """
    sv = np.sort(clreddf[f'p_{col}'])
    sv = sv[~np.isnan(sv)]

    # The magic part of FDR
    # Each p-val gets compared to the Bon thres of n - w tests
    # where w is the order (smallest to largest) of the p value of interest
    # and n is the total number of tests
    pass_thres = sv <= thresBon*np.arange(1, 1 + len(sv))
    sel = sv[pass_thres]

    # pass_thres is a binary array of if the ith p-value passes the FDR limit
    # We will now check that only the smallest (most significant) p values pass the FDR limit
    # Additionally, FDR is such that the first instance of when a p-value fails to pass the FDR limit is the FDR limit
    # So we will check for that too

    if len(sel) > 0:
        thresFDR = (sv <= sel.max()).sum() * thresBon
    else:
        thresFDR = thresBon
    return(thresFDR)


def manhattan_plot(multi, lk, groupid=None, thres=0.05, n_t=None,
                   ylim=None, plot_height=7, useDefaults=True, colorsMap=None, 
                   label=False, label_thres='FDR', max_chars = 25):
    """
    Create a manhattan plot based on info in clreddf
    ylim changes the bounds of the plot
    ylim should be a tuple
    label will use the descriptions to add labels to the plot
    useDefaults uses the default colour scheme 
    """

    # ylim should be a tuple
    # Find FDR Thres
    if n_t is None:
        n_t = multi.shape[0]
    if thres is not None:
        thresBon = thres/n_t

        thresFDR = findFDR(multi, lk, thresBon)


    if useDefaults:
        groupid, colors = default_order()
        colorsMap= {groupid[ii]: colors[ii] for ii in groupid.keys()}
        grs = np.unique(multi.index.get_level_values(0))
        cat_order = [multi.loc[i, :].iloc[0]['group'] for i in grs]
        color_order = [colorsMap[i] for i in grs]
    elif colorsMap is None:
        raise AttributeError("If you are not using the default disease class ordering/coloring, please supply either a list of colors to use or a dictionary of disease class # id and colors pairs (colorsMap)")

    style = 'type' if 'type' in multi else None

    plot = sns.relplot(data=multi, x='i', y=f'-logp_{lk}',  edgecolor='k',
                        aspect=1.3, height=plot_height, hue='groupid',
                        legend=None, palette=colorsMap, style=style)
    # t_df = clreddf.groupby('groupid')['i'].median()
    # t_dfm = clreddf.groupby('groupid')['i'].max()[:-1]
    grs = np.unique(multi.index.get_level_values(0))

    t_df = [multi.loc[g]['i'].median() for g in grs]
    t_dfm = [multi.loc[g]['i'].max() for g in grs]
    t_dfm = t_dfm[:-1] # don't need a line on the far right of the graph

    plot.ax.set_ylabel('$-\\log_{10}(p)$')
    plot.ax.set_xlabel('')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot.ax.set_xticks(t_df)
        plot.ax.set_xticklabels(cat_order, rotation=90, ha='right')

    for xtick, color in zip(plot.ax.get_xticklabels(), color_order):
        xtick.set_color(color)

    plt.tick_params(axis='x', bottom=False)
    # plot.fig.suptitle(f'Manhattan plot of {lk}')
    plt.title(f'Manhattan plot of {lk}')
    if thres is not None:
        plt.axhline(y=-np.log10(thresFDR), color='k')
        plt.axhline(y=-np.log10(thresBon), color='k')
    else:
        plt.axhline(y=0, color='k')
    plt.axhline(y=0, color='k')


    [plt.axvline(x=xc, color='grey', linestyle='--') for xc in t_dfm]
    plot.fig.tight_layout()

    if ylim:
        plot.fig.tight_layout()
        plot.set(ylim=ylim)
        plot.fig.tight_layout()
    if thres is not None:
        locs, labels = plt.yticks()
        plt.yticks([*locs, -np.log10(thresBon), -np.log10(thresFDR)],
                    [*labels, 'BON', "FDR"])
                    
    plot.fig.tight_layout()

    if label == True: 
        if label_thres == 'FDR':
            lbl_thres = -np.log10(thresFDR)
        elif label_thres == 'Bon':
            lbl_thres = -np.log10(thresBon)
        else: 
            lbl_thres = label_thres
        n_rpl = 76
        
        rpl_x = multi['i'].values
        rpl_y = multi[f'-logp_{lk}'].values
        # To repel from thresFDR and thresBon
        if thres is not None:
            rpl_x2 = np.linspace(multi['i'].min(), multi['i'].max(), n_rpl)
            rpl_y2 = [[-np.log10(thresFDR), -np.log10(thresBon)][i%2] for i in np.arange(n_rpl)]

            rpl_x = np.append(rpl_x2, rpl_x)
            rpl_y = np.append(rpl_y2, rpl_y)

        # txt_col = if 'type' in multi else 'desc'
        texts = [plot.ax.text(p[1]['i'], p[1][f'-logp_{lk}'],
                            '\n'.join(textwrap.wrap(p[1]['desc'], width=max_chars, break_on_hyphens=False)), 
                            size = 'small', color=GREY30, backgroundcolor='white', fontstretch='condensed',
                            bbox={'pad':0, 'ec':'white',  'alpha':0.7, 'color':'white'} )
                for p in  multi[multi[f'-logp_{lk}']>lbl_thres].iterrows()]
        print(f"Currently adding {len(texts)} disease labels")
        print(f"Please be patient, more labels take longer to render & position")
        adjust_text(texts, rpl_x, rpl_y , ax = plot.ax, expand_text=(1.05, 1.25),
                    arrowprops=dict(arrowstyle='->', color=GREY50, relpos=(0.5, 1)))

    return plot      


def miami_plot(df_top, df_bot, lk, n_t=None, lbls=None, new_idx=None, 
               groupid=None, thres=0.05, thresFDRs = None, ylim=None, figsize=(15, 10), 
               useDefaults=True, 
               colorsMap=None, label=False, label_thres='FDR', max_chars = 25):
    """
    Creates a Miami plot, i.e. two manhattan plots,
    one above the x-axis (df_top) one below (df_bot)
    Expects TWO dataframes containing output from phenom_correlat function
    That is, all -logp and p values associated with
    all 977 behavioural phenotypes

    It is expected that the column names are the same in both dataframes
    That is, the column of interest exists in both dataframs
    and lk takes the same column from both dataframes to plot

    This function cannot plot two columns from the same dataframe

    lbls should be a tuple or list of size two
    containing the names of the groups
    e.g. Young/Old; Male/Female; High IQ/Low IQ; Lonely/Not Lonely etc.

    ylim should be a single value (e.g. 5)
    which indicates the upper/lower bound of the plot
    If unselected, it will be chosen as the largest -log(p) vals available
    """
    # Make a copy so you don't alter input dataframes
    clreddf = df_top.copy()
    clreddf2 = df_bot.copy()

    # ylim should be a single value
    # Find FDR Thres
    if n_t is None:
        n_t = clreddf.shape[0]
    if thres is not None:
        thresBon = thres/n_t
    else: 
        thresBon = 0.05/n_t

    if thresFDRs is None:
        thresFDR = findFDR(clreddf, lk, thresBon)
        thresFDR2 = findFDR(clreddf2, lk, thresBon)
    elif len(thresFDRs)!=2:
        raise AttributeError("thresFDRs should be a list of size 2, giving the FDRs to use on top and bottom respectively. You supplied: ", thresFDRs)
    else:
        thresFDR = thresFDRs[0]
        thresFDR2 = thresFDRs[1]

    # Get color maps
    if useDefaults:
        groupid, colors = default_order()
        colorsMap= {groupid[ii]: colors[ii] for ii in groupid.keys()}
        grs = np.unique(clreddf.index.get_level_values(0))
        cat_order = [clreddf.loc[i, :].iloc[0]['group'] for i in grs]
        color_order = [colorsMap[i] for i in grs]
    elif colorsMap is None:
        raise AttributeError("If you are not using the default disease class ordering/coloring, please supply either a list of colors to use or a dictionary of disease class # id and colors pairs (colorsMap)")

    # We need to rearrange both dataframes to be in the same order
    new_idx = np.arange(len(grs)) if new_idx is None else new_idx

    # Replace infinite correlations with biggest hit in group
    max_corr = max([clreddf[f'-logp_{lk}'].max(), 
                    clreddf[f'-logp_{lk}'].max()])
    clreddf.replace([np.inf, -np.inf], 1.1*max_corr, inplace=True)
    clreddf2.replace([np.inf, -np.inf], 1.1*max_corr, inplace=True)

    # If the data has different shapes, use them
    style = 'type' if 'type' in clreddf else None

    order_top = clreddf.index.get_level_values(1)
    order_bottom = clreddf2.index.get_level_values(1)

    if (order_top == order_bottom).sum() != clreddf2.shape[0]:
        # The columns are out of order
        # We need to rearrange them
        clreddf2 = clreddf2.reorder_levels(['phecode', 'groupid'])
        clreddf2 = clreddf2.loc[order_top]
        clreddf2['i'] = np.arange(clreddf2.shape[0])
        clreddf2 = clreddf2.reorder_levels(['groupid', 'phecode'])

    # Select/Define Limits of the graph
    if ylim:
        ymaxx = ylim
    else:
        y1 = clreddf[clreddf[f'-logp_{lk}'].notnull()][[f'-logp_{lk}']]
        y1 = np.abs(np.array(y1)).max()
        y2 = clreddf2[clreddf2[f'-logp_{lk}'].notnull()][[f'-logp_{lk}']]
        y2 = np.abs(np.array(y2)).max()
        ymaxx = max(y1, y2) + 1

    # Begin Plotting
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=clreddf, x='i', y=f'-logp_{lk}', edgecolor='k', ax=ax,
                    hue='groupid', palette=colorsMap, legend=None, style=style)
    grs = np.unique(clreddf.index.get_level_values(0))

    # Define Boundarys of Categories
    # To make the lines (at max)
    # and for labels (max - 3)

    t_df = [clreddf.loc[g]['i'].median() for g in grs]
    t_dfm = [clreddf.loc[g]['i'].max() for g in grs]
    t_dfm = t_dfm[:-1] # don't need a line on the far right of the graph

    # t_df = clreddf.groupby('catid')['i'].max() - 3
    # t_dfm = clreddf.groupby('catid')['i'].max()[:-1]

    # # Shift select cat labels for enhanced readibility
    # cat_size = t_df + 3 - clreddf.groupby('catid')['i'].min()
    # t_df[31] = t_df[31] + 3  # Bone Density
    # t_df[30] = t_df[30] - cat_size[30] + 25  # Blood Assays
    # t_df[51] = t_df[51] - cat_size[51] + 40  # Mental Health

    clreddf2[f'-logp_{lk}_flip'] = -1 * clreddf2[f'-logp_{lk}']
    sns.scatterplot(data=clreddf2, x='i', y=f'-logp_{lk}_flip', edgecolor='k', ax=ax,
                    hue='groupid', palette=colorsMap, legend=None, style=style)

    plt.ylim(-1*ymaxx, ymaxx)
    texts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if lbls is not None:
            ylbl = f'{lbls[1]}\t'+'\t$-\\log_{10}(p)$\t' + f' \t{lbls[0]}'
        else:
            ylbl = '$-\\log_{10}(p)$'  # Latex formatting

        ax.set_ylabel(ylbl)
        ax.set_xlabel('')

        ax.set_xticklabels('', rotation=90, ha='left')
        ax.set_xticks(t_df)
        ax.xaxis.tick_top()

        for xtick, cat, color in zip(t_df, cat_order,  color_order):
            plt.text(xtick + 25, -ymaxx + ymaxx/25, cat,
                     c=color, ha='right', rotation=90)

    plt.tick_params(axis='x', bottom=False, top=False)
    # fig.suptitle(f'Manhattan plot of {lk}');
    plt.axhline(y=-np.log10(thresFDR), color='k')
    plt.axhline(y=np.log10(thresFDR2), color='k')  # Lower Manhattan
    plt.axhline(y=-np.log10(thresBon), color='k')
    plt.axhline(y=np.log10(thresBon), color='k')  # Lower Manhattan
    plt.axhline(y=0, color='k')
    [plt.axvline(x=xc, color='k', linestyle='--') for xc in t_dfm]

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    labels = [item.get_text() for item in ax.get_yticklabels()]
    locs, _ = plt.yticks()
    new_text = [t.replace('-', '') for t in labels]
    plt.yticks([*locs, -np.log10(thresBon), -np.log10(thresFDR),
                np.log10(thresBon), np.log10(thresFDR2)],
               [*new_text, 'BON', "FDR", 'BON', "FDR"])
    plt.tight_layout()
    if lbls is not None:
        ypos = ymaxx - ymaxx/10
        texts.append(plt.text(950 - 8*len(lbls[0]), ypos, lbls[0]))
        texts.append(plt.text(950 - 8*len(lbls[1]), -1*ypos, lbls[1]))
    # adjust_text(texts)
    return



def hits(clreddf, lk, to_file=False, hits_file=None,
         hits_folder=None, extra_notes=None, thresBon=None,
        thres=0.05, useFDR=True, sort=False):
    """
    Output all hits associated with column lk in input clreddf

    By default outputs everything above FDR threshold
    can change to only Bon hits with useFDR=False

    By default, it will print hits to terminal

    If you want to save to a file, you need to_file=True
    By default, it prints to hits_{lk}.txt if to_file is True
    you can customize which file it goes to by passing a filepath to hits_file
    Note that hits_file doesn't do anything unles to_file is set to true

    you can add more information to output with extra_notes param

    Setting sort=True will output hits from most to least significant (per domain)

    Format is a tuple with the following value
    (x val in graph, phecode, -log10(p), r, descriptive column name)
    """
    n_t = clreddf.shape[0]
    if thresBon is None:
        thresBon = thres/n_t
    thresFDR = findFDR(clreddf, lk, thresBon)

    if to_file:
        # Save a ref to the original standard output
        original_stdout = sys.stdout

        if hits_file is None:
            hits_file = f"hits_{lk}.txt"

        if hits_folder is not None:
            hits_file = os.path.join(hits_folder, hits_file)
        

        with open(os.path.join(hits_file), 'w') as f:
            # Change the standard output to the file we created.
            sys.stdout = f
            # Print out all hits
            printing(clreddf, lk, 
                     extra_notes, useFDR, thresBon, thresFDR, sort)
            # Reset the standard output to its original value
            sys.stdout = original_stdout
    else:
        # Print hits to original output (terminal)
        printing(clreddf, lk,
                 extra_notes, useFDR, thresBon, thresFDR, sort)
    return

  
def printing(clreddf, lk,
             extra_notes, useFDR, thresBon, thresFDR, sort):
    """
    Auxillary function called by hits
    Created to aid readability
    and make it possible to print to file (or not)
    using the same function
    """
    # This prints out all significant columns for the current component
    print('\nComponent', lk)
    if extra_notes is not None:
        print(extra_notes)

    cn = lk  

    print(f"MAX: {clreddf[f'-logp_{cn}'].max():.3f}")
    print(f"N ABOVE BON ({-np.log10(thresBon):.2f}): ",
            (clreddf[f'-logp_{cn}'] > -np.log10(thresBon)).sum())
    print(f"N ABOVE FDR ({-np.log10(thresFDR):.2f}): ",
            (clreddf[f'-logp_{cn}'] > -np.log10(thresFDR)).sum())

    thresUsed = thresFDR if useFDR else thresBon
    print("Fine Grain info above ", 'FDR' if useFDR else 'Bon')
    print("Printing out " +
            str((clreddf[f'-logp_{cn}'] > -np.log10(thresUsed)).sum())
            + " hits")

    sigcol = clreddf[(clreddf[f'-logp_{lk}'] > -np.log10(thresUsed))].reset_index()

    for catn in np.unique(sigcol['groupid']):
        sigcol_subset = sigcol[sigcol['groupid'] == catn]
        # sorting only works if you
        #   (a) have hits to sort
        #   (b) have more than one hit
        if sort and not sigcol_subset.empty and sigcol_subset.shape[0]>1:
            sigcol_subset = sigcol_subset.sort_values(by=f'p_{lk}')
        cs = [c for c in sigcol_subset['phecode']]

        if len(cs) > 0:
            print(f"\n COMP {lk}",
                    sigcol_subset['group'].values[0].upper(),
                    f" ({len(sigcol_subset)} hit(s))")

            print(*[(iv,  c, f"{pp:.2f}",f"{rr:.2f}", pd)
                    for ii, (iv, c, pp, rr, pd) in enumerate(
                    sigcol_subset[['i', 'phecode', f'-logp_{lk}', f'r_{lk}', 'desc']]
                    .values)], sep='\n')
    return

def hits_csv(clreddf, lk, OUT_FOLD, thres=0.05, thresBon=None, useFDR=True, sort=False):

    hits_keys = ['i', 'phecode', '-log10(p)', 'r', 'description']

    n_t = clreddf.shape[0]
    if thresBon is None:
        thresBon = thres/n_t
    thresFDR = findFDR(clreddf, lk, thresBon)
    thresUsed = thresFDR if useFDR else thresBon

    catns = np.unique(clreddf.index.get_level_values(0))

    sigcol = clreddf[(clreddf[f'-logp_{lk}'] > -np.log10(thresUsed))].reset_index()
    if sigcol.empty:
        print(f"No Hits were associated with {lk}")
        print("Now terminating without creating csv file")
        return

    sub = 'FDR' if useFDR else 'Bon'
    with pd.ExcelWriter(os.path.join(OUT_FOLD, f'hits_{lk}_N{len(sigcol)}_{sub}.xlsx')) as writer:  

        for catn in catns:

            sigcol_subset = sigcol[sigcol['groupid'] == catn]
            if sigcol_subset.empty:
                continue
            if sort and sigcol_subset.shape[0]>1:
                # sigcol_subset = sigcol_subset.sort_values(by=f'p_{lk}')
                sigcol_subset = sigcol_subset.sort_values(by=f'-logp_{lk}', ascending=False)
            cs = [c for c in sigcol_subset['phecode']]

            col_data = {key: [] for key in hits_keys}

            cat_name = sigcol_subset['group'].values[0].replace('/', ' or ')
            
            if len(cs) > 0:
                # print(f"\n COMP {lk}",
                #         cat_name.upper(), f"{len(cat_name)} char"
                #         f" ({len(sigcol_subset)} hit(s))")

                for ic, c in enumerate(cs):
                    col_data['i'].append(sigcol_subset['i'].iloc[ic])
                    col_data['phecode'].append(float(sigcol_subset['phecode'].iloc[ic]))
                    col_data['-log10(p)'].append(sigcol_subset[f'-logp_{lk}'].iloc[ic])
                    col_data['r'].append(sigcol_subset[f'r_{lk}'].iloc[ic])
                    col_data['description'].append(sigcol_subset['desc'].iloc[ic].replace('/', ' or '))
                        
                dftest = pd.DataFrame(col_data)
                dftest.to_excel(writer, sheet_name=f"{cat_name} ({len(sigcol_subset)})",
                                index = False)
        writer.save()