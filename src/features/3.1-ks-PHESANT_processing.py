
###########################################################################################
# Step 0.5
# Pre-FUNPACK
# Extracting columns
# -----------------NOTES---------------------------------------------------------------------
# For whatever reason, FUNPACK is struggling to extract
#   (1) only subjects from the given list
#        (if i'm asking to extract particular columns at the same time)
#   (2) Only columns associated with specified visit
#         (I'm getting pollution, which is recorded once at visit 0, when I specified only visit 2)
#
# To circumvent this, because I know I only want visit 2 (3) and the list of people
# I'm just going to manually extract the people and columns I want
# To not let FUNPACK mess it up
# This will also have repercussions for the funpack code I use later
###########################################################################################

# I may be missing import statements, so add if you find something I missed
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)

FULL_DATA_FOLDER = 'data/raw'
print("Checking for ukbb")
t1= time.time()
if 'ukbb' not in locals():
    print("UKBB not found, now loading")
    ukbb = pd.read_csv(os.path.join(FULL_DATA_FOLDER, 'ukb40500_cut_merged.csv'), low_memory = False)
ukbb.shape
t2=time.time()
print(f"Loading took  {(t2-t1)/60:.4f} min ")

ukbb_idx = ukbb.set_index('eid').sort_index()

SAVE_FOLDER = 'models/Asymm_Patterns/Changes/2023_02_post_deconf'
df_amt_chg_rate = pd.read_csv(os.path.join(SAVE_FOLDER, 'Rate_Adjusted', 'Rate_Amount_Change.csv'), index_col=0)
ukbb_reduced = ukbb_idx.loc[df_amt_chg_rate.index]

cl_v2 = [ii for ii in ukbb_idx if '-2.' in ii]
cl_v3 = [ii for ii in ukbb_idx if '-3.' in ii]

OUT_FOLDER = "data/processed/2309_Investigate_double"
SMALL_FILE = lambda v: f'ukbb_1425_vi{v}.csv'
ukbb_reduced[cl_v3].to_csv(os.path.join(OUT_FOLDER, SMALL_FILE(3)))
ukbb_reduced[cl_v2].to_csv(os.path.join(OUT_FOLDER, SMALL_FILE(2)))

###########################################################################################
# Step 1
# FUNPACK
# -----------------NOTES---------------------------------------------------------------------
# Needs to be run twice (once per visit)
# '-vi 2' and '-vi 3' should be used
# I've extracted a csv of eids for those with two visits already
# 1425 individuals (n.b. these ppl have data for all 3 modalities (dMRI, T1 MRI)
#
# Two methods:
# 1. Extract all in go. This worked once and then not again (2 funpack runs)
# 2. Extract first the ~1500 ppl, and then the categories of interest (3 funpack runs)
###########################################################################################

DATA_FOLDER = "~/Documents/asymm_pat_age_progression/data"
# DATA_FOLDER = "Users/karinsaltoun/dblabs/asymm_pat_age_progression"
LOCAL_FOLDER = "~/dblabs/asymm_pat_age_progression"
# OUT_FOLDER = "data/processed/2309_Investigate_double"
PHESANT_FOLDER = "PHES_out"  # this is a file that should exist inside of OUT_FOLDER

EID_FILE = "interim/eid-codes/1-0-eid_2_visit.csv"
CAT_FILE = "processed/ukbb_double_visit_phenomes/categories_amend_220113.tsv"
OUT_FILE = lambda x: f"processed/ukbb_double/ukb_mh_miller_vi{x}.csv"
REDUCE_FILE = "processed/ukbb_double/ukb_2visit.csv"
DATA_FILE = "~/Documents/Karin\ Asymm\ Project/ukbb_no_idx.csv"

'''
# METHOD 1
# ALL IN ONE GO
FUNPACK_CMD =  lambda vi: f"""
 funpack -wl -n -n -n -wde -ws -ow \
 -vi {vi} \
--subject {os.path.join(DATA_FOLDER, EID_FILE)}  \
 -cfg fmrib \
 --category_file {os.path.join(DATA_FOLDER, CAT_FILE)}  \
 -c 3 -c 10 -c 11 -c 12 -c 13 -c 14 \
 -c 20 -c 21 -c 22 -c 30 -c 32 -c 51 \
{os.path.join(DATA_FOLDER, OUT_FILE(vi))}  \
 {DATA_FILE}
"""

# FUNPACK COMMANDS
# RUN the output of this in terminal (with funpack v2.5.0)
VISIT = 2 # 2 or 3
print(FUNPACK_CMD(VISIT))

VISIT = 3 # 2 or 3
print(FUNPACK_CMD(VISIT))

# METHOD 2
# BREAK IT UP
FUNPACK_CMD_CUT = f"""
 funpack -wl -n -n -n -wde -ws -ow \
 --subject {os.path.join(DATA_FOLDER, EID_FILE)} \
 --category_file {os.path.join(DATA_FOLDER, CAT_FILE)}  \
 {os.path.join(DATA_FOLDER, REDUCE_FILE)}  \
 {DATA_FILE}
"""

FUNPACK_CMD =  lambda vi: f"""
 funpack -wl -n -n -n -wde -ws -ow \
 -vi {vi} \
 --category_file {os.path.join(DATA_FOLDER, CAT_FILE)}  \
 -c 3 -c 10 -c 11 -c 12 -c 13 -c 14 \
 -c 20 -c 21 -c 22 -c 30 -c 32 -c 51 \
 -cfg fmrib {os.path.join(DATA_FOLDER, OUT_FILE(vi))}  \
 {os.path.join(DATA_FOLDER, REDUCE_FILE)}
"""


# FUNPACK COMMANDS
# RUN the output of this in terminal (with funpack v2.5.0)

print(FUNPACK_CMD_CUT)

VISIT = 2 # 2 or 3
print(FUNPACK_CMD(VISIT))

VISIT = 3 # 2 or 3
print(FUNPACK_CMD(VISIT))

'''

FUNPACK_FOLDER = lambda v: f'double_cohort_funpack_v{v}'
UKBB_FILE = lambda v: f'funpack_ukbb_proc_v{v}.csv'
FUNPACK_CMD =  lambda vi: f"""
 funpack -wl -n -n -n -wde -ws -ow \
 -vi {vi} \
 --category_file {os.path.join(DATA_FOLDER, CAT_FILE)}  \
 -c 3 -c 10 -c 11 -c 12 -c 13 -c 14 \
 -c 20 -c 21 -c 22 -c 30 -c 32 -c 51 \
 -cfg fmrib   \
 {os.path.join(LOCAL_FOLDER, OUT_FOLDER, FUNPACK_FOLDER(vi),  UKBB_FILE(vi))} \
 {os.path.join(LOCAL_FOLDER, OUT_FOLDER, SMALL_FILE(vi))}
"""


# FUNPACK COMMANDS
# RUN the output of this in terminal (with funpack v2.5.0)

VISIT = 2 # 2 or 3
Path(os.path.join(OUT_FOLDER, FUNPACK_FOLDER(VISIT))).mkdir(parents=True, exist_ok=True)
print(FUNPACK_CMD(VISIT))

VISIT = 3 # 2 or 3
Path(os.path.join(OUT_FOLDER, FUNPACK_FOLDER(VISIT))).mkdir(parents=True, exist_ok=True)
print(FUNPACK_CMD(VISIT))

###########################################################################################
# Step 1b
# CATEGORY MAPPER
# -----------------NOTES---------------------------------------------------------------------
# Technically not needed until you get to the plotting/output hits stage
# Does need the results from FUNPACK however
# Note here and above, I'm using a categories_amend file
# which is the categories file i've extracted internally from funpack
# with the addition of :
#       fixing miscast FEV variables  (from cognitive -> physiscal general)
#       fixing miscast 'illness of father' (from food -> lifestyle gen)
#       fixing miscast medication taken (from cogn -> health)
###########################################################################################

CAT_MAPPER = lambda vi: f"""
python /Users/karinsaltoun/Documents/Asymm_Code/cat_mapping.py \
-wn {os.path.join(DATA_FOLDER, CAT_FILE)}  {os.path.join(DATA_FOLDER, OUT_FILE(vi))}
"""

# RUN the output of this in terminal
VISIT = 2 # 2 or 3
print(CAT_MAPPER(VISIT))

VISIT = 3 # 2 or 3
print(CAT_MAPPER(VISIT))


###########################################################################################
# Step 1a
# PHESANT PREPROCESSING
# translate FUNPACK output to R-readable format
# ----------------------------------------
# PROBLEM: PHESANT is written in R
# SOLUTION: need to rename all the columns
# pipe into phesant twice and later undo the SS that phesant does
# PHESANT QUIRKS
#       First column MUST be userID otherwise PHESANT crashes/can't process data
#       PHESANT needs 709-0.0 (first visit # in household)
#
# PROBLEM: PHESANT will process/standard scale on a whole phenotype/columns automatically
#       When performing phesant on two separate csvs, the phesant output is no longer comparable
#       Since same phenotype has been standard scaled on a different scale
#       Therefore a value of A for a phenotype visit 1 is not equivalent to A for phenotype visit 2
# SOLUTION: Pipe into PHESANT twice (Once per visit)
#       For all values labelled as continuous,
#           (1) Take the original SS from input data
#           (1.5) Check for quirks in the data-log from PHESANT
#           (2) Undo PHESANT standard scaler
#           (2.5) Check that the data matches original
#           (3) Subtract
#           (4) Apply Standard Scaler on the subtracted data

###########################################################################################

ukbb_y_v1i = pd.read_csv(os.path.join(LOCAL_FOLDER, OUT_FOLDER, FUNPACK_FOLDER(2), UKBB_FILE(2)), low_memory=False, index_col=0)
ukbb_y_v2i = pd.read_csv(os.path.join(LOCAL_FOLDER, OUT_FOLDER, FUNPACK_FOLDER(3), UKBB_FILE(3)), low_memory=False, index_col=0)

#PHESANT QUIRK
#It requires 709_0_0 for something else it doesn't run
ukbb_y_v2i_T = ukbb_y_v2i.copy().merge(ukbb_idx[['709-0.0']], left_index=True, right_index=True)
ukbb_y_v1i_T =ukbb_y_v1i.copy().merge(ukbb_idx[['709-0.0']], left_index=True, right_index=True)

# Manual entry
# The weight category got dropped by funpack
# Likely due to the redundancy requirement
# Since it is important (imo)
# I'm manually adding it in
ukbb_y_v2i_T['21002-3.0'] = ukbb_reduced['21004-3.0']
ukbb_y_v1i_T['3160-2.0'] = ukbb_reduced['3160-2.0']

# Make R readable
curr_cols_v1i = ukbb_y_v1i_T.columns.values
curr_cols_v1i = ['x'+col.replace('.','_').replace('-', '_') for col in curr_cols_v1i]
curr_cols_v2i = ukbb_y_v2i_T.columns.values
curr_cols_v2i = ['x'+col.replace('.','_').replace('-', '_') for col in curr_cols_v2i]

PHESANT_FILE = lambda v : f"ukbb_phenoR_concat_redone0904_vi{v}.csv"
newdf_v1i = pd.DataFrame(ukbb_y_v1i_T.reset_index().values, columns = ['eid', *curr_cols_v1i])
newdf_v2i = pd.DataFrame(ukbb_y_v2i_T.reset_index().values, columns = ['eid', *curr_cols_v2i])
newdf_v1i.to_csv(os.path.join(OUT_FOLDER, PHESANT_FILE(2)), index = False)
newdf_v2i.to_csv(os.path.join(OUT_FOLDER, PHESANT_FILE(3)), index = False)

###########################################################################################
# Step 1b
# PHESANT
# ----------------------------------------
# If you're running this for the first time, there are various things you must do to set up PHESANT/ R
# Which have been long enough ago that i dont remember
#
# To run the code,
# You need to navigate to the WAS folder in PHESANT
# (I downloaded the package from the github page)
#
# resDir must be an EXISTING folder
#
# Even though I have standardise = False it doesn't seem to do anything
#
# We are in the following folder:
# /Users/karinsaltoun/Documents/PHESANT-master_20230828/WAS
# Where I have changed line 173 in testContinuous.r
# Before and after:
#       phenoIRNT = irnt(phenoAvg)
#       phenoIRNT = phenoAvg
#
# I have also removed the skip if less than 500 entries rule
###########################################################################################

PHESANT_FOLDER = lambda v: f"PHES_out_funpack_separate_changeR_vi{v}"
PHESANT_CMD = lambda v: f"""
Rscript phenomeScan.r \
--phenofile="{os.path.join(LOCAL_FOLDER, OUT_FOLDER, PHESANT_FILE(v))}" \
--variablelistfile="../variable-info/outcome-info.tsv" \
--datacodingfile="../variable-info/data-coding-ordinal-info.txt" \
--resDir="{os.path.join(LOCAL_FOLDER, OUT_FOLDER, PHESANT_FOLDER(v))}/" \
--standardise=FALSE \
--userId="eid" \
--save \
"""

from pathlib import Path
# RUN the output of this in terminal
# Inside the WAS folder of phesant
# The phesant folder MUST EXIST
Path(os.path.join(OUT_FOLDER, PHESANT_FOLDER(2))).mkdir(parents=True, exist_ok=True)
print(PHESANT_CMD(2))
print('\n\n')

Path(os.path.join(OUT_FOLDER, PHESANT_FOLDER(3))).mkdir(parents=True, exist_ok=True)
print(PHESANT_CMD(3))


###########################################################################################
# Step 3a
# PHESANT concatenation
# ----------------------------------------
# Load and Concatenate PHESANT Outputs
# Input is the doubled file which has two entries per person
#
# PHESANT will output into 4 diff data files
# categorical unordered needs to be 'get dummy'ified
#
# Once cat-unordered is dummified
# we add in all the other data types as-is
# merging on the basis of userid
#
# We notice that there is some oddities in the continous variables
# So we manually check that they all look the same as input
# If not, we take the data from the original
# After checking for any nan reassignments
###########################################################################################

import processing.manhattan_plot_util as man_plot
def load_desc(BASE_FOLDER='/Users/karinsaltoun/Documents/UKBB_files', y_group="_miller_mh_v1"):
    fname = os.path.join(BASE_FOLDER, f'ukbb{y_group}_description.txt')
    y_desc_dict = pd.read_csv(fname, sep='\t', header=None,
                              index_col=0).to_dict()[1]
    fname = os.path.join(BASE_FOLDER, f'ukbb{y_group}_cat_map.csv')
    y_cat_dict = pd.read_csv(fname, index_col=0)
    return y_desc_dict, y_cat_dict

y_desc_dict, y_cat_dict = load_desc()

phesdf_full = {}
newdf_v1i_idx = newdf_v1i.set_index('eid')
newdf_v2i_idx = newdf_v2i.set_index('eid')
cont_col = {}
weird_col = {}
phes_desc_df = {}

for vi in [2, 3]:
    print('\n\nVISIT ', vi)
    PHESANT_FOLD = os.path.join(LOCAL_FOLDER, OUT_FOLDER, PHESANT_FOLDER(vi))
    phesant_files = ['data-catunord-all.txt', "data-binary-all.txt", "data-catord-all.txt", 'data-cont-all.txt']

    phesdf = pd.read_csv(os.path.join(PHESANT_FOLD, phesant_files[0]), index_col=0)
    #This is categorical unordered, so we dummify everying in it
    phesdf = pd.get_dummies(phesdf, columns  = phesdf.columns)
    for ii in range(1, 4):
        if ii == 3:
            continue
        print(phesant_files[ii])
        phesdf1 = pd.read_csv(os.path.join(PHESANT_FOLD, phesant_files[ii]), index_col=0)
        print(phesdf1.shape)
        # phesdf = phesdf.merge(phesdf1, on='userID', how='left')
        phesdf = phesdf.merge(phesdf1, left_index=True, right_index=True, how='left')

    # Processing Continuous Data
    # PHESANT cannot undo the standard scaling by itself
    # I have changed part of the testContinuous.r package
    # as per above notes
    # however there is still an issue with some of the columns
    # therefore I want to match columns to original
    # Also read in the datalog to check for reassignments

    phesdf_cont = pd.read_csv(os.path.join(PHESANT_FOLD, phesant_files[3]), index_col=0)
    print(phesdf_cont.shape)

    # Combine data which came through phesant with what went into phesant
    # This was used for diagnostics to make sure data hasn't been changed somehow when going through Phesant
    translate_phesant = lambda col, vi: f'x{col}_{vi}_0'
    cont_org_merge = phesdf_cont.copy().merge(newdf_v1i_idx if vi == 2 else newdf_v2i_idx, left_index=True, right_index=True)

    # Scroll through the data
    weird_col_mini = []
    man_check = []
    keep_col = []
    for col in phesdf_cont:
        if translate_phesant(col, vi) in cont_org_merge:
            # Column is in original and phesant processed
            # Make sure all non-na values are equivalent
            mask = cont_org_merge[col].notna() & cont_org_merge[translate_phesant(col, vi)].notna()
            success = cont_org_merge.loc[mask, col].astype('float')\
                        .equals(cont_org_merge.loc[mask, translate_phesant(col, vi)].astype('float'))
            if success == False:
                weird_col_mini.append(col)
                keep_col.append(translate_phesant(col, vi))
            else:
                keep_col.append(col)
        else:
            # Column not in both original and phesant processed
            # Usually for data which is spread across multiple inputs (e.g. 2.1, 2.2, 2.3)
            if vi == 2:
                blah = [i for i in newdf_v1i_idx if i.startswith('x'+col)]
            elif vi == 3:
                blah = [i for i in newdf_v2i_idx if i.startswith('x'+col)]
            if len(blah) == 0:
                print('Not found: ', col)
            mask = cont_org_merge[col].notna() & (cont_org_merge[blah].notna().sum(1) == len(blah))
            success = cont_org_merge.loc[mask, col].equals(cont_org_merge[blah].mean(1).loc[mask])
            if success == False:
                weird_col_mini.append(col)
            keep_col.append(col)
            man_check.append(col)

    # Load in phesant explainers of "weird" columns
    # Was for testing, but ended up using the full dataset below
    descriptions = {}
    with open(os.path.join(OUT_FOLDER, PHESANT_FOLDER(vi), 'data-log-all.txt')) as file:
        for line in file:
            col = line.rstrip().split('_')[0]
            if col in man_check:
                descriptions[col] = line
            if col in weird_col_mini:
                descriptions[col] = line
    phes_desc_df_mini = pd.DataFrame.from_dict(descriptions, orient='index', columns=['Full_Desc'])
    phes_desc_df_mini['reassignments'] = phes_desc_df_mini['Full_Desc'].map(lambda x: [a for a in x.split('||') if 'reassignments' in a])

    # All the phesant explaination
    descriptions = {}
    with open(os.path.join(OUT_FOLDER, PHESANT_FOLDER(vi), 'data-log-all.txt')) as file:
        for line in file:
            col = line.rstrip().split('_')[0]
            descriptions[col] = line
    phes_desc_df[vi] = pd.DataFrame.from_dict(descriptions, orient='index', columns=['Full_Desc'])
    phes_desc_df[vi]['reassignments'] = phes_desc_df[vi]['Full_Desc'].map(lambda x: [a for a in x.split('||') if 'reassignments' in a]).map(lambda x: x[0] if len(x) !=0 else None)
    phes_desc_df[vi]['output'] = phes_desc_df[vi]['Full_Desc'].map(lambda x:  x.split('results')[1] if 'results' in x else '')
    phes_desc_df[vi]['order'] = phes_desc_df[vi]['Full_Desc'].map(lambda x: [a.split('order: ')[1] for a in x.split('||') if 'order:' in a]).map(lambda x: x[0] if len(x) !=0 else None)


    # Here are the strange columns
    for col in weird_col_mini:
        print(col, ' : ', y_desc_dict[man_plot.translate(col, y_cat_dict)[0]])

    # Push continuous columns (get subtracted, must scale) and full dataset to dictonary
    cont_col[vi] = keep_col
    weird_col[vi] = weird_col_mini
    phesdf_full[vi] = phesdf.merge(cont_org_merge[keep_col], left_index=True, right_index=True, how='left')


###########################################################################################
# Step 3a.5
# PHESANT clean up
# ----------------------------------------
# Some PHESANT outputs have been processed differently
# In that they are categorized as binary output in one time point (2 options)
# and the other timepoint is cat ordered
# With different values encoding the same response
# Here I try to identify where this occurs, and fix it
#
# Note that this also occurs in the categorical (unordered) group
#  but that is easier to deal with (and is dealt with later)
# Because if you're not part of the group... youre not part of the group
# Which allows us to set them as all zero
###########################################################################################
CLEAN_DIFF_CAT = True
if CLEAN_DIFF_CAT:
    # FInd things that are processed differently
    phes_desc_df_comb = phes_desc_df[2].merge(phes_desc_df[3], right_index=True, left_index=True, suffixes=['_v2', '_v3'])
    weird_col_mini = phes_desc_df_comb['output_v2'] != phes_desc_df_comb['output_v3']
    phes_desc_df_comb[weird_col_mini][['output_v2', 'output_v3']]
    '''
    In [934]: phes_desc_df_comb[weird_col_mini].shape
    Out[934]: (23, 6)
    '''
    # Most of these are categorical

    # These columns need special attention
    # Only 8 of them really need work to be done
    # The rest are unordered binary, whicblah[3]h are dealt with later
    # Some are missing in one but not the other
    mismatch =  ['5790', '5556', '6770', '4294', '4253', '403', '398', '2877']

    # Notes on each:
    # 5790 is categorical, so it can be ignored
    # 5556 is categorical, so it can be ignored
    # 6770 exists in visit 2 but not visit 3, so it could be ignored
    # 4294 is categorical in visit 2 but not in 3
    '''
    In [85]: [i for i in phesdf_full[3] if i.startswith(mismatch[3])]
    Out[85]: ['4294']

    In [86]: [i for i in phesdf_full[2] if i.startswith(mismatch[3])]
    Out[86]: ['4294_0.0', '4294_1.0', '4294_9.0']

    In [87]: phes_desc_df_comb.loc[mismatch[3]]['Full_Desc_v2']
    Out[87]: '4294_2|| CAT-SINGLE || Inc(>=10): 1(1134) || Inc(>=10): 0(45) || Inc(>=10): 9(14) || CAT-SINGLE-UNORDERED || reference: 1=1134 || SUCCESS results-notordered-logistic \n'

    In [88]: phes_desc_df_comb.loc[mismatch[3]]['Full_Desc_v3']
    Out[88]: '4294_3|| CAT-SINGLE || Inc(>=10): 1(1314) || Inc(>=10): 0(78) || Removed 9: 6<10 examples || CAT-SINGLE-BINARY || sample 78/1314(1392) || SUCCESS results-logistic-binary \n'

    In [89]: phesdf_full[3][mismatch[3]].value_counts()
    Out[89]:
    1.0    1314
    0.0      78
    Name: 4294, dtype: int64
    '''
    phes_desc_df_comb.loc[mismatch[3]]['Full_Desc_v2']
    phesdf_full[3] = pd.get_dummies(phesdf_full[3], columns  = [mismatch[3]])
    # 4253 has different encodings
    '''
    In [1001]: phesdf_full[3][mismatch[4]].value_counts()
    Out[1001]:
    3011.0    852
    3010.0    494
    Name: 4253, dtype: int64

    In [1002]: phesdf_full[2][mismatch[4]].value_counts()
    Out[1002]:
    1.0    608
    0.0    309
    2.0    271
    Name: 4253, dtype: int64
    In [1003]: phes_desc_df_comb.loc[mismatch[4]]['Full_Desc_v2']
    Out[1003]: '4253_2|| INTEGER || CONTINUOUS || >20% IN ONE CATEGORY || Bin 1: <3011, bin 2: ==3011, bin 3: >3011 || cat N: 309, 608, 271 || CAT-ORD || order: 0|1|2 || num categories: 3 || SUCCESS results-ordered-logistic\n'

    In [1004]: phes_desc_df_comb.loc[mismatch[4]]['Full_Desc_v3']
    Out[1004]: '4253_3|| INTEGER || Inc(>=10): 3010(494) || Inc(>=10): 3011(852) || Removed 3012: 6<10 examples || Removed 3009: 2<10 examples || sample 494/852(1346) || SUCCESS results-logistic-binary \n'
    '''
    phesdf_full[3].loc[phesdf_full[3]['4253'] == 3010, '4253'] = 0
    phesdf_full[3].loc[phesdf_full[3]['4253'] == 3011, '4253'] = 1
    # 403 is different encoding
    '''
    In [1033]: phesdf_full[2][mismatch[5]].value_counts()
    Out[1033]:
    0.4    1106
    0.6      69
    Name: 403, dtype: int64

    In [1034]: phesdf_full[3][mismatch[5]].value_counts()
    Out[1034]:
    1.0    1239
    2.0     102
    0.0      57
    Name: 403, dtype: int64

    In [1035]: phes_desc_df_comb.loc[mismatch[5]]['Full_Desc_v3']
    Out[1035]: '403_3|| INTEGER || CONTINUOUS || >20% IN ONE CATEGORY || Bin 1: <0.6666667, bin 2: ==0.6666667, bin 3: >0.6666667 || cat N: 57, 1239, 102 || CAT-ORD || order: 0|1|2 || num categories: 3 || SUCCESS results-ordered-logistic\n'

    In [1036]: phes_desc_df_comb.loc[mismatch[5]]['Full_Desc_v2']
    Out[1036]: '403_2|| INTEGER || Inc(>=10): 0.4(1106) || Inc(>=10): 0.6(69) || Removed 0.2: 5<10 examples || Removed 0: 4<10 examples || Removed 0.8: 5<10 examples || Removed 1: 2<10 examples || Removed 2.4: 1<10 examples || Removed 18.8: 1<10 examples || sample 1106/69(1175) || SUCCESS results-logistic-binary \n'
    '''

    phesdf_full[2].loc[phesdf_full[2]['403'] == 0.4, '403'] = 1
    phesdf_full[2].loc[phesdf_full[2]['403'] == 0.6, '403'] = 2

    # 398 is differently encoded
    # I don't understand how the encodings work
    # since original data has no 5.66; but the outcome has ~600 people in that category
    # So im going to just remove it
    # this is Number of correct matches in round
    phesdf_full[2] = phesdf_full[2].drop( [i for i in phesdf_full[2] if i.startswith('398')], axis =1)
    phesdf_full[3] = phesdf_full[3].drop( [i for i in phesdf_full[3] if i.startswith('398')], axis =1)

    '''
    In [1045]: phesdf_full[2][mismatch[6]].value_counts()
    Out[1045]:
    8.0    508
    0.0    264
    Name: 398, dtype: int64

    In [1046]: phesdf_full[3][mismatch[6]].value_counts()
    Out[1046]:
    5.666667    585
    4.500000    453
    3.000000    291
    0.000000     42
    3.500000     10
    Name: 398, dtype: int64

    In [1047]: phes_desc_df_comb.loc[mismatch[6]]['Full_Desc_v2']
    Out[1047]: '398_2|| INTEGER || Inc(>=10): 8(508) || Inc(>=10): 0(264) || sample 264/508(772) || SUCCESS results-logistic-binary \n'

    In [1048]: phes_desc_df_comb.loc[mismatch[6]]['Full_Desc_v3']
    Out[1048]: '398_3|| INTEGER || Inc(>=10): 3(291) || Inc(>=10): 5.66666666666667(585) || Inc(>=10): 4.5(453) || Inc(>=10): 0(42) || Removed 0.333333333333333: 1<10 examples || Removed 2.33333333333333: 5<10 examples || Inc(>=10): 3.5(10) || Removed 4.66666666666667: 2<10 examples || Removed 5: 4<10 examples || Removed 3.33333333333333: 1<10 examples || Removed 0.666666666666667: 1<10 examples || Removed 1.5: 1<10 examples || Removed 4.33333333333333: 1<10 examples || Removed 2: 1<10 examples || 3-20 values || CAT-ORD || order: 0|3|3.5|4.5|5.66666666666667 || num categories: 5 || SUCCESS results-ordered-logistic\n'
    '''
    # 2877 is categorical in one and not in the other
    # but its a little strange in terms of the processing
    # It is types of tobacco smoked, but its unclear if theres a non potion
    # so im just going to drop it
    phesdf_full[2] = phesdf_full[2].drop( [i for i in phesdf_full[2] if i.startswith('2877')], axis =1)
    phesdf_full[3] = phesdf_full[3].drop( [i for i in phesdf_full[3] if i.startswith('2877')], axis =1)

'''
In [187]: phesdf_full[2].shape
Out[187]: (1425, 404)

In [188]: phesdf_full[3].shape
Out[188]: (1425, 388)
'''

CLEAN_DIFF_ORD = True
if CLEAN_DIFF_ORD:
    # FInd things that are processed differently
    phes_desc_df_comb = phes_desc_df[2].merge(phes_desc_df[3], right_index=True, left_index=True, suffixes=['_v2', '_v3'])
    # Remove double nans
    weird_col_mask = phes_desc_df_comb[['order_v2', 'order_v3']].isna().sum(1) != 2
    weird_col_mini = phes_desc_df_comb['order_v2'][weird_col_mask] != phes_desc_df_comb['order_v3'][weird_col_mask]
    mismatch_order = phes_desc_df_comb[weird_col_mask][weird_col_mini][['order_v2', 'order_v3']].index
    mismatch_order = [b for b in mismatch_order if b not in mismatch]
    '''
    In [229]: len([b for b in mismatch_order if b not in mismatch])
    Out[229]: 23
    '''
    phes_desc_df_comb.loc[mismatch_order][['order_v2', 'order_v3']]

    # Most of these are ordered
    # and these columns come up bc one group has the value come up
    # and the other doesn't
    mismatch_true =  ['2867', '2887']
    '''
    In [238]:  phes_desc_df_comb.loc[mismatch_true[0]]['Full_Desc_v2']
    Out[238]: '2867_2|| INTEGER || reassignments: -1=NA|-3=NA || CONTINUOUS || >20% IN ONE CATEGORY || Bin 1: <16, bin 2: >=16AND < 18, bin 3: >=18 || cat N: 81, 92, 123 || CAT-ORD || order: 0|1|2 || num categories: 3 || SUCCESS results-ordered-logistic\n'

    In [239]:  phes_desc_df_comb.loc[mismatch_true[0]]['Full_Desc_v3']
    Out[239]: '2867_3|| INTEGER || reassignments: -1=NA|-3=NA || Inc(>=10): 19(20) || Inc(>=10): 16(48) || Inc(>=10): 14(24) || Inc(>=10): 21(18) || Removed 22: 7<10 examples || Inc(>=10): 15(34) || Removed 13: 9<10 examples || Inc(>=10): 17(36) || Removed 0: 3<10 examples || Inc(>=10): 20(18) || Inc(>=10): 18(53) || Removed 12: 9<10 examples || Removed 23: 2<10 examples || Removed 10: 4<10 examples || Removed 25: 5<10 examples || Removed 26: 1<10 examples || Removed 11: 3<10 examples || Removed 9: 1<10 examples || Removed 30: 1<10 examples || 3-20 values || CAT-ORD || order: 14|15|16|17|18|19|20|21 || num categories: 8 || SUCCESS results-ordered-logistic\n'
    '''
    # For 2867 visit 3 is weird
    # so use raw information and pipe it through visit 2 process
    curr_data = newdf_v2i_idx[translate_phesant(mismatch_true[0], 3)].copy()
    if (curr_data < 0).sum() > 0:
        print('Deal with Nan recoding')
    curr_data[curr_data < 16] = 0
    curr_data[curr_data >= 18] = 2
    curr_data[(curr_data >= 16) & (curr_data < 18)] = 1
    phesdf_full[3][mismatch_true[0]] = curr_data

    '''
    In [267]: phes_desc_df_comb.loc[mismatch_true[1]]['Full_Desc_v3']
    Out[267]: '2887_3|| INTEGER || reassignments: -1=NA|-10=NA || Inc(>=10): 40(12) || Removed 6: 3<10 examples || Inc(>=10): 15(43) || Inc(>=10): 20(115) || Removed 8: 9<10 examples || Inc(>=10): 10(50) || Inc(>=10): 30(15) || Removed 12: 7<10 examples || Removed 25: 6<10 examples || Inc(>=10): 5(12) || Removed 3: 3<10 examples || Removed 60: 2<10 examples || Removed 14: 2<10 examples || Removed 0: 5<10 examples || Removed 7: 1<10 examples || Removed 2: 1<10 examples || Removed 50: 1<10 examples || Removed 18: 1<10 examples || 3-20 values || CAT-ORD || order: 5|10|15|20|30|40 || num categories: 6 || SUCCESS results-ordered-logistic\n'

    In [268]: phes_desc_df_comb.loc[mismatch_true[1]]['Full_Desc_v2']
    Out[268]: '2887_2|| INTEGER || reassignments: -1=NA|-10=NA || CONTINUOUS || >20% IN ONE CATEGORY || Bin 1: <15, bin 2: >=15AND < 20, bin 3: >=20 || cat N: 87, 52, 143 || CAT-ORD || order: 0|1|2 || num categories: 3 || SUCCESS results-ordered-logistic\n'
'''
    # For 2667 visit 3 is weird
    # so use raw information and pipe it through visit 2 process
    curr_data = newdf_v2i_idx[translate_phesant(mismatch_true[1], 3)].copy()
    if (curr_data < 0).sum() > 0:
        print('Deal with Nan recoding')
    curr_data[curr_data < 15] = 0
    curr_data[curr_data >= 20] = 2
    curr_data[(curr_data >= 15) & (curr_data < 20)] = 1
    phesdf_full[3][mismatch_true[1]] = curr_data

# SAVE
POST_PHES_FOLDER = 'post_Phesant'
POST_PHESANT_FILE = lambda v : f"ukbb_post_phes_vi{v}.csv"
phesdf_full[2].to_csv(os.path.join(OUT_FOLDER, POST_PHES_FOLDER, POST_PHESANT_FILE(2)), index = False)
phesdf_full[3].to_csv(os.path.join(OUT_FOLDER, POST_PHES_FOLDER, POST_PHESANT_FILE(3)), index = False)


###########################################################################################
# Step 3b
# Subtraction + Standard Scaling
# ----------------------------------------
#
# Subtraction cannot be done if one of the visits has a NAN value
# which we check for after subtraction
###########################################################################################

# Subtraction
# We cant use the pandas subtract method
# Since it loses nan values
# phn_change_test = phesdf_v2.sub(phesdf_v1)
phn_change = {}
col_lost = []
cont_col_change = []
cont_col_change_stats = {}
all_col = list(set([*phesdf_full[3].columns.values, *phesdf_full[2].columns.values]))
for c in all_col:
    if c in phesdf_full[2].columns and c in phesdf_full[3].columns:
        if c == '709.x':
            newc = '709'
        else:
            newc = c
        # Change type to float or else binary subtraction outputs 255
        phn_change[newc] = phesdf_full[3][c].astype('float') - phesdf_full[2][c].astype('float')
        if c in cont_col[2] or c in cont_col[3]:
            # Track continuous columns
            # so we can standard scale them later
            cont_col_change.append(newc)
            cont_col_change_stats[newc] = [phn_change[c].mean(), phn_change[c].std()]
    elif 'x' in c:
        # Unprocessed data from before phesant
        # Since phesant applied some unknown processing
        cc = c.split('_')[0].split('x')[1] # Readable name
        # We are assured thath the '_' does not come from cat bc dummy hasn't been applied to this data
        if cc not in phn_change:
            # this col has not been seen before
            if c.replace('_2_', '_3_') in phesdf_full[3] and c.replace('_3_', '_2_') in phesdf_full[2]:
                phn_change[cc] = phesdf_full[3][c.replace('_2_', '_3_')] - phesdf_full[2][c.replace('_3_', '_2_')]
                if c in cont_col[2] or c in cont_col[3]:
                    # Track continuous columns
                    # so we can standard scale them later
                    cont_col_change.append(cc)
                    cont_col_change_stats[cc] = [phn_change[cc].mean(), phn_change[cc].std()]
            elif '_' in c or '#' in c:
                # Theoritically this code block shouldn't trigger
                # But it did for blood pressure related variables.
                # so we will skip any output
                print("This shouldn't hit: ", c)
                col_lost.append(c)
    elif '_' in c or '#' in c:
        # These are categorical
        # So we can assume that the unanswered variables are all zero
        if c in phesdf_full[3]:
            phn_change[c] = phesdf_full[3][c]
        elif c in phesdf_full[2]:
            phn_change[c] = -1 * phesdf_full[2][c]
    else:
        col_lost.append(c)
        print(c)
phn_change = pd.DataFrame.from_dict(phn_change)

y_desc_dict['21004-0.0'] = "Number of puzzles correct"
y_cat_dict.loc['21004-0.0'] = [32, 'cognitive phenotypes']
for col in col_lost:
    if 'x' in col:
        col = col.split('_')[0].split('x')[1]
    print(col, ' : ', y_desc_dict[man_plot.translate(col, y_cat_dict)[0]])

phes_v1_not_v2 = [i for i in phesdf_full[2] if i not in [a.replace('_3_', '_2_') for a in phesdf_full[3]]]
phes_v2_not_v1 = [i for i in phesdf_full[3] if i not in [a.replace('_2_', '_3_') for a in phesdf_full[2]]]

print("In original visit but not second")
for col in phes_v1_not_v2:
    if 'x' in col:
        col = col.split('_')[0].split('x')[1]
    print(col, ' : ', y_desc_dict[man_plot.translate(col, y_cat_dict)[0]])

print("\n\nIn follow-up visit but not original")
for col in phes_v2_not_v1:
    if 'x' in col:
        col = col.split('_')[0].split('x')[1]
    print(col, ' : ', y_desc_dict[man_plot.translate(col, y_cat_dict)[0]])

[ii for ii in y_cat_dict.index if ii.startswith(col+'-')]
# Check that number of nans is perseved before and after subtraction
# If either of the columns had a nan value for a given row;
# the output of subtraction should also be nan
for c in phn_change:
    if c in phesdf_full[2] and c in phesdf_full[3]:
        a = np.logical_or(phesdf_full[2][c].isna(), phesdf_full[3][c].isna()).sum()
        b = phn_change[c].isna().sum()
        assert a == b, f'Fails at column {c}; orig is {a} nans and subtraction results in {b} nans'

# TODO SAVE
df_cont_col_stats = pd.DataFrame.from_dict(cont_col_change_stats, orient = 'index', columns = ['Mean', 'STD'])
df_cont_col_stats.to_csv(os.path.join(OUT_FOLDER, POST_PHES_FOLDER, 'ukbb_y_2vis_continuous_stats.csv'), index = False)
phn_change.to_csv(os.path.join(OUT_FOLDER, POST_PHES_FOLDER, 'ukbb_y_2vis_subtract_noSS.csv'), index = False)

###########################################################################################
# Step 3c
# Final processing
# ----------------------------------------
# Imputation on final subtracted dataframe
#
# Data Reduction:
#   Remove columns with only NAN values
#   Remove columns with only one response given
#
# phn_change_reduced is the final dataframe
###########################################################################################

# Standard Scaling
from sklearn.preprocessing import StandardScaler
phn_change_SS = phn_change.copy()
change_SS = StandardScaler()
phn_change_SS[cont_col_change] = change_SS.fit_transform(phn_change_SS[cont_col_change].values)
phn_change_SS.to_csv(os.path.join(OUT_FOLDER, POST_PHES_FOLDER, 'ukbb_y_2vis_subtract_w_SS.csv'), index = False)

import joblib
joblib.dump(change_SS, os.path.join(OUT_FOLDER, POST_PHES_FOLDER, 'SS_ukbb_y_2vis_subtract.bin'), compress=True)


# TODO SAVE data
# TODO Save SS object and our own SS Stats
def impute(X, seed = 0):
    np.random.seed(seed)
    X = np.asarray(X)
    # non-parametric single-column imputation
    for i_col in range(X.shape[1]):
        vals = X[:, i_col] #pick off column
        vals_set = vals[~np.isnan(vals)] #get all values that aren't nans
        inds_nan = np.where(np.isnan(vals))[0] #get idxs of nan rows
        n_misses = len(inds_nan)
        if len(vals_set) > 0:
            inds_repl = np.random.randint(0, len(vals_set), n_misses)
            vals[inds_nan] = vals_set[inds_repl]
            assert np.all(np.isfinite(vals))
            X[:, i_col] = vals
    return X

# Impute data
phn_change_filled = pd.DataFrame(impute(phn_change_SS.values), columns = phn_change_SS.columns, index = phn_change_SS.index)
# Check/remove fully nan columns
# Also remove columns where > NAN_THRESHOLD of individuals don't have data
NAN_THRESHOLD = 0.9
N_DOUBLES = phn_change_filled.shape[0]
d = (phn_change_filled.isna().sum() == N_DOUBLES).values
e = (phn_change.isna().sum() == N_DOUBLES).values
f = (phn_change.isna().sum() > int(NAN_THRESHOLD * N_DOUBLES)).values
phn_change_filled = phn_change_filled.drop(columns = phn_change_filled.columns[f])

# Keep only columns with at least two distinct possible values
col_keep = [col for col in phn_change_filled if len(np.unique(phn_change_filled[col])) > 1]

phn_change_reduced = phn_change_filled[col_keep]
phn_change_reduced.to_csv(os.path.join(OUT_FOLDER, POST_PHES_FOLDER, 'ukbb_y_2vis_subtract_final.csv'))


###########################################################################################
# Step 4
# Quality Control
# ----------------------------------------
# Print out all names
# and manually see that they look right
# I.e. they don't look like something that is static
###########################################################################################

import processing.manhattan_plot_util as man_plot
# load in 977 phenotypes
ukbb_y, y_desc_dict, y_cat_dict = man_plot.load_phenom(BASE_FOLDER = 'data/processed/')

for ii, col in enumerate(phn_change_reduced):
    if col != 'userID' and col != '3089':
        if 'x' in col:
            col2 = col
            col = col2.split('_')[0].split('x')[1]
        print(ii, col, ' : ', y_desc_dict[man_plot.translate(col, y_cat_dict)[0]])

for col in blah:
    if col != 'userID' and col != '3089':
        print(col, ' : ', y_desc_dict[man_plot.translate(col, y_cat_dict)[0]])

# Weird Columnd
# 2139: Age at first intercourse: this goes up/down a couple years
phes_desc_df_comb.loc[mismatch_true[0]]['Full_Desc_v3']
phes_desc_df_comb.loc['4196']['Full_Desc_v3']