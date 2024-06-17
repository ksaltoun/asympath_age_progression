###########################################################################################
# Part 1
# Funpack
# Separately take out ICD 9 and ICD 10 stuff
###########################################################################################



# I may be missing import statements, so add if you find something I missed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Good news, the full dataset has the eid as the first row!!

cicd9 = [40013, 41271, 41203, 41205]

cicd10 = [41270, 41202, 41204, 40006, 40001, 40002]

# Redo with external cause of death? 41201

DATA_FOLDER = "/Users/karinsaltoun/dblabs/asymm_pat_age_progression"
OUT_FOLDER = "data/interim/icd10_work/full_data"
ICDMAP_FOLD = 'data/external/ICD9_10_Mapping'

OUT_FILE = lambda x: f"ukbb_all_icd{x}.csv"
OUT_PROC_FILE = lambda x: f"ukbb_all_proc_icd{x}.csv"


# Run the convert function
# Pick out the rows of interest
# And process them with funpack


FUNPACK_CMD =  lambda vi, v_num: f"""
funpack -wl -n -n -n -wde -ws -ow \
-cfg fmrib \
{''.join([f" -v {v} " for v in vi])}\
{''.join([f' -cl {v} "codeToNumeric('+f"'icd{v_num}'"+')"' for v in vi])} \
-imf icd{v_num}_codes.tsv  \
-pf /Users/karinsaltoun/Documents/Funpack/ksconfig/ksmri/processing.tsv \
{os.path.join(DATA_FOLDER, OUT_FOLDER, OUT_PROC_FILE(v_num))} \
/Users/karinsaltoun/Documents/ukb40501.csv
"""

# {os.path.join(DATA_FOLDER, OUT_FOLDER, OUT_FILE(v_num))}

print(FUNPACK_CMD(cicd9, 9))
print(FUNPACK_CMD(cicd10, 10))

# Funpack complete
###########################################################################################
# Part 2
# Interpret Funpack Codes
# Separately take out ICD 9 and ICD 10 stuff
###########################################################################################



ukbb_large_icd9_full = pd.read_csv(os.path.join(DATA_FOLDER, OUT_FOLDER, OUT_PROC_FILE(9)))
ukbb_large_icd10_full = pd.read_csv(os.path.join(DATA_FOLDER, OUT_FOLDER, OUT_PROC_FILE(10)))
print('\a')

'''
In [628]: ukbb_large_icd10.shape
Out[628]: (5, 14217)

In [629]: ukbb_large_icd9.shape
Out[629]: (5, 2837)

In [55]: ukbb_large_icd9_full.shape
Out[55]: (502507, 2837) # includes eid column
'''

ukbb_large_icd9 = pd.read_csv(os.path.join(DATA_FOLDER, OUT_FOLDER, OUT_PROC_FILE(9)), nrows=5)
ukbb_large_icd10 = pd.read_csv(os.path.join(DATA_FOLDER, OUT_FOLDER, OUT_PROC_FILE(10)), nrows=5)

desc_icd10_V2 = pd.read_csv(os.path.join(DATA_FOLDER, OUT_FOLDER, 'icd10_codes.tsv'), sep='\t')

desc_icd10 = pd.read_csv(os.path.join(DATA_FOLDER, OUT_FOLDER,
                                      OUT_PROC_FILE(10).split('.')[0]+'_description.txt'),
                         sep='\t', header=None)
desc_icd9 = pd.read_csv(os.path.join(DATA_FOLDER, OUT_FOLDER,
                                      OUT_PROC_FILE(9).split('.')[0]+'_description.txt'),
                         sep='\t', header=None)
desc_icd10.columns = ["col", 'desc']
desc_icd9.columns = ["col", 'desc']
# We really need to map the icd10/9 codes to whatever FUNPACK outputs

desc_icd10['fp_code'] = desc_icd10['col'].map(lambda x: int(x.split('.')[-1]))
desc_icd9['fp_code'] = desc_icd9['col'].map(lambda x: int(x.split('.')[-1]))
desc_icd9['icd9_code'] = desc_icd9['desc'].map(lambda x: str(x).split('(')[1].split(' ')[0])


# Before the icd10/9 as per the phecodes
# Make a reverse map
# Which associates a fp-code with all columns it calls
icd10dict = {k: [k.split('.')[-1]] for k in dficd10 if k !='eid'}

revdict10 = {}
for key, values in icd10dict.items():
     for value in values:
             revdict10.setdefault(value, []).append(key)

icd9dict = {k: [k.split('.')[-1]] for k in dficd9 if k !='eid'}

revdict9 = {}
for key, values in icd9dict.items():
     for value in values:
             revdict9.setdefault(value, []).append(key)


# Make a dictionary that maps
dis_list = list(set(np.concatenate([list(revdict10.keys()), list(revdict9.keys())])))


# FP code, ICD9 code, ICD10 Code, Description
fp2icd_dict = {dis:{'ICD10_Code': [],
                    'ICD9_Code': [],
                    'ICD10_Desc': [],
                    'ICD9_Desc': []} for dis in dis_list}

# Combine into one df with all fp translations
for dis in tqdm(dis_list):
    dis_int = int(dis)
    if int(dis_int) in desc_icd10_V2['value'].values:
        ic10_stuff = desc_icd10_V2[desc_icd10_V2['value'] == dis_int]
        if ic10_stuff.shape[0] > 1:
            # some have multiple hits
            print('\nDisease ', dis, f'{ic10_stuff.shape[0]} rows\n', ic10_stuff)

            # if len(set(ic9_list)) == 1:
            #     ic9 = ic9_list[0]
            # else:
            #     print('\nDisease ', dis, f'{ic9_stuff.shape[0]} rows\n', ic9_stuff)
        fp2icd_dict[dis]['ICD10_Code'] = ic10_stuff['code'].values[0]
        fp2icd_dict[dis]['ICD10_Desc'] = ic10_stuff['description'].values[0]
    else:
        fp2icd_dict[dis]['ICD10_Code'] = np.nan
        fp2icd_dict[dis]['ICD10_Desc'] = np.nan

    if int(dis_int) in desc_icd9['fp_code'].values:
        ic9_stuff = desc_icd9[desc_icd9['fp_code'] == int(dis_int)]
        if ic9_stuff.shape[0] > 1:
            # # some have multiple hits
            # ic9_list = []
            # for v in ic9_stuff[1].values:
            #     ic9_list.append(str(v).split('(')[1].split(' ')[0])
            # ic9 = ic9_stuff['icd9_code'].values[0]
            # if len(set(ic9_list)) > 1:
            if len(set(ic9_stuff['icd9_code'].values)) > 1:

                print('\nDisease ', dis, f'{ic9_stuff.shape[0]} rows\n', ic9_stuff)

        # else:
            # ic9 = str(ic9_stuff[1]).split('(')[1].split(' ')[0]
        ic9 = ic9_stuff['icd9_code'].values[0]
        fp2icd_dict[dis]['ICD9_Code'] = ic9
        fp2icd_dict[dis]['ICD9_Desc'] = ic9_stuff['desc'].values[0]
    else:
        fp2icd_dict[dis]['ICD9_Code'] = np.nan
        fp2icd_dict[dis]['ICD9_Desc'] = np.nan


fp2icd = pd.DataFrame.from_dict(  fp2icd_dict  ).T



phecodes = pd.read_csv(os.path.join(ICDMAP_FOLD, 'UKB_Phecode_v1-2b1_ICD_Mapping.txt'), sep='\t')
phecodes_idx = phecodes.set_index('phecode')

dficd09eid = ukbb_large_icd9_full.set_index('eid').sort_index()
dficd10eid = ukbb_large_icd10_full.set_index('eid').sort_index()
print('\a')

N_PPL = len(ukbb_large_icd9_full)
eids = ukbb_large_icd9_full['eid']
# Make dict that maps phecode to
# columns from dficd9/dficd10 you need to grab
ppl_phecodes = pd.DataFrame(np.zeros(N_PPL), index = eids)

fp10codenf = [] # code not found
fp9codenf = []
nohits = []

# This will keep track of which diseases were used
# So I can see if all the ones we have in inclusions include all the diseases we have in input
fp9used = []
fp10used = []

# there will be some fp 10 codes that are not found
# bc they are output from the map funpack made
# but not actually in the dataframe
ppl_phecodes_dict = {'eid': eids.values}

# will be 1750 interations
for pc in tqdm(phecodes_idx.T):
    pin = phecodes_idx.loc[pc]['icd_codes'].split(',')
    pex = phecodes_idx.loc[pc]['exclude_phecodes'].split(',')

    in_series = pd.Series(np.zeros(N_PPL), index = eids, name=pc)

    # Look through all the inclusion codes
    # And find if they exist inside the dficd10 file

    # Go through ICD 10
    for pcode in pin:
        if pcode not in fp2icd['ICD10_Code'].values:
            # not here, go to the next
            continue
        # We have this pcode, therefore find its fp name
        fp10code = fp2icd[fp2icd['ICD10_Code'] == pcode].index.values
        if len(fp10code) == 1:
            fp10code = fp10code[0]
        else:
            print(f'{pcode} maps to following fpcodes: {fp10code}')
            print('Exiting codeblock')
            continue

        if fp10code not in revdict10.keys():
            # not here, go to the next
            # print(f'fpcodes {fp10code} not in dataset')
            # print('Exiting codeblock')
            fp10codenf.append(fp10code)
            continue
        # Keep track of codes used
        fp10used.append(fp10code)

        # Add current columns to existing columns
        for col in revdict10[fp10code]:
             in_series = in_series.add(dficd10eid[col], fill_value=0)

    # Go through ICD 9
    for pcode in pin:
        if pcode not in fp2icd['ICD9_Code'].values:
            # not here, go to the next
            continue
        # We have this pcode, therefore find its fp name
        fp9code = fp2icd[fp2icd['ICD9_Code'] == pcode].index.values
        if len(fp9code) == 1:
            fp9code = fp9code[0]
        else:
            print(f'{pcode} maps to following fpcodes: {fp9code}')
            print('Exiting codeblock')
            continue

        if fp9code not in revdict9.keys():
            # # not here, go to the next
            # print(f'fpcodes {fp9code} not in dataset')
            # print('Exiting codeblock')
            fp9codenf.append(fp9code)
            continue
        # Keep track of codes used
        fp9used.append(fp9code)

        # Add current columns to existing columns
        for col in revdict9[fp9code]:
             in_series = in_series.add(dficd09eid[col], fill_value=0)

    if in_series.sum()  == 0:
        # no ppl associated with this disease showed up
        nohits.append(pc)
        continue
    ppl_phecodes_dict[pc] = (in_series>0).astype(int).copy().values

# drop the column we put in to initiate the data
ppl_phecodes = pd.DataFrame.from_dict(ppl_phecodes_dict)
ppl_phecodes = ppl_phecodes.set_index('eid').sort_index()

print('\a')

# Work on making the exclusions file
# Will also need to adapt man_plot_util to make a manplot equiv

ppl_xcldphes_dict = {'eid': eids.values}

for pc in tqdm(ppl_phecodes.columns):
    pex = phecodes_idx.loc[pc]['exclude_phecodes'].split(',')
    pex_flt = [float(ii) for ii in pex]
    # Many, if not all exclusions involve the code of interest itself
    # I don't think it makes sense to keep that
    # rather, I want to keep everything to excude that is not itself
    # so when we do a phewas I will first remove all the exclusion ppl from the group
    # pex_flt.remove(pc)

    # We now have an exclusion list
    ot_series = pd.Series(np.zeros(N_PPL), index = eids, name=pc)

    if len(pex_flt) == 0:
        # No exclusions
        # Push to the main dataframe and go to next phecode
        ppl_xcldphes_dict[pc] = ot_series.copy()
        continue

    for px in pex_flt:
        if px not in ppl_phecodes.columns.values:
            # No one has this exclusion criteria
            continue
        ot_series = ot_series.add(ppl_phecodes[px], fill_value=0)

    # We've gone through all exclusion
    # Potentionally someone may have more than one exclusion
    # So map all things which are bigger than 0 to 1
    ot_series = (ot_series>0).astype(int)

    # Update, some of the exclusions are a big umbrella in which the current column is a part
    # so first subtract the ppl in the in group
    ot_series = ot_series.subtract(ppl_phecodes[pc], fill_value=0)

    ppl_xcldphes_dict[pc] = ot_series.astype(int).copy().values

# Turn into a dataframe
ppl_xcldphe = pd.DataFrame.from_dict(ppl_xcldphes_dict)
ppl_xcldphe = ppl_xcldphe.set_index('eid').sort_index()
print('\a')

# Save
# os.path.join(DATA_FOLDER, OUT_FOLDER, OUT_PROC_FILE(v_num)
ppl_xcldphe.to_csv(os.path.join(DATA_FOLDER, OUT_FOLDER, 'exclusions_full.csv'))
ppl_phecodes.to_csv(os.path.join(DATA_FOLDER, OUT_FOLDER, 'inclusions_full.csv'))
print('\a')

# Aside:
# Plot the frequency of things

counts = ppl_phecodes.sum(0).copy()
freq = pd.DataFrame(100*counts/N_PPL, columns=['-logp_Freq'])
freq.index.name = 'phecode'
freq['p_Freq'] = ppl_xcldphe.sum(0).copy()
freq['r_Freq'] = counts

import processing.ks_1_0_MeDiWAS_fncs as MeDiWAS

groupid, _ = MeDiWAS.default_order()

freq['group'] = phecodes_idx.loc[counts.index.values]['group']
freq['desc'] = phecodes_idx.loc[counts.index.values]['description']
freq['groupid'] = freq['group'].map(groupid)

freq = freq.reset_index().set_index(['groupid', 'phecode']).sort_index()
freq['i'] = np.arange(len(freq))

plot = MeDiWAS.manhattan_plot(freq, 'Freq', label_thres=5)
plot.ax.set_ylabel(f'% Frequency across {N_PPL:,} individuals')

MeDiWAS.hits_csv(freq, 'Freq', 'reports/ICD_Work', sort=True, thresBon=10**-0.2, useFDR=False)



print('Loading Phenotypes')
DATA_FOLD = 'data/interim/icd10_work'
ICDMAP_FOLD = 'data/external/ICD9_10_Mapping'

# Load in
ppl_xcldphe_small = pd.read_csv(os.path.join(DATA_FOLD,'proc', 'exclusion.csv'), index_col = 0)
ppl_phecodes_small = pd.read_csv(os.path.join(DATA_FOLD,'proc', 'inclusion.csv'), index_col = 0)
N_PPL_SMALL = len(ppl_phecodes_small)

counts_small = ppl_phecodes_small.sum(0).copy()
freq_small = pd.DataFrame(100*counts_small/N_PPL_SMALL, columns=['-logp_Freq'])
freq_small.index.name = 'phecode'
freq_small.index = freq_small.index.astype(float)

freq_small['p_Freq'] = ppl_xcldphe_small.sum(0).copy()
freq_small['r_Freq'] = counts_small


groupid, _ = MeDiWAS.default_order()

freq_small['group'] = phecodes_idx.loc[counts_small.index.values.astype(float)]['group']
freq_small['desc'] = phecodes_idx.loc[counts_small.index.values.astype(float)]['description']
freq_small['groupid'] = freq_small['group'].map(groupid)

freq_small = freq_small.reset_index().set_index(['groupid', 'phecode']).sort_index()
freq_small['i'] = np.arange(len(freq_small))

plot = MeDiWAS.manhattan_plot(freq_small, 'Freq', label_thres=10, thres=None)
plot.ax.set_ylabel(f'% Frequency across {N_PPL_SMALL:,} individuals')
