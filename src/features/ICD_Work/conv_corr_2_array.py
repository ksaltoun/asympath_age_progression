import os
import numpy as np
import glob
import joblib
import pandas as pd
from tqdm import tqdm
import re


# TAR_DIR = '/Users/danilo/f/UKBB/access_40k_UKB_holmes/fMRI/rfMRI_corr_matrix_dim100'
# TAR_DIR = '/Users/danilo/f/UKBB/access_40k_UKB_holmes/fMRI/rfMRI_partialcorr_matrix_dim100'
TAR_DIR = 'rfMRI_corr_matrix_dim25'
TAR_DIR = 'rfMRI_partialcorr_matrix_dim25'
TAR_DIR = 'rfMRI_corr_matrix_dim100'
TAR_DIR = 'rfMRI_partialcorr_matrix_dim100'

# KARIN

CURR_PATH = '/Users/karinsaltoun/dblabs/asymm_pat_age_progression'
os.chdir(CURR_PATH)
TAR_FOLDER = 'data/raw/resting-state-fMRI'


files = glob.glob(TAR_FOLDER + '/'+ TAR_DIR + '/*.txt')

list_subnames = []
list_corrs = []
n_skips = 0

ncor = 210 if '25' in re.findall(r'\d+', TAR_DIR) else 1485

for cur_f in tqdm(files):
    sub = np.int(os.path.basename(cur_f).split('_')[0])
    # print(sub)

    try:
        sub_floats = np.loadtxt(cur_f)
    except:
        f = open(cur_f, 'r')
        cont = f.readlines()

        flt_nums = cont[0].split(' ')

        sub_floats = []
        for num in flt_nums:
            if len(num) > 0:
                sub_floats.append(np.float(num))
        sub_floats = np.array(sub_floats)


    # print(len(sub_floats))
    # assert len(sub_floats) == 1485
    if len(sub_floats) != ncor:
        n_skips += 1
        continue

    list_subnames.append(sub)
    list_corrs.append(sub_floats)

array_subnames = np.array(list_subnames)
array_corrs = np.array(list_corrs)

print('%i skippped items !' % n_skips)

joblib.dump([array_subnames, array_corrs], TAR_FOLDER + '/dumps/dump_' + TAR_DIR, compress=9)
