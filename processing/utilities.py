
import numpy as np
import nibabel as nib
from nilearn import datasets as ds
import pandas as pd
import os, re

HO_atlas_cort = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm', symmetric_split=True)
HO_atlas_sub = ds.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm', symmetric_split=True)

#Load in stored atlases
CURR_FOLDER = 'data/external'
JHU_nii = nib.load(os.path.join(CURR_FOLDER, 'JHU-ICBM-labels-1mm.nii'))
cereb_lut = pd.read_csv(os.path.join(CURR_FOLDER, 'Cereb_atlasesMNI', 'Lobules-SUIT.nii.lut'), 
    sep = '\s+', names=['id', 'v1', 'v2','v3','name'])
cereb_nii = nib.load(os.path.join(CURR_FOLDER, 'Cereb_atlasesMNI', 'Lobules-SUIT.nii'))


TRACT_DICT = {
        # mean variants (TBSS, 48 tracts, JHU atlas)
        'middle cerebellar peduncle': 1,
        'pontine crossing tract': 2,
        'genu of corpus callosum': 3,
        'body of corpus callosum': 4,
        'splenium of corpus callosum': 5,
        'fornix': 6,
        'Right corticospinal tract': 7,
        'Left corticospinal tract': 8,
        'Right medial lemniscus': 9,
        'Left medial lemniscus': 10,
        'Right inferior cerebellar peduncle': 11,
        'Left inferior cerebellar peduncle': 12,
        'Right superior cerebellar peduncle': 13,
        'Left superior cerebellar peduncle': 14,
        'Right cerebral peduncle': 15,
        'Left cerebral peduncle': 16,
        'Right anterior limb of internal capsule': 17,
        'Left anterior limb of internal capsule': 18,
        'Right posterior limb of internal capsule': 19,
        'Left posterior limb of internal capsule': 20,
        'Right retrolenticular part of internal capsule': 21,
        'Left retrolenticular part of internal capsule': 22,
        'Right anterior corona radiata': 23,
        'Left anterior corona radiata': 24,
        'Right superior corona radiata': 25,
        'Left superior corona radiata': 26,
        'Right posterior corona radiata': 27,
        'Left posterior corona radiata': 28,
        'Right posterior thalamic radiation': 29,
        'Left posterior thalamic radiation': 30,
        'Right sagittal stratum': 31,
        'Left sagittal stratum': 32,
        'Right external capsule': 33,
        'Left external capsule': 34,
        'Right cingulum cingulate gyrus': 35,
        'Left cingulum cingulate gyrus': 36,
        'Right cingulum hippocampus': 37,
        'Left cingulum hippocampus': 38,
        'Right fornix cres+stria terminalis': 39,
        'Left fornix cres+stria terminalis': 40,
        'Right superior longitudinal fasciculus': 41,
        'Left superior longitudinal fasciculus': 42,
        'Right superior fronto-occipital fasciculus': 43,
        'Left superior fronto-occipital fasciculus': 44,
        'Right uncinate fasciculus': 45,
        'Left uncinate fasciculus': 46,
        'Right tapetum': 47,
        'Left tapetum': 48}
DMRI_TYPES = ['FA', 'MD', 'MO', 'L1', 'L2', 'L3', 'ICVF', 'OD', 'ISOVF']


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
#--------

def remove_rows(df, rows_2_rem):
    rows_2_rem = list(set(rows_2_rem))
    bad_df = df.index.isin(rows_2_rem)
    df = df[~bad_df]
    return(df)


def parse_name(curr_feat, expt_type_abbr, side = None):
    #given cur feat name and expt type return name
    cur_feat_name = curr_feat.split(' (')[0]
    if 'Stem' in cur_feat_name:
        pass
    elif expt_type_abbr == "Regular":
        #ASK: won't this inaccurately add Left to age, age*sex, and IQ which we added in outselves?
        if 'right' not in curr_feat and 'left' not in curr_feat:
            pass
        else:
            cur_feat_name = ('Right ' if 'right' in curr_feat else 'Left ') + cur_feat_name
    else:
        #there isn't a right/left in other representations, so we're just going with right
        side = 'Right ' if side > 0 else 'Left '
        cur_feat_name = side + cur_feat_name 
    
    # HACK
    if 'Ventral Striatum' in cur_feat_name:
        cur_feat_name = cur_feat_name.replace('Ventral Striatum', 'Accumbens')
    
    return(cur_feat_name)


def create_nii(coefs, area_names, expt_type_abbr, z_return = False, coefs_z = None):
    #recieves a dataframe of 55 X n 
    #returns an array of nii files, with n returned files

    #if we want z nii images return, double check everything is fine
    if z_return == True:
        if coefs_z == None: 
            raise NameError("Asked to return z but coefs_z not provided")
        if coefs_z.shape != coefs.shape:
            raise ValueError("coefs and coefs_z are not the same size")
    
    #change coefs into numpy arrays
    print(expt_type_abbr)
    coefs = np.asarray(coefs)
    if z_return == True: coefs_z = np.asarray(coefs_z)

    #if 1-d we need to reshape data
    coefs = coefs.reshape(1, -1) if len(coefs.shape) == 1 else coefs
    if z_return: coefs_z = coefs_z.reshape(1, -1) if len(coefs_z.shape) == 1 else coefs_z

    #make sure our col names matches the length of our coefs
    if coefs.shape[-1] != len(area_names):
        raise ValueError("Size of coef array and number of regions given do not match")

    nii_array = []
    nii_array_z = []
    print(coefs.shape[-1])
    for ii in range(coefs.shape[0]):
        # print(f"CURRENT Analysis: {lbl}")
        #ASK: why don't we reset the two masks? Aren't we just doing a whole lot of aggregates by the end, esp if we have dump significant only to be true? Or is that not the case since we are looping through the whole region space for each SES, so every round we're over writing the previous loop and we already know that each round has the same info [i.e. locations]
        SES_in_brain_data = np.zeros((HO_atlas_cort.maps.shape))
        SES_in_brain_data_z = np.zeros((HO_atlas_cort.maps.shape))
        for i_feat in range(len(area_names)): #across every brain region/input param (since we have age and sex here too)
            #Parse the name of the brain region 
            cur_feat_name = parse_name(area_names[i_feat], expt_type_abbr, side = coefs[ii, i_feat])

            b_found_roi = False
            #here we search for out labels in two sets of databases
            #we already checked that these atlases don't have any overlap in areas they map
            #we then take the region that the atlas marks and assign the coeffs and stdev to our map of the brain
            #which shows the relationship btwn this SES and the contribution of this brain region
            for i_cort_label, cort_label in enumerate(HO_atlas_cort.labels):
                if cur_feat_name in cort_label:
                                #find the mask
                    b_roi_mask = HO_atlas_cort.maps.get_fdata() == i_cort_label
                    n_roi_vox = np.sum(b_roi_mask)
                    print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))
                    if n_roi_vox == 0: print("EMPTY AREA: (cortical)", cort_label)
                    # print(type(n_roi_vox), n_roi_vox, n_roi_vox == 0)

                    #assign the coeffs that our analysis provided to our map of the brain IN THE REGION IT APPLIES
                    #so if we found that [brain region] has [coef] influence, mark only [brain region] with the coeff
                    SES_in_brain_data[b_roi_mask] = coefs[ii, i_feat]
                    if z_return == True:
                        SES_in_brain_data_z[b_roi_mask] = coefs_z[ii, i_feat]

                    b_found_roi = True
            
            #would it be more efficient to have a if statement here checking if b_found_roi is still false? 
            #If its true, this whole section should be skipped tbh
            for i_cort_label, cort_label in enumerate(HO_atlas_sub.labels):
                if cur_feat_name in cort_label:
                    b_roi_mask = HO_atlas_sub.maps.get_fdata() == i_cort_label
                    n_roi_vox = np.sum(b_roi_mask)
                    print('Found: %s (%i voxels)' % (cort_label, n_roi_vox))
                    if n_roi_vox == 0: print("EMPTY AREA: (subcort) ", cort_label)
                    # print(type(n_roi_vox), n_roi_vox, n_roi_vox == 0)


                    SES_in_brain_data[b_roi_mask] = coefs[ii, i_feat]
                    if z_return == True:
                        SES_in_brain_data_z[b_roi_mask] = coefs_z[ii, i_feat]

                    b_found_roi = True

            if not b_found_roi:
                print('NOT Found: %s !!!' % (cur_feat_name))
        SES_in_brain_nii = nib.Nifti1Image(SES_in_brain_data, HO_atlas_cort.maps.affine)
        nii_array.append(SES_in_brain_nii)
        if z_return == True:
            SES_in_brain_nii_z = nib.Nifti1Image(SES_in_brain_data_z, HO_atlas_cort.maps.affine)
            nii_array_z.append(SES_in_brain_nii_z)

    nii_array = nii_array[0] if coefs.shape[0] == 1 else nii_array
    if z_return == True:
        nii_array_z = nii_array_z[0] if coefs_z.shape[0] == 1 else nii_array_z
    return (nii_array, nii_array_z) if z_return == True else nii_array

def create_JHU_nii(coefs, expt_type_abbr):
    #coefs is a dataframe of tract label x n
    #There could be one of 9 different dMRI subtype
    #As well, it is possible that there is left/right asymm, which requires us to change hemisphere we plot to
    #This will not have the capacity to plot multiple coef rows, so coefs should be a series
    #however, if we have multiple modalities in coefs, all will be separately plotted
    if not isinstance(coefs, pd.Series):
        raise ValueError("Passed values should be a pandas series type object")

    nii_array = {}

    DMRI_TYPES = ['FA', 'MD', 'MO', 'L1', 'L2', 'L3', 'ICVF', 'OD', 'ISOVF']
    TRACT_DICT = {
        # mean variants (TBSS, 48 tracts, JHU atlas)
        'middle cerebellar peduncle': 1,
        'pontine crossing tract': 2,
        'genu of corpus callosum': 3,
        'body of corpus callosum': 4,
        'splenium of corpus callosum': 5,
        'fornix': 6,
        'corticospinal tract (right)': 7,
        'corticospinal tract (left)': 8,
        'medial lemniscus (right)': 9,
        'medial lemniscus (left)': 10,
        'inferior cerebellar peduncle (right)': 11,
        'inferior cerebellar peduncle (left)': 12,
        'superior cerebellar peduncle (right)': 13,
        'superior cerebellar peduncle (left)': 14,
        'cerebral peduncle (right)': 15,
        'cerebral peduncle (left)': 16,
        'anterior limb of internal capsule (right)': 17,
        'anterior limb of internal capsule (left)': 18,
        'posterior limb of internal capsule (right)': 19,
        'posterior limb of internal capsule (left)': 20,
        'retrolenticular part of internal capsule (right)': 21,
        'retrolenticular part of internal capsule (left)': 22,
        'anterior corona radiata (right)': 23,
        'anterior corona radiata (left)': 24,
        'superior corona radiata (right)': 25,
        'superior corona radiata (left)': 26,
        'posterior corona radiata (right)': 27,
        'posterior corona radiata (left)': 28,
        'posterior thalamic radiation (right)': 29,
        'posterior thalamic radiation (left)': 30,
        'sagittal stratum (right)': 31,
        'sagittal stratum (left)': 32,
        'external capsule (right)': 33,
        'external capsule (left)': 34,
        'cingulum cingulate gyrus (right)': 35,
        'cingulum cingulate gyrus (left)': 36,
        'cingulum hippocampus (right)': 37,
        'cingulum hippocampus (left)': 38,
        'fornix cres+stria terminalis (right)': 39,
        'fornix cres+stria terminalis (left)': 40,
        'superior longitudinal fasciculus (right)': 41,
        'superior longitudinal fasciculus (left)': 42,
        'superior fronto-occipital fasciculus (right)': 43,
        'superior fronto-occipital fasciculus (left)': 44,
        'uncinate fasciculus (right)': 45,
        'uncinate fasciculus (left)': 46,
        'tapetum (right)': 47,
        'tapetum (left)': 48}    
    TRACT_DICT = {
        # mean variants (TBSS, 48 tracts, JHU atlas)
        'middle cerebellar peduncle': 1,
        'pontine crossing tract': 2,
        'genu of corpus callosum': 3,
        'body of corpus callosum': 4,
        'splenium of corpus callosum': 5,
        'fornix': 6,
        'Right corticospinal tract': 7,
        'Left corticospinal tract': 8,
        'Right medial lemniscus': 9,
        'Left medial lemniscus': 10,
        'Right inferior cerebellar peduncle': 11,
        'Left inferior cerebellar peduncle': 12,
        'Right superior cerebellar peduncle': 13,
        'Left superior cerebellar peduncle': 14,
        'Right cerebral peduncle': 15,
        'Left cerebral peduncle': 16,
        'Right anterior limb of internal capsule': 17,
        'Left anterior limb of internal capsule': 18,
        'Right posterior limb of internal capsule': 19,
        'Left posterior limb of internal capsule': 20,
        'Right retrolenticular part of internal capsule': 21,
        'Left retrolenticular part of internal capsule': 22,
        'Right anterior corona radiata': 23,
        'Left anterior corona radiata': 24,
        'Right superior corona radiata': 25,
        'Left superior corona radiata': 26,
        'Right posterior corona radiata': 27,
        'Left posterior corona radiata': 28,
        'Right posterior thalamic radiation': 29,
        'Left posterior thalamic radiation': 30,
        'Right sagittal stratum': 31,
        'Left sagittal stratum': 32,
        'Right external capsule': 33,
        'Left external capsule': 34,
        'Right cingulum cingulate gyrus': 35,
        'Left cingulum cingulate gyrus': 36,
        'Right cingulum hippocampus': 37,
        'Left cingulum hippocampus': 38,
        'Right fornix cres+stria terminalis': 39,
        'Left fornix cres+stria terminalis': 40,
        'Right superior longitudinal fasciculus': 41,
        'Left superior longitudinal fasciculus': 42,
        'Right superior fronto-occipital fasciculus': 43,
        'Left superior fronto-occipital fasciculus': 44,
        'Right uncinate fasciculus': 45,
        'Left uncinate fasciculus': 46,
        'Right tapetum': 47,
        'Left tapetum': 48}

    for curr_type in DMRI_TYPES:
        #plot only type
        out_nii = np.zeros(HO_atlas_cort.maps.shape)
        area_found = 0 
        for curr_reg in coefs.index:
            if curr_type in curr_reg:
                #this column is of the type we are interested in
                cur_feat_name = curr_reg.replace(curr_type, '').strip()
            else:
                #we don't have values of our dMRI type, move onto next region
                continue

            #We now know we are looking at the type we are interested in
            #now check for tract type
            #first we have to convert the names
            cur_feat_name = parse_name(cur_feat_name, expt_type_abbr, side = coefs[curr_reg])

            #look for our region in the tract dictionary. 
            #If exists, continue, else move on
            my_tract_ind = -1
            for tract_str, tract_ind in TRACT_DICT.items():
                if tract_str == cur_feat_name:
                    my_tract_ind = tract_ind
                    break #Stop searching, and continue past this loop

            if my_tract_ind == -1:
                #It will move onto next region
                print("NOT FOUND: " + cur_feat_name)
                continue
            
            area_found += 1
            mask = JHU_nii.get_fdata() == my_tract_ind
            out_nii[mask] = coefs[curr_reg]
            print(f"FOUND ({mask.sum()} voxels) {cur_feat_name}")


        if area_found != 0:
            #We have a hit for this dMRI trype
            #make a nifty inage from it
            print(f"FOUND {area_found} tract(s) for {curr_type}")
            final_nii = nib.Nifti1Image(out_nii, affine = JHU_nii.affine)
            nii_array[curr_type] = final_nii
        else:
            print(f"NOT FOUND: {curr_type} values")
        print("\n")
    return nii_array


def create_cereb_nii(coefs, expt_type_abbr):
    #coefs is a dataframe of tract label x n
    #As well, it is possible that there is left/right asymm, which requires us to change hemisphere we plot to
    #This will not have the capacity to plot multiple coef rows, so coefs should be a series
    #however, if we have multiple modalities in coefs, all will be separately plotted
    if not isinstance(coefs, pd.Series):
        raise ValueError("Passed values should be a pandas series type object")

    # nii_array = []
    final_nii = []

    # out_nii = np.zeros(HO_atlas_cort.maps.shape)
    out_nii = np.zeros(cereb_nii.shape)
    area_found = 0 
    for curr_reg in coefs.index:
        #first we have to convert the names
        #Names are of the form
        #[Side]_[Abbr]
        #so we have to do some post processing to the name as well
        #Post process
        curr_feat_name = curr_reg if 'vermis' not  in curr_reg else 'Vermis ' + curr_reg.replace('(vermis)', '').strip()
        curr_feat_name = parse_name(curr_feat_name, expt_type_abbr, side = coefs[curr_reg])
        curr_feat_name = curr_feat_name.replace("Crus ", "Crus").replace('-','_')
        curr_feat_name = curr_feat_name.replace("Cerebellum", "").strip()
        curr_feat_name = curr_feat_name.replace(" ", "_").strip()

        #look for our region in the tract dictionary. 
        #If exists, continue, else move on
        my_tract_ind = -1
        for row_id, tract_str in cereb_lut.iterrows():
            if tract_str['name'] == curr_feat_name:
                my_tract_ind = tract_str['id']
                break #Stop searching, and continue past this loop

        if my_tract_ind == -1:
            #It will move onto next region
            print("NOT FOUND: " + curr_feat_name)
            continue

        area_found += 1
        mask = cereb_nii.get_fdata() == my_tract_ind
        out_nii[mask] = coefs[curr_reg]
        print(f"FOUND ({mask.sum()} voxels) {curr_feat_name}")


    if area_found != 0:
        #We have a hit for this dMRI trype
        #make a nifty inage from it
        print(f"FOUND {area_found} tract(s) for {expt_type_abbr}")
        final_nii = nib.Nifti1Image(out_nii, affine = cereb_nii.affine)
    else:
        print(f"NOT FOUND: final nii is empty")
    print("\n")

    return final_nii


def parseMatlabRange(r):
    """Parses a string containing a MATLAB-style ``start:stop`` or
    ``start:step:stop`` range, where the ``stop`` is inclusive).

    :arg r:   String containing MATLAB_style range.
    :returns: List of integers in the fully expanded range.
    """
    elems = [int(e) for e in r.split(':')]

    if len(elems) == 3:
        start, step, stop = elems
        if   step > 0: stop += 1
        elif step < 0: stop -= 1

    elif len(elems) == 2:
        start, stop  = elems
        stop        += 1
        step         = 1
    elif len(elems) == 1:
        start = elems[0]
        stop  = start + 1
        step  = 1
    else:
        raise ValueError('Invalid range string: {}'.format(r))

    return list(range(start, stop, step))


from nilearn import plotting as niplot

def plot_single_area(area):
    SES_in_brain_data_z = np.zeros((HO_atlas_cort.maps.shape))
    found = True
    if area in HO_atlas_sub.labels:
        b_roi_mask = HO_atlas_sub.maps.get_data() == HO_atlas_sub.labels.index(area)
        n_roi_vox = np.sum(b_roi_mask)
        print('Found: %s (%i voxels)' % (area, n_roi_vox))
        SES_in_brain_data_z[b_roi_mask] = 1
    elif area in HO_atlas_cort.labels:
        b_roi_mask = HO_atlas_cort.maps.get_data() == HO_atlas_cort.labels.index(area)
        n_roi_vox = np.sum(b_roi_mask)
        print('Found: %s (%i voxels)' % (area, n_roi_vox))
        SES_in_brain_data_z[b_roi_mask] = 1
    else: 
        print(f"NOT FOUND: {area}")
        found = False
    if found == True:
        SES_in_brain_nii_z = nib.Nifti1Image(SES_in_brain_data_z, HO_atlas_cort.maps.affine)
        niplot.plot_glass_brain(SES_in_brain_nii_z, display_mode='lyrz', title = f"HIGHLIGHTED: {area}")
    return


def plot_single_tract(area):
    SES_in_brain_data_z = np.zeros((JHU_nii.shape))
    area_new = area
    for mod in DMRI_TYPES:
        area_new = area_new.replace(mod, '')
    area_new = re.sub(' +', ' ', area_new).strip()

    found = True
    if area_new in TRACT_DICT.keys():
        b_roi_mask =JHU_nii.get_fdata() == TRACT_DICT[area_new]
        n_roi_vox = np.sum(b_roi_mask)
        print('Found: %s (%i voxels)' % (area, n_roi_vox))
        SES_in_brain_data_z[b_roi_mask] = 1
    else: 
        print(f"NOT FOUND: {area_new}")
        found = False
    if found == True:
        SES_in_brain_nii_z = nib.Nifti1Image(SES_in_brain_data_z, JHU_nii.affine)
        niplot.plot_glass_brain(SES_in_brain_nii_z, display_mode='lyrz', title = f"HIGHLIGHTED: {area}")
    return