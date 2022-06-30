import numpy as np
import nibabel as nib
import os
import random
from glob import glob
from pathlib import Path
from tqdm import tqdm

def listdir_nothidden_filenames(path, filetype=None):
    if not filetype:
        paths = glob(os.path.join(path, '*'))
    else:
        paths = glob(os.path.join(path, '*.%s' % filetype))
    files = [os.path.basename(path) for path in paths]
    return files

def load_data(path):
    IDs = listdir_nothidden_filenames(path)
    data = {}
    for i, ID in enumerate(IDs):
        ID_data = {}
        for quantity in listdir_nothidden_filenames(os.path.join(path, ID)):
            ID_data[quantity.split('.')[0]] = np.load(os.path.join(path, ID, quantity))
        data[i] = ID_data
    
    return data

def create_working_dataset(master_path, copy_path):
    Path(copy_path).mkdir(parents=True, exist_ok=True)

    subject_ids = os.listdir(master_path)
    num_subjects = len(subject_ids)

    seed = 12
    random.seed(seed)
    random.shuffle(subject_ids) #shuffle the subject list

    #train,val,test split = [0.8, 0.1, 0.1]
    #create train dataset
    train_path = os.path.join(copy_path, 'train')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    for subject_id in tqdm(subject_ids[:int(0.8*num_subjects)]):
        subject_id_dir = os.path.join(master_path, subject_id)
        subject_scan_dates = os.listdir(subject_id_dir)
        #num_scans = len(subject_scan_dates) #number of scans for this subject

        for i, date in enumerate(subject_scan_dates):
            subject_date_dir = os.path.join(subject_id_dir, date)

            save_subject_dir = os.path.join(train_path, '%s_%d' % (subject_id, i+1))
            Path(save_subject_dir).mkdir(parents=True, exist_ok=True)

            mri_path = os.path.join(subject_date_dir, 'mri')
            mri_scans = os.listdir(mri_path)
            #save_mri_path = os.path.join(save_subject_dir, 'mri')
            #Path(save_mri_path).mkdir(parents=True, exist_ok=True)
            
            pet_path = os.path.join(subject_date_dir, 'pet')
            pet_scans = os.listdir(pet_path)
            #save_pet_path = os.path.join(save_subject_dir, 'pet')
            #Path(save_pet_path).mkdir(parents=True, exist_ok=True)

            for mri_scan in mri_scans:
                img_nifti = nib.load(os.path.join(mri_path, mri_scan))
                img_npy = np.array(img_nifti.dataobj)
                #print('mri shape: ', img_npy.shape)
                np.save(file=os.path.join(save_subject_dir, 'img_mri'), arr = img_npy)
            
            for pet_scan in pet_scans:
                img_nifti = nib.load(os.path.join(pet_path, pet_scan))
                img_npy = np.array(img_nifti.dataobj)
                #print('pet shape: ', img_npy.shape)
                np.save(file=os.path.join(save_subject_dir, 'img_pet'), arr = img_npy)


    #create the validation dataset
    validation_path = os.path.join(copy_path, 'validation')
    Path(validation_path).mkdir(parents=True, exist_ok=True)
    for subject_id in tqdm(subject_ids[int(0.8*num_subjects):int(0.9*num_subjects)]):
        subject_id_dir = os.path.join(master_path, subject_id)
        subject_scan_dates = os.listdir(subject_id_dir)
        #num_scans = len(subject_scan_dates) #number of scans for this subject

        for i, date in enumerate(subject_scan_dates):
            subject_date_dir = os.path.join(subject_id_dir, date)

            save_subject_dir = os.path.join(validation_path, '%s_%d' % (subject_id, i+1))
            Path(save_subject_dir).mkdir(parents=True, exist_ok=True)

            mri_path = os.path.join(subject_date_dir, 'mri')
            mri_scans = os.listdir(mri_path)
            #save_mri_path = os.path.join(save_subject_dir, 'mri')
            #Path(save_mri_path).mkdir(parents=True, exist_ok=True)
            
            pet_path = os.path.join(subject_date_dir, 'pet')
            pet_scans = os.listdir(pet_path)
            #save_pet_path = os.path.join(save_subject_dir, 'pet')
            #Path(save_pet_path).mkdir(parents=True, exist_ok=True)

            for mri_scan in mri_scans:
                img_nifti = nib.load(os.path.join(mri_path, mri_scan))
                img_npy = np.array(img_nifti.dataobj)
                #print('mri shape: ', img_npy.shape)
                np.save(file=os.path.join(save_subject_dir, 'img_mri'), arr = img_npy)
            
            for pet_scan in pet_scans:
                img_nifti = nib.load(os.path.join(pet_path, pet_scan))
                img_npy = np.array(img_nifti.dataobj)
                #print('pet shape: ', img_npy.shape)
                np.save(file=os.path.join(save_subject_dir, 'img_pet'), arr = img_npy)

    #create the test dataset
    test_path = os.path.join(copy_path, 'test')
    Path(test_path).mkdir(parents=True, exist_ok=True)
    for subject_id in tqdm(subject_ids[int(0.9*num_subjects):]):
        subject_id_dir = os.path.join(master_path, subject_id)
        subject_scan_dates = os.listdir(subject_id_dir)
        #num_scans = len(subject_scan_dates) #number of scans for this subject

        for i, date in enumerate(subject_scan_dates):
            subject_date_dir = os.path.join(subject_id_dir, date)

            save_subject_dir = os.path.join(test_path, '%s_%d' % (subject_id, i+1))
            Path(save_subject_dir).mkdir(parents=True, exist_ok=True)

            mri_path = os.path.join(subject_date_dir, 'mri')
            mri_scans = os.listdir(mri_path)
            #save_mri_path = os.path.join(save_subject_dir, 'mri')
            #Path(save_mri_path).mkdir(parents=True, exist_ok=True)
            
            pet_path = os.path.join(subject_date_dir, 'pet')
            pet_scans = os.listdir(pet_path)
            #save_pet_path = os.path.join(save_subject_dir, 'pet')
            #Path(save_pet_path).mkdir(parents=True, exist_ok=True)

            for mri_scan in mri_scans:
                img_nifti = nib.load(os.path.join(mri_path, mri_scan))
                img_npy = np.array(img_nifti.dataobj)
                #print('mri shape: ', img_npy.shape)
                np.save(file=os.path.join(save_subject_dir, 'img_mri'), arr = img_npy)
            
            for pet_scan in pet_scans:
                img_nifti = nib.load(os.path.join(pet_path, pet_scan))
                img_npy = np.array(img_nifti.dataobj)
                #print('pet shape: ', img_npy.shape)
                np.save(file=os.path.join(save_subject_dir, 'img_pet'), arr = img_npy)


master_path = '/home/gb511/score_sde_pytorch-1/datasets/selected_organised_data'
copy_path = '/home/gb511/score_sde_pytorch-1/datasets/selected_ADNI_data_tr'

create_working_dataset(master_path, copy_path)