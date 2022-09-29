#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Toor
"""

import pandas as pd
import wfdb
import os
import sys
import numpy as np
from bs4 import BeautifulSoup
import requests


df = pd.read_csv("/Users/Toor/Desktop/MIMICIII/DIAGNOSES_ICD.csv")

ICD9 = df['ICD9_CODE'].value_counts()
#print(ICD9.head(10))
#print("\n")

name = pd.read_csv("/Users/Toor/Desktop/MIMICIII/D_ICD_DIAGNOSES.csv")
#print(name.loc[name["ICD9_CODE"] == "4019"]) # hypertension: 20703 records
#print(name.loc[name["ICD9_CODE"] == "4280"]) # congestive heart failure: 13111 records 
#print(name.loc[name["ICD9_CODE"] == "42731"]) # atrial fibrillation: 12891 records

# use atrial fibrillation for example 
# find all the records related to AF
AF = df.loc[df["ICD9_CODE"]=="4019"]
admission = AF['HADM_ID'].value_counts()
patient_AF = AF['SUBJECT_ID'].value_counts()
# total 17613 patients are diagnosed with AF

"""eg_id = 20643
# get all this patient's record in AF 
example = df.loc[df["SUBJECT_ID"]==eg_id]
# meanwhile this person has been diagnosed with a lot of other diseases 
# patient not guaranteed to have PPG data in the matched database 

eg_id2 = 7809
example2 = df.loc[df["SUBJECT_ID"]==eg_id2]

eg_id3 = 20966
example3 = df.loc[df["SUBJECT_ID"]==eg_id3]"""

only_AF_patient = AF["SUBJECT_ID"].value_counts()[AF["SUBJECT_ID"].value_counts() ==1].to_frame()
only_AF_patient.reset_index(inplace = True)
only_AF_patient.columns=['SUBJECT_ID','count']
len(only_AF_patient)
# total 15402 patients have only AF detected 
# for unique labels 


# finding a normal fibrillation (people that do not have atrial fibrillation)
# if finding it in another database, it would have different sampling rate
# MIMIC III = 125Hz 
# find people with PLETH 

# no healthy people 
df.loc[df["ICD9_CODE"]=='']

# find people with no AF 
no_AF_patient = df.merge(AF.drop_duplicates(), on = ['ROW_ID'], how='left',indicator = True)
no_AF_patient = no_AF_patient[no_AF_patient['_merge'] == 'left_only']
no_AF_patient = no_AF_patient.iloc[:,0:5]
no_AF_patient.columns = df.columns
no_AF_patient.head()
# total 630344 people with no AF


# those who have AF and who do not 
# figure out the shape of tensors 
# find out those who are in the database and have PPG data 

# match the AF patients in the clinical database with those in the matched database 
patient_db = pd.read_csv("/Users/Toor/Desktop/MIMICIII/RECORDS-numerics.txt")
patient_db = patient_db.iloc[:,0].str.split('/').str[1]
patient_db = patient_db.tolist()

# change the subject id into matchable states for those in the matched database 
def matchable(db):
    l = []
    for i in db["SUBJECT_ID"]:
        p_id = str(i)
        j = len(p_id)
        while j < 6:
            p_id = "0" + p_id
            j = j + 1
        p_id = "p" + p_id
        l.append(p_id)
    return l
    
AF_patient_id = matchable(only_AF_patient)
no_AF_patient_id = matchable(no_AF_patient)

# get the list of AF of patients that exist in the matchable db 
matched_AF_patient_list = [i for i in AF_patient_id if i in patient_db]
# make sure there is not duplicate patient id
matched_AF_patient_list = list(set(matched_AF_patient_list))
# 4045 patients with AF are in the matched dataset 

matched_no_AF_patient_list = [i for i in no_AF_patient_id if i in patient_db]
# delete all duplicates 
matched_no_AF_patient_list = list(set(matched_no_AF_patient_list))
# make sure the patient id here are different from the patient id in matched_AF 
matched_no_AF_patient_list = list(filter(lambda a: a not in matched_AF_patient_list, matched_no_AF_patient_list))
#print(any(check in matched_AF_patient_list for check in matched_no_AF_patient_list))
# 6223 patients with no AF are in the matched dataset 


# 1D CNN 
# task is to figure out people with AF and people without 
# shape of the signals should be similar 

def get_url_paths(url, folder = ''):
    response = requests.get(url + folder)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    temp = 'mimic3wdb-matched/' + folder
    parent = [temp + node.get('href')[:-1] for node in soup.find_all('a')]
    return parent

url = 'https://physionet.org/static/published-projects/mimic3wdb-matched/1.0/'
patients = get_url_paths(url)

matched_AF_patient_pre = list({i[:3] for i in matched_AF_patient_list})
matched_no_AF_patient_pre = list({i[:3] for i in matched_no_AF_patient_list})

# get the url for matched AF patients 
matched_AF_patients_url = []
for i in matched_AF_patient_pre:
    matched_AF_patients_url.append(get_url_paths(url, i+'/'))

# get the url for matched non AF patients 
matched_no_AF_patients_url = []
for i in matched_no_AF_patient_pre:
    matched_no_AF_patients_url.append(get_url_paths(url, i+'/'))

# check if there is any common element in nested list 
# set.intersection(*map(set,matched_AF_patients_url), *map(set,matched_no_AF_patients_url))
# no, hooray!


cwd = os.getcwd()
PPG_data_AF = []
PPG_data_no_AF = []
patients_to_use = 3

def download_files(stored_in, list_used):
    for i in range(1):
        for m, patient in enumerate(list_used[i][1:1+patients_to_use]):
            print("Loading Patient " + str(m) + "'s data")
            # Turn off print statements
            sys.stdout = open(os.devnull, 'w')
    
            wfdb.io.dl_database(patient, cwd)
            
            # Turn print statements back on
            sys.stdout = sys.__stdout__
    
            # Load Patient PLETH Data
            for filename in os.listdir(cwd):
                if filename.endswith('.hea'):
                    with open(cwd+'/'+filename, "r") as f:
                        if 'layout' not in filename: 
                            try: 
                                for idx, line in enumerate(f.readlines()):
                                    if "PLETH" in line:
                                        record = wfdb.rdrecord(filename[:-4])
                                        if type(record.p_signal) != None and record.fs == 125:
                                            stored_in.append(record.p_signal[:, idx - 1])
                                        break 
                                os.remove(filename[:-4]+ '.dat')
                            except:
                                pass
                    os.remove(filename)
                if filename.endswith('n.dat'):
                    os.remove(filename)
                    
download_files(PPG_data_AF,matched_AF_patients_url)
download_files(PPG_data_no_AF,matched_no_AF_patients_url)

print('Total number of records (different lengths): ' + str(len(PPG_data_AF)))
print('Total number of records (different lengths): ' + str(len(PPG_data_no_AF)))

# What to do with nan?

# Split data into 10-12 second intervals from each array
# (sampling rate is 125x per second)
recordings_AF = []
recordings_no_AF = []
sample_length = 1250
def split_data(list_used, stored_in):  
    for patient in list_used:
        temp = 0
        length = len(patient) - sample_length
        while(length > temp):
            temp2 = patient[temp:temp+sample_length]
            if not np.isnan(temp2).any(): stored_in.append(temp2)
            temp += sample_length
#del PPG_data_AF

split_data(PPG_data_AF, recordings_AF)
split_data(PPG_data_no_AF, recordings_no_AF)

AF_output = [1 for i in range(len(recordings_AF))]
AF_no_output = [0 for i in range(len(recordings_no_AF))]

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

