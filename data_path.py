import os

# 10 GENRES USED FOR CLASSIFICATION
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# DEFINE PATHS
DATAPATH = os.getcwd() + '/Data/genres_original/'#
RAW_DATAPATH = os.getcwd() + '/utils/raw_data.pkl'
SET_DATAPATH = os.getcwd() + '/utils/data_set.pkl'
MODELPATH = os.getcwd() + '/model/net.pt'