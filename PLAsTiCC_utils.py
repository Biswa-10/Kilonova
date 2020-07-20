import numpy as npgit
from astropy.table import Table

def load_metadata(file_path="/media/biswajit/drive/PLAsTiCC_data/training_set_metadata.csv"):
    df_meta_data = Table.read(file_path,delimiter=",")
    return df_meta_data


def load_data(file_path="/media/biswajit/drive/PLAsTiCC_data/training_set.csv"):
    table = Table.read(file_path)
    return table


def extract_all_ids(file_path="/media/biswajit/drive/PLAsTiCC_data/training_set_metadata.csv"):
    df_meta_data = Table.read(file_path, delimiter=",")
    return np.array(df_meta_data['object_id'])


def extract_kilonova_ids(file_path="/media/biswajit/drive/PLAsTiCC_data/training_set_metadata.csv"):
    df_meta_data = Table.read(file_path, delimiter=",")
    kilonova_index = df_meta_data['target'] == 64
    return np.array(df_meta_data[kilonova_index]['object_id'])


def getredshift(object_id):
    df_meta_data = load_metadata()
    index = np.where(df_meta_data['object_id']==object_id)
    red_shift_specz = df_meta_data['hostgal_specz'][index]
    red_shift_photoz = df_meta_data['hostgal_photoz'][index]
    return red_shift_specz, red_shift_photoz


def getddf(object_id):
    df_meta_data = load_metadata()
    index = np.where(df_meta_data['object_id']==object_id)
    ddf = df_meta_data['ddf'][index]
    return ddf
