from astropy.table import Table
from SNANA_FITS_to_pd import *

def load_data_and_meta(filepath ="/media/biswajit/drive/ZTF_20190512/ZTF_MSIP_MODEL64/ZTF_MSIP_NONIaMODEL0-0001_PHOT.FITS", drop_separators=True):
    df_header, df_phot = read_fits(filepath)
    return df_header, df_phot

def extract_all_ids(df_header):
    df_meta_data = Table.read(file_path, delimiter=",")
    return np.array(['object_id'])


def extract_kilonova_ids(file_path="/media/biswajit/drive/PLAsTiCC_data/training_set_metadata.csv"):
    df_meta_data = Table.read(file_path, delimiter=",")
    kilonova_index = df_meta_data['target'] == 64
    return np.array(df_meta_data[kilonova_index]['object_id'])