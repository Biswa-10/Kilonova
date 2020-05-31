from SNANA_FITS_to_pd import read_fits
from dataframe import Data
from astropy.table import Table


def load_ztf_data(filepath="/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/ZTF_MSIP_MODEL64/ZTF_MSIP_NONIaMODEL0-0001_PHOT.FITS",
                  drop_separators=True):
    df_header, df_phot = read_fits(filepath)
    data_ob = Data(df_metadata=df_header, df_data=df_phot, object_id_col_name='SNID', time_col_name='MJD',
                   band_col_name='FLT', flux_col_name='FLUXCAL', flux_err_col_name='FLUXCALERR',
                   band_map={b'g ': 'g', b'r ': 'r'})
    return data_ob


def load_PLAsTiCC_data(phot_df_file_path="/media/biswajit/drive/Kilonova_datasets/PLAsTiCC_data/training_set.csv",
                       meta_df_file_path="/media/biswajit/drive/Kilonova_datasets/PLAsTiCC_data/training_set_metadata.csv"):
    df_meta_data = Table.read(meta_df_file_path, delimiter=",")
    df_data = Table.read(phot_df_file_path)
    data_ob = Data(df_metadata=df_meta_data, df_data=df_data, object_id_col_name='object_id', time_col_name='mjd',
                   band_col_name='passband', flux_col_name='flux', flux_err_col_name='flux_err',
                   target_col_name='target', band_map={0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'})
    return data_ob


def load_RESSPECT_data(phot_df_file_path="/media/biswajit/drive/Kilonova_datasets/RESSPECT"
                                         "/RESSPECT_PERFECT_LIGHTCURVE.csv",
                       meta_df_file_path="/media/biswajit/drive/Kilonova_datasets/RESSPECT/RESSPECT_PERFECT_HEAD.csv"):
    df_meta_data = Table.read(meta_df_file_path, delimiter=",")
    df_data = Table.read(phot_df_file_path)
    data_ob = Data(df_metadata=df_meta_data, df_data=df_data, object_id_col_name='SNID', time_col_name='MJD',
                   band_col_name='FLT', flux_col_name='FLUXCAL', flux_err_col_name='FLUXCALERR',
                   band_map={'u': 'u', 'g': 'g', 'r': 'r', 'i': 'i', 'z': 'z', 'Y': 'y'})
    return data_ob
