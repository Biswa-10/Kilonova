from dataframe import Data
from astropy.table import Table, vstack
from random import random
import numpy as np


def load_ztf_data(
        filepath="/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/ZTF_MSIP_MODEL64/ZTF_MSIP_NONIaMODEL0-0001_PHOT.FITS",
        drop_separators=True):
    df_header, df_phot = read_fits(filepath)
    data_ob = Data(df_metadata=Table.from_pandas(df_header), df_data=Table.from_pandas(df_phot),
                   object_id_col_name='SNID', time_col_name='MJD',
                   band_col_name='FLT', flux_col_name='FLUXCAL', flux_err_col_name='FLUXCALERR',
                   band_map={b'g ': 'g', b'r ': 'r'})
    return data_ob


def load_ztf_mixed():
    m_numbers = ['01', '02', '03', '12', '13', '14', '41', '43', '45', '50', '51', '60', '61', '62', '63', '64', '70',
                 '80', '81', '90']
    base_path = '/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/ZTF_MSIP_MODEL'
    data_ob = None

    for model_num in m_numbers:

        rand_int = 25
        if (int(model_num) != 45) & (int(model_num) != 50):
            if rand_int < 10:
                file_path = base_path + model_num + '/ZTF_MSIP_NONIaMODEL0-000' + str(rand_int) + '_PHOT.FITS'
                current_ob = load_ztf_data(filepath=file_path)
            else:
                file_path = base_path + model_num + '/ZTF_MSIP_NONIaMODEL0-00' + str(rand_int) + '_PHOT.FITS'
                current_ob = load_ztf_data(filepath=file_path)


        else:
            if rand_int < 10:
                file_path = base_path + model_num + '/ZTF_MSIP_NONIa-000' + str(rand_int) + '_PHOT.FITS'
                current_ob = load_ztf_data(filepath=file_path)
            else:
                file_path = base_path + model_num + '/ZTF_MSIP_NONIa-00' + str(rand_int) + '_PHOT.FITS'
                current_ob = load_ztf_data(filepath=file_path)

        if data_ob is None:
            current_ob.df_metadata['target'] = np.ones(len(current_ob.df_metadata)) * int(model_num)
            data_ob = current_ob

        else:
            current_ob.df_metadata['target'] = int(model_num)
            data_ob.df_data = vstack([data_ob.df_data, current_ob.df_data])
            current_ob.df_metadata['target'] = np.ones(len(current_ob.df_metadata)) * int(model_num)
            data_ob.df_metadata = vstack([data_ob.df_metadata, current_ob.df_metadata], join_type='inner')

            # data_ob.df_metadata = .df_data.keys())
            # data_ob.df_data= vstack([data_ob.df_metadata, current_ob.df_metadata['SNID']])
    data_ob.target_col_name = 'target'
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


def read_fits(fname, drop_separators=False):
    """Load SNANA formatted data and cast it to a PANDAS dataframe

    Args:
        fname (str): path + name to PHOT.FITS file
        drop_separators (Boolean): if -777 are to be dropped

    Returns:
        (pandas.DataFrame) dataframe from PHOT.FITS file (with ID)
        (pandas.DataFrame) dataframe from HEAD.FITS file
    """

    # load photometry
    dat = Table.read(fname, format='fits')
    df_phot = dat.to_pandas()
    # failsafe
    if df_phot.MJD.values[-1] == -777.0:
        df_phot = df_phot.drop(df_phot.index[-1])
    if df_phot.MJD.values[0] == -777.0:
        df_phot = df_phot.drop(df_phot.index[0])

    # load header
    header = Table.read(fname.replace("PHOT", "HEAD"), format="fits")
    df_header = header.to_pandas()
    df_header["SNID"] = df_header["SNID"].astype(np.int32)

    # add SNID to phot for skimming
    arr_ID = np.zeros(len(df_phot), dtype=np.int32)
    # New light curves are identified by MJD == -777.0
    arr_idx = np.where(df_phot["MJD"].values == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df_phot)])))
    # Fill in arr_ID
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        # index starts at zero
        arr_ID[start:end] = df_header.SNID.iloc[counter - 1]
    df_phot["SNID"] = arr_ID

    if drop_separators:
        df_phot = df_phot[df_phot.MJD != -777.000]

    return df_header, df_phot
