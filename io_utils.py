from dataframe import Data
from astropy.table import Table, vstack
from random import random
import numpy as np
from pathlib import Path


def load_ztf_data(
        phot_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/ZTF_MSIP_MODEL64/ZTF_MSIP_NONIaMODEL0-0001_PHOT.FITS',
        drop_separators=True):
    df_header, df_phot = read_fits(phot_path)
    df_header = Table.from_pandas(df_header)
    df_phot = Table.from_pandas(df_phot)
    df_phot['FLT'][df_phot['FLT'] == b'g '] = 'g'
    df_phot['FLT'][df_phot['FLT'] == b'r '] = 'r'
    data_ob = Data(df_metadata=df_header, df_data=df_phot,
                   object_id_col_name='SNID', time_col_name='MJD', target_col_name='SNTYPE',
                   band_col_name='FLT', flux_col_name='FLUXCAL', flux_err_col_name='FLUXCALERR',
                   band_map={'g': 'g', 'r': 'r'}, bands=['g', 'r'])
    if drop_separators:
        data_ob.df_data = data_ob.df_data[data_ob.df_data[data_ob.time_col_name] != -777]
    return data_ob


def load_ztf_train_data(
        head_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/train_HEAD.FITS',
        phot_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/train_PHOT.FITS',
        drop_separators=True):
    df_header = Table.read(head_path, format='fits')
    df_phot = Table.read(phot_path, format='fits')
    data_ob = Data(df_metadata=df_header, df_data=df_phot,
                   object_id_col_name='SNID', time_col_name='MJD', target_col_name='SNTYPE',
                   band_col_name='FLT', flux_col_name='FLUXCAL', flux_err_col_name='FLUXCALERR',
                   band_map={'g': 'g', 'r': 'r'}, bands=['g', 'r'])
    return data_ob


def load_ztf_test_data(
        head_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/test_HEAD.FITS',
        phot_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/test_PHOT.FITS'):
    df_header = Table.read(head_path, format='fits')
    df_phot = Table.read(phot_path, format='fits')
    data_ob = Data(df_metadata=df_header, df_data=df_phot,
                   object_id_col_name='SNID', time_col_name='MJD', target_col_name='SNTYPE',
                   band_col_name='FLT', flux_col_name='FLUXCAL', flux_err_col_name='FLUXCALERR',
                   band_map={'g': 'g', 'r': 'r'})
    return data_ob


def load_ztf_mixed(m_numbers=None):
    if m_numbers is None:
        m_numbers = ['01', '02', '03', '12', '13', '14', '41', '43', '45', '50', '51', '60', '61', '62',
                     '63', '64', '70', '80', '81', '90']
    base_path = '/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/ZTF_MSIP_MODEL'
    data_ob = None

    for model_num in m_numbers:

        rand_int = 25
        if (int(model_num) != 45) & (int(model_num) != 50):
            if rand_int < 10:
                file_path = base_path + model_num + '/ZTF_MSIP_NONIaMODEL0-000' + str(rand_int) + '_PHOT.FITS'
                current_ob = load_ztf_data(phot_path=file_path)
            else:
                file_path = base_path + model_num + '/ZTF_MSIP_NONIaMODEL0-00' + str(rand_int) + '_PHOT.FITS'
                current_ob = load_ztf_data(phot_path=file_path)


        else:
            if rand_int < 10:
                file_path = base_path + model_num + '/ZTF_MSIP_NONIa-000' + str(rand_int) + '_PHOT.FITS'
                current_ob = load_ztf_data(phot_path=file_path)
            else:
                file_path = base_path + model_num + '/ZTF_MSIP_NONIa-00' + str(rand_int) + '_PHOT.FITS'
                current_ob = load_ztf_data(phot_path=file_path)

        if data_ob is None:
            data_ob = current_ob

        else:

            data_ob.df_data = vstack([data_ob.df_data, current_ob.df_data])
            data_ob.df_metadata = vstack([data_ob.df_metadata, current_ob.df_metadata], join_type='inner')

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


def create_alert_data_obj(data, bands):
    band_map = {}
    for item in bands:
        band_map[item] = item
    data_ob = Data(df_metadata=data, df_data=data, object_id_col_name='SNID', time_col_name='MJD',
                   band_col_name='FLT', flux_col_name='FLUXCAL', flux_err_col_name='FLUXCALERR',
                   band_map=band_map, bands=bands)
    return data_ob


def read_fits(fname, drop_separators=True):
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


def save_fits(df, fname):
    """Save data frame in fits table

    Arguments:
        df {pandas.DataFrame} -- data to save
        fname {str} -- outname, must end in .FITS
    """

    keep_cols = df.keys()
    df = df.reset_index()
    df = df[keep_cols]

    outtable = Table.from_pandas(df)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    outtable.write(fname, format='fits', overwrite=True)

def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
    """ Conversion from magnitude to Fluxcal from SNANA manual
    Parameters
    ----------
    magpsf: float
        PSF-fit magnitude from ZTF
    sigmapsf: float
    Returns
    ----------
    fluxcal: float
        Flux cal as used by SNANA
    fluxcal_err: float
        Absolute error on fluxcal (the derivative has a minus sign)
    """
    if magpsf is None:
        return None, None
    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10 ** 10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err
#
# Use examples
#

# SNANA.FITS to pd
# df_header, df_phot = read_fits('./raw/DES_Ia-0001_PHOT.FITS', drop_separators=True)

# pd to FITS
# this saves the whole data frame as a 1-D FITS table
# save_fits(df_header, "DES_Ia-0001_HEAD.FITS")
