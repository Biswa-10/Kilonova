from src.dataframe import Data
from src.io_utils import *
import matplotlib.pyplot as plt
from src.RFModel import RFModel
from pandas import DataFrame

from src.prediction_evaluation import PredictionEvaluation


def main():
    train_ob = load_ztf_train_data(
        head_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/train_master_HEAD.FITS',
        phot_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/train_master_PHOT.FITS')
    test_ob = load_ztf_test_data(head_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/test_master_HEAD.FITS',
                                 phot_path='/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/test_master_PHOT.FITS')

    num_pc_components = 3
    prediction_type_nos = [150, 151]
    color_band_dict = {'g': 'C2', 'r': 'C3'}
    num_alert_days = 50
    use_number_of_points_per_band = False

    min_flux_threshold = 200
    use_random_current_date = True
    bands = ['g', 'r']

    sample_numbers_train = {101: 1000,
                            102: 1000,
                            103: 1000,
                            112: 1000,
                            113: 1000,
                            114: 1000,
                            141: 1000,
                            143: 1000,
                            145: 1000,
                            150: 1000,
                            151: 1000,
                            160: 1000,
                            161: 1000,
                            162: 1000,
                            163: 1000,
                            164: 1000,
                            170: 1000,
                            180: 1000,
                            181: 1000,
                            183: 1000,
                            190: 10}

    sample_numbers_test = {101: 1000,
                           102: 1000,
                           103: 1000,
                           112: 1000,
                           113: 1000,
                           114: 1000,
                           141: 1000,
                           143: 1000,
                           145: 1000,
                           150: 1000,
                           151: 1000,
                           160: 1000,
                           161: 1000,
                           162: 1000,
                           163: 1000,
                           164: 1000,
                           170: 1000,
                           180: 1000,
                           181: 1000,
                           183: 1000,
                           190: 1000}

    train_features_path = '/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/train_features_master_3_pcs_u_band.csv'
    test_features_path = '/media/biswajit/drive/Kilonova_datasets/ZTF_20190512/test_features_master_3_pcs_u_band.csv'

    rf_model = RFModel(prediction_type_nos=prediction_type_nos, sample_numbers_train=sample_numbers_train,
                       sample_numbers_test=sample_numbers_test, min_flux_threshold=min_flux_threshold,
                       num_pc_components=num_pc_components, use_random_current_date=use_random_current_date,
                       bands=bands, num_alert_days=num_alert_days)

    rf_model.create_features_df("train", train_ob)
    rf_model.create_features_df("test", train_ob)
    print("training")
    rf_model.train_model(use_number_of_points_per_band=use_number_of_points_per_band)

    rf_model.plot_predictions(color_band_dict=color_band_dict, save_fig_path="results/")


if __name__ == "__main__":
    main()