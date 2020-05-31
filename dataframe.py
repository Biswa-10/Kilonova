import numpy as np


class Data:

    def __init__(self, df_data, object_id_col_name, time_col_name, band_col_name, band_map, df_metadata=None,
                 mag_or_flux=0,
                 flux_col_name=None, flux_err_col_name=None, mag_col_name=None, mag_err_col_name=None,
                 target_col_name=None):

        self.object_id_col_name = object_id_col_name
        self.time_col_name = time_col_name
        self.flux_col_name = flux_col_name
        self.mag_col_name = mag_col_name
        self.flux_err_col_name = flux_err_col_name
        self.mag_err_col_name = mag_err_col_name
        self.band_col_name = band_col_name
        self.target_col_name = target_col_name
        self.df_metadata = df_metadata
        self.df_data = df_data
        self.band_map = band_map

        if mag_or_flux == 1:
            if mag_col_name is None:
                print("error")

        elif mag_or_flux == 0:
            if flux_col_name is None:
                print("error")

        else:
            print("error")

    def get_all_object_ids(self):
        if self.df_metadata is not None:
            return self.df_metadata[self.object_id_col_name]
        else:
            return np.unique(self.df_data[self.object_id_col_name])

    def get_ids_of_event_type(self, target):
        if self.target_col_name is None:
            print("Target name not given")
        else:
            event = self.df_metadata[self.target_col_name]
            index = event == target
            object_ids = self.get_all_object_ids()
        return object_ids[index]

    def get_data_of_event(self, target):
        index = self.df_data[self.object_id_col_name] == target
        return self.df_data[index]

    def get_band_data(self, band):
        index = self.df_data[self.band_col_name] == band
        return self.df_data[index]

    def get_object_type(self, object_id):
        if self.target_col_name is None:
            print("Target name not given")
        object_type = np.array(self.df_metadata[self.target_col_name][np.argwhere(self.df_metadata[self.object_id_col_name] == object_id)])
        return object_type[0][0]

