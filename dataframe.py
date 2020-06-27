import numpy as np
from astropy.table import vstack


class Data:

    def __init__(self, df_data, object_id_col_name, time_col_name, band_col_name, band_map, df_metadata=None,
                 mag_or_flux=0, flux_col_name=None, flux_err_col_name=None, mag_col_name=None, mag_err_col_name=None,
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
        if isinstance(target, int):
            if self.target_col_name is None:
                print("Target name not given")
            else:
                event = self.df_metadata[self.target_col_name]
                index = event == target
                object_ids = self.get_all_object_ids()
                class_ids = object_ids[index]
        else:
            class_ids = None

            for target_id in target:
                event = self.df_metadata[self.target_col_name]
                index = event == target_id
                object_ids = self.get_all_object_ids()
                if class_ids is None:
                    class_ids = object_ids[index]
                else:
                    class_ids = vstack([class_ids, object_ids[index]])

        return class_ids

    def get_data_of_event(self, target):
        index = self.df_data[self.object_id_col_name] == target
        return self.df_data[index]

    def get_band_data(self, band):
        index = self.df_data[self.band_col_name] == band
        return self.df_data[index]

    def get_object_type_number(self, object_id):
        if self.target_col_name is None:
            print("Target name not given")
        object_type = np.array(
            self.df_metadata[self.target_col_name][np.argwhere(self.df_metadata[self.object_id_col_name] == object_id)])
        return object_type[0][0]

    def get_object_type_for_PLAsTiCC(self, object_id):
        if self.target_col_name is None:
            print("Target name not given")
        object_type = np.array(
            self.df_metadata[self.target_col_name][np.argwhere(self.df_metadata[self.object_id_col_name] == object_id)])
        object_num = object_type[0][0]
        if object_num == 90:
            return "SN-Ia"
        elif object_num == 67:
            return "SN-Ia-91bg"
        elif object_num == 52:
            return "SN-Iax"
        elif object_num == 42:
            return "SNII"
        elif object_num == 62:
            return "SNIbc"
        elif object_num == 95:
            return "SLSN-I"
        elif object_num == 15:
            return "TDE"
        elif object_num == 64:
            return "KN"
        elif object_num == 88:
            return "AGN"
        elif object_num == 92:
            return "RRL"
        elif object_num == 65:
            return "M-dwarf"
        elif object_num == 16:
            return "EB"

        elif object_num == 53:
            return "Mira"
        elif object_num == 6:
            return "micro-lens Single"
        else:
            return "unknown"

    def is_transient(self, object_id):
        if self.target_col_name is None:
            print("Target name not given")
        object_num = np.array(
            self.df_metadata[self.target_col_name][np.argwhere(self.df_metadata[self.object_id_col_name] == object_id)])
        object_num = object_num[0][0]
        if (object_num == 90) | (object_num == 67) | (object_num == 52) | (object_num == 42) | (object_num == 62) | (
                object_num == 95) | (object_num == 15) | (object_num == 64) | (object_num == 65):
            return 1
        elif (object_num == 88) | (object_num == 92) | (object_num == 16) | (object_num == 53) | (object_num == 6):
            return 0
        else:
            return None
