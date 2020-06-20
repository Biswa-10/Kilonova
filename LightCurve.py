import numpy as np
from astropy.table import Table
import copy
from statistics import median
import matplotlib.pyplot as plt
from dataframe import Data


class LightCurve:

    def __init__(self, object_id, input_df, time_col_name, brightness_col_name, brightness_err_col_name, band_col_name, band_map):
        self.df = Table()
        self.object_id = object_id
        self.df[time_col_name] = input_df[time_col_name]
        self.df[brightness_col_name] = input_df[brightness_col_name]
        self.df[brightness_err_col_name] = input_df[brightness_err_col_name]
        self.df[band_col_name] = input_df[band_col_name]

        self.time_col_name = time_col_name
        self.brightness_col_name = brightness_col_name
        self.brightness_err_col_name = brightness_err_col_name
        self.band_col_name = band_col_name
        self.band_map = band_map
        # print(input_df[band_col_name])
        self.points_of_maximum, self.dates_of_maximum = self.get_dates_of_maximum()

    def get_band_data(self, band):
        index = self.df[self.band_col_name] == band
        return self.df[index]

    def get_dates_of_maximum(self):
        '''
        retrurns max flux dates and points
        for only the bands present in self.df

        enter original name as in the dataset
        '''
        dates_of_maximum = []
        points_of_maximum = {}
        for band, pb_name in self.band_map.items():
            # pb_name = band
            current_band_data = self.get_band_data(band)
            if len(current_band_data) > 0:
                current_max_index = np.argmax(current_band_data[self.brightness_col_name])
                current_max_date = current_band_data[self.time_col_name][current_max_index]
                dates_of_maximum.append(current_max_date)
                points_of_maximum[band] = [current_max_date,
                                           current_band_data[self.brightness_col_name][current_max_index]]
            # print(points_of_maximum)

        return points_of_maximum, dates_of_maximum

    def plot_light_curve(self, color_band_dict, fig=None, band=None, start_date=None, end_date=None,
                         plot_points=False, mark_label=True, mark_maximum=True, label_postfix="", xlims=None,
                         alpha=1.0):
        if fig is None:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.gca()

        if start_date is None:
            start_date = np.amin(self.df[self.time_col_name])
        if end_date is None:
            end_date = np.amax(self.df[self.time_col_name])

        if band is not None:

            if band in self.band_map.keys():

                event_df = self.get_time_sliced_df(start_date=start_date, end_date=end_date)
                band_df = self.extract_band_data(band=band, event_df=event_df)
                pb_name = self.band_map[band]
                # band_index = self.df[self.band_col_name] == band
                # start_index = self.df[self.time_col_name] >= start_date
                # end_index = self.df[self.time_col_name] <= end_date
                # index = band_index * start_index * end_index

                if plot_points:
                    ax.errorbar(band_df[self.time_col_name], band_df[self.brightness_col_name],
                                band_df[self.brightness_err_col_name], color=color_band_dict[band], fmt='o',
                                label=pb_name + label_postfix if mark_label else "", alpha=alpha)
                else:
                    ax.errorbar(band_df[self.time_col_name], band_df[self.brightness_col_name],
                                band_df[self.brightness_err_col_name],
                                color=color_band_dict[band], label=pb_name + label_postfix if mark_label else "",
                                alpha=alpha)

                if mark_maximum:
                    fig = self.mark_maximum_in_plot(color_band_dict=color_band_dict, fig=fig, band=band,
                                                    start_date=start_date, end_date=end_date)
                if xlims is not None:
                    ax.set_xlim([start_date, end_date])

            else:
                print("the band requested is not present")

        else:

            data_points_found = 0

            for band, pb_name in self.band_map.items():

                band_index = self.df[self.band_col_name] == band
                start_index = self.df[self.time_col_name] >= start_date
                end_index = self.df[self.time_col_name] <= end_date

                index = band_index * start_index * end_index

                # print(sum(index))
                if sum(index) > 0:
                    data_points_found = 1
                    df_plot_data = self.df[index]

                    if plot_points:
                        ax.errorbar(df_plot_data[self.time_col_name], df_plot_data[self.brightness_col_name],
                                    df_plot_data[self.brightness_err_col_name],
                                    color=color_band_dict[band],
                                    label=pb_name + label_postfix if mark_label else "", fmt='o',
                                    alpha=alpha)
                    else:
                        ax.errorbar(df_plot_data[self.time_col_name], df_plot_data[self.brightness_col_name],
                                    df_plot_data[self.brightness_err_col_name],
                                    color=color_band_dict[band], label=pb_name + label_postfix if mark_label else "",
                                    alpha=alpha)

                if mark_maximum:
                    fig = self.mark_maximum_in_plot(color_band_dict=color_band_dict, fig=fig, band=band,
                                                    start_date=start_date, end_date=end_date)

            if data_points_found == 0:
                print("There are no data points in the given date range")

            min_date = np.amin(self.df[self.time_col_name])
            max_date = np.amax(self.df[self.time_col_name])

            # ax.plot([start_date, end_date], [0, 0], label='y=0')
            if xlims is not None:
                ax.set_xlim([start_date, end_date])

        # ax.legend()
        # ax.remove()
        plt.xlabel("mjd", fontsize=20)
        plt.ylabel("flux", fontsize=20)
        # fig.close()

        return fig

    def get_time_sliced_df(self, start_date=None, end_date=None, current_date=None):
        event_df = self.df
        if start_date is None:
            if end_date is None:
                return event_df
            start_date = np.amax(event_df[self.time_col_name])
        if end_date is None:
            end_date = np.amax(event_df[self.time_col_name])
        start_index = event_df[self.time_col_name] >= start_date
        end_index = event_df[self.time_col_name] <= end_date
        if current_date is None:
            return event_df[start_index & end_index]
        else:
            past_index = event_df[self.time_col_name] <= current_date
            return event_df[start_index & end_index & past_index]

    def extract_band_data(self, band, event_df):
        if event_df is None:
            event_df = self.df
        band_index = event_df[self.band_col_name] == band
        return event_df[band_index]

    def get_max_point_of_band(self, band, start_date=None, end_date=None, event_df=None):
        if event_df is None:
            event_df = self.get_time_sliced_df(start_date, end_date)
        band_df = self.extract_band_data(band, event_df)
        if len(band_df) > 0:
            loc = np.argmax(band_df[self.brightness_col_name])
            max_time = band_df[self.time_col_name][loc]
            max_flux = band_df[self.brightness_col_name][loc]
            return (max_time, max_flux)
        else:
            return None

    def mark_maximum_in_plot(self, color_band_dict, fig, band=None, start_date=None, end_date=None):
        ax = fig.gca()
        if band is None:
            bands = self.band_map.keys()
            for band in bands:
                max_point = self.get_max_point_of_band(band=band, start_date=start_date, end_date=end_date)
                if max_point is not None:
                    ax.plot(max_point[0], max_point[1], color=color_band_dict[band], marker='o', markersize=15)
        else:
            max_point = self.get_max_point_of_band(band=band, start_date=start_date, end_date=end_date)
            if max_point is not None:
                ax.plot(max_point[0], max_point[1], color=color_band_dict[band], marker='o', markersize=15)
        return fig

    def find_region_priority(self, total_days_range=100):

        # print(dates_of_maximum_copy)
        dates_of_maximum_copy = copy.copy(self.dates_of_maximum)
        dates_of_maximum_copy.sort()
        priority_regions = [[]]

        for date in dates_of_maximum_copy:

            if len(priority_regions[0]) == 0:
                priority_regions[0].append(date)

            else:
                region_flag = 0
                for region in priority_regions:

                    modified_region = copy.copy(region)
                    modified_region.append(date)

                    new_median = median(modified_region)
                    # print(region)

                    for region_date in region:

                        if ((date - region_date) <= 14) | ((date - new_median) <= total_days_range / 2):
                            # print(1)
                            region.append(date)
                            region_flag = 1
                            break

                if region_flag != 1:
                    priority_regions.append([date])

        def find_len(e) -> int:
            return len(e)

        priority_regions.sort(reverse=True, key=find_len)
        return priority_regions

    def plot_max_flux_regions(self, color_band_dict, event_days_range=100, plot_points=False, priority=None):

        priority_regions = self.find_region_priority(event_days_range)
        if priority is not None:
            if priority <= 0:
                print("Error in priority value, priority number must be greater than 1")

        fig = plt.figure(figsize=(12, 6))

        for i, ranges in enumerate(priority_regions):

            mid_pt = median(ranges)
            # print(mid_pt)
            start_date = mid_pt - event_days_range / 2
            end_date = mid_pt + event_days_range / 2

            if priority is None:
                fig = self.plot_light_curve(color_band_dict, start_date=start_date, end_date=end_date,
                                            plot_points=plot_points)

            else:
                if (i < priority) | (len(ranges) == len(priority_regions[i - 1])):
                    single_band_plot = self.plot_light_curve(color_band_dict, start_date=start_date, end_date=end_date,
                                                             plot_points=plot_points)
                    ax = single_band_plot.gca()
                    ax.remove()
                    ax.figure = fig
                    fig.axes.append(ax)
                    fig.add_axes(ax)
                    plt.close(single_band_plot)
                    del single_band_plot

                    for j in range(i):
                        fig.axes[j].change_geometry(i + 1, 1, j + 1)

                    dummy = fig.add_subplot(i + 1, 1, i + 1)
                    ax.set_position(dummy.get_position())
                    dummy.remove()
                    del dummy
                    # print(ranges)

                else:
                    break

        return fig
