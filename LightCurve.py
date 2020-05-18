import numpy as np
from astropy.table import Table
import copy
from statistics import median
import matplotlib.pyplot as plt
from dataframe import Data


class LightCurve:

    def __init__(self, input_df, time_col_name, brightness_col_name, brightness_err_col_name, band_col_name,
                 band_map):
        self.df = Table()
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

    def plot_light_curve(self, color_band_dict, band=None, start_date=None, end_date=None, plot_points=False):

        print(plot_points)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)

        if start_date is None:
            start_date = np.amin(self.df[self.time_col_name])
        if end_date is None:
            end_date = np.amax(self.df[self.time_col_name])

        if band is not None:

            if band in self.band_map.keys():

                pb_name = self.band_map[band]
                band_index = self.df[self.band_col_name] == band
                start_index = self.df[self.time_col_name] >= start_date
                end_index = self.df[self.time_col_name] <= end_date
                index = band_index * start_index * end_index

                if sum(index) <= 0:
                    print("the band requested has no data points in the given date range")

                df_plot_data = self.df[index]

                if plot_points == True:
                    ax.errorbar(df_plot_data[self.time_col_name], df_plot_data[self.brightness_col_name],
                                df_plot_data[self.brightness_err_col_name],
                                color=color_band_dict[band], fmt='o',
                                label=pb_name)
                else:
                    ax.errorbar(df_plot_data[self.time_col_name], df_plot_data[self.brightness_col_name],
                                df_plot_data[self.brightness_err_col_name],
                                color=color_band_dict[band], label=pb_name)

                ax.plot(self.points_of_maximum[band][0], self.points_of_maximum[band][1],
                        color=color_band_dict[band],
                        marker='o', markersize=10)

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

                    if plot_points == True:
                        ax.errorbar(df_plot_data[self.time_col_name], df_plot_data[self.brightness_col_name],
                                    df_plot_data[self.brightness_err_col_name],
                                    color=color_band_dict[band], label=pb_name, fmt='o')
                    else:
                        ax.errorbar(df_plot_data[self.time_col_name], df_plot_data[self.brightness_col_name],
                                    df_plot_data[self.brightness_err_col_name],
                                    color=color_band_dict[band], label=pb_name)

                if self.points_of_maximum is not None:
                    if band not in self.points_of_maximum.keys():
                        print("could not find the band number " + str(band) + " in points_of_maximum")

                    else:
                        ax.plot(self.points_of_maximum[band][0], self.points_of_maximum[band][1],
                                color=color_band_dict[band], marker='o', markersize=10)

            if data_points_found == 0:
                print("There are no data points in the given date range")

            min_date = np.amin(self.df[self.time_col_name])
            max_date = np.amax(self.df[self.time_col_name])

            # ax.plot([start_date, end_date], [0, 0], label='y=0')
            ax.set_xlim([start_date, end_date])

        ax.legend()
        # ax.remove()
        ax.set_xlabel("mjd", fontsize=20)
        ax.set_ylabel("flux", fontsize=20)
        # fig.close()

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

        fig = plt.figure(figsize=(16, 16))

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

                    for j in range(i):
                        fig.axes[j].change_geometry(i + 1, 1, j + 1)

                    dummy = fig.add_subplot(i + 1, 1, i + 1)
                    ax.set_position(dummy.get_position())
                    dummy.remove()

                    # print(ranges)

                else:
                    break

        return fig
