import numpy as np
from astropy.table import Table
import copy
from statistics import median
import matplotlib.pyplot as plt


class LightCurve:

    def __init__(self, input_df, time_col_name, brightness_col_name, brightness_err_col_name, passband_col_name):
        self.df = Table()
        self.df['mjd'] = input_df[time_col_name]
        self.df['brightness'] = input_df[brightness_col_name]
        self.df['brightness_err'] = input_df[brightness_err_col_name]
        self.df['passband'] = input_df[passband_col_name]
        passband_values = np.unique(input_df[passband_col_name])
        passband_dict = {}
        for i, pb_val in enumerate(passband_values):
            passband_dict[i] = pb_val
        self.passband_dict = passband_dict
        self.points_of_maximum, self.dates_of_maximum = self.get_dates_of_maximum()

    def get_dates_of_maximum(self):
        '''
        retrurns max flux dates and points
        for only the bands present in self.df

        enter original name as in the dataset
        '''
        dates_of_maximum = []
        points_of_maximum = {}
        for band in self.passband_dict.keys():
            ind = self.df['passband'] == self.passband_dict[band]
            pb_name = self.passband_dict[band]
            current_band_data = self.df[ind]
            current_max_index = np.argmax(current_band_data['brightness'])
            current_max_date = current_band_data['mjd'][current_max_index]
            dates_of_maximum.append(current_max_date)
            points_of_maximum[pb_name] = [current_max_date, current_band_data['brightness'][current_max_index]]
            #print(points_of_maximum)

        return points_of_maximum, dates_of_maximum

    def plot_light_curve(self, color_band_dict, band_num=None, start_date=None, end_date=None, points = False):

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)

        if start_date is None:
            start_date = np.amin(self.df['mjd'])
        if end_date is None:
            end_date = np.amax(self.df['mjd'])

        if band_num is not None:

            if band_num in self.passband_dict.values():

                pbname = band_num
                #print(pbname)
                band_index = self.df['passband'] == band_num
                start_index = self.df['mjd'] >= start_date
                end_index = self.df['mjd'] <= end_date
                index = band_index * start_index * end_index

                if sum(index) <= 0:
                    print("the band requested has no data points in the given date range")

                df_plot_data = self.df[index]

                ax.errorbar(df_plot_data['mjd'], df_plot_data['brightness'], df_plot_data['brightness_err'],
                            color=color_band_dict[band_num], fmt = 'o',
                            label=pbname)
                # ax.plot([start_date, end_date], [0, 0], label='y=0')

                ax.plot(self.points_of_maximum[band_num][0], self.points_of_maximum[band_num][1],
                        color=color_band_dict[band_num],
                        marker='o', markersize=10)

                ax.set_xlim([0, 100])

            else:
                print("the band requested is not present")

        else:

            data_points_found = 0
            for band in self.passband_dict.values():

                #print(band)
                pb_name = band

                band_index = self.df['passband'] == band
                start_index = self.df['mjd'] >= start_date
                end_index = self.df['mjd'] <= end_date

                index = band_index * start_index * end_index

                # print(sum(index))
                if sum(index) > 0:
                    data_points_found = 1

                    df_plot_data = self.df[index]
                    ax.errorbar(df_plot_data['mjd'], df_plot_data['brightness'], df_plot_data['brightness_err'],
                                color=color_band_dict[band], label=pb_name, fmt='o')

                if self.points_of_maximum is not None:

                    if band_num in self.points_of_maximum.keys():
                        print("could not find the band number " + str(band_num) + " in points_of_maximum")

                    else:

                        ax.plot(self.points_of_maximum[band][0], self.points_of_maximum[band][1],
                                color=color_band_dict[band], marker='o', markersize=10)

            if data_points_found == 0:
                print("There are no data points in the given date range")

            min_date = np.amin(self.df['mjd'])
            max_date = np.amax(self.df['mjd'])

            # ax.plot([start_date, end_date], [0, 0], label='y=0')
            ax.set_xlim([0, 100])

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

    def plot_max_flux_regions(self, color_band_dict, event_days_range=100, priority=None):

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
                fig = self.plot_light_curve(color_band_dict, start_date=start_date, end_date=end_date)

            else:
                if (i < priority) | (len(ranges) == len(priority_regions[i - 1])):
                    single_band_plot = self.plot_light_curve(color_band_dict, start_date=start_date, end_date=end_date)
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
