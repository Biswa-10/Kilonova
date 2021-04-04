from dataframe import Data
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn import metrics
import matplotlib.pyplot as plt


def ztf_ob_type_name(type_no: int):
    if type_no == 141:
        return '141: 91BG'
    if type_no == 143:
        return '143: Iax'
    if type_no == 145:
        return '145: point Ia'
    if type_no == 150:
        return '150: KN GW170817'
    if type_no == 151:
        return '151: KN Karsen 2017'
    if type_no == 160:
        return '160: Superluminous SN'
    if type_no == 161:
        return '161: pair instability SN'
    if type_no == 162:
        return '162: ILOT'
    if type_no == 163:
        return '163: CART'
    if type_no == 164:
        return '164: TDE'
    if type_no == 170:
        return '170: AGN'
    if type_no == 180:
        return '180: RRLyrae'
    if type_no == 181:
        return 'M 181: dwarf_flares'
    if type_no == 183:
        return '183: PHOEBE'
    if type_no == 190:
        return '190: uLens_BSR'
    if type_no == 191:
        return '191: uLens_Bachelet'
    if type_no == 192:
        return '192: uLens_STRING'
    if type_no == 114:
        return '114: MOSFIT-IIn'
    if type_no == 113:
        return '113: Core collapse Type II using pca '
    if type_no == 112:
        return '112: Core collapse Type II'
    if type_no == 102:
        return '102: MOSFIT-Ibc'
    if type_no == 103:
        return '103: Core collapse Type Ibc'
    if type_no == 101:
        return '101: Ia SN'
    if type_no == 0:
        return '0: Unknown'


class PredictionEvaluation:
    """
    Class to to evaluate predictions of classifier for ZTF dataset

    :param train_ob: training data object
    :param y_pred_train: predictions on training data
    :param test_ob: testing data object
    :param y_pred_test: predictions on test data
    """

    def __init__(self, train_ob: Data, y_pred_train, test_ob: Data, y_pred_test):

        self.train_ob = train_ob
        self.train_sample_numbers = train_ob.sample_numbers
        if train_ob.features_df is None:
            print("train features df not created, use create_features_df func")
        self.train_ob.features_df['y_pred'] = np.asarray(y_pred_train)

        if train_ob.prediction_type_nos.sort() == test_ob.prediction_type_nos.sort():
            self.prediction_type_nos = train_ob.prediction_type_nos
        else:
            print("prediction type inconsistant")
        if train_ob.num_pc_components == test_ob.num_pc_components:
            self.num_pc_components = train_ob.num_pc_components
        else:
            print("number of components inconsistent")

        self.test_ob = test_ob
        self.test_sample_numbers = test_ob.sample_numbers
        if test_ob.features_df is None:
            print("test features df not created, use create_features_df func")
        self.test_ob.features_df['y_pred'] = np.asarray(y_pred_test)
        self.performance_statistics_df = self.get_performance_statistics_df()

    def plot_contamination_statistics(self, ax=None):
        """
        plot displaying total number of events and number of events correctly classified for each event type.
        :param ax: axes on which plot is to be made
        :return:
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_axes([0, 0, .9, .9])

        col_type_nos = np.array(self.performance_statistics_df.columns)
        # print(col_type_nos)
        pred_col_names = [ztf_ob_type_name(item) for item in self.prediction_type_nos]
        non_pred_types = col_type_nos[~np.isin(col_type_nos, self.prediction_type_nos)]
        # print(pred_col_names)
        non_pred_type_names = [ztf_ob_type_name(item) for item in non_pred_types]
        col_type_names = [ztf_ob_type_name(item) for item in col_type_nos]

        # print(col_type_nos)
        # print(non_pred_types)

        # print(np.where(np.in1d(non_pred_types, col_type_nos)))
        ax.barh(col_type_names, self.performance_statistics_df.loc[1], alpha=.6, tick_label=col_type_names,
                color='bisque', ec='black', linewidth=1, label="total number of events")
        ax.barh(non_pred_type_names, self.performance_statistics_df[non_pred_types].loc[0], alpha=.6, color='red',
                ec='black', label='Correctly classified: class 0')
        ax.barh(pred_col_names, self.performance_statistics_df[self.prediction_type_nos].loc[0], alpha=.6,
                color='chartreuse', ec='black', label='Correctly classified: class 1')
        #plt.rc('ytick', labelsize=15)
        #plt.rc('xtick', labelsize=15)
        ax.tick_params(axis='both', labelsize=20)
        # print(col_type_nos)
        for i, v in enumerate(col_type_nos):
            ax.text(self.performance_statistics_df[v].values[1] + 10, i - .1,
                    str(self.performance_statistics_df[v].values[0]) + "/" + str(
                        self.performance_statistics_df[v].values[1]) + " | " + str(
                        self.performance_statistics_df[v].values[2]),
                    color='blue', fontweight='bold', fontsize=10)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        plt.xlim(right=np.max(self.performance_statistics_df.loc[1].values) * 120 / 100)
        # plt.savefig('important_plots/correct_classifications_plot')
        # fig.tight_layout()
        # plt.xticks(rotation=90)
        # return fig

    def get_performance_statistics_df(self):
        """
        functions to evaluate performance for each event type
        :return: df with number of events of each type: correctly classified, total number of events of the type and
            number of events of the type in training set
        """
        prediction_stat = {}
        for i, object_id in enumerate(self.test_ob.features_df['id']):
            # print(1)
            type_no = self.test_ob.features_df['type'].values[np.where(self.test_ob.features_df['id'] == object_id)][0]
            # print(self.train_sample_numbers)
            num_training_events = self.train_sample_numbers[type_no]
            if num_training_events == 0:
                type_no = 0

            if type_no not in prediction_stat:
                prediction_stat[type_no] = [0, 1, num_training_events]
            else:
                prediction_stat[type_no][1] = prediction_stat[type_no][1] + 1

            if (type_no in self.prediction_type_nos) & (self.test_ob.features_df['y_pred'].values[i] == 1):
                prediction_stat[type_no][0] = prediction_stat[type_no][0] + 1

            elif (self.test_ob.features_df['y_pred'].values[i] == 0) & (type_no not in self.prediction_type_nos):
                prediction_stat[type_no][0] = prediction_stat[type_no][0] + 1
        stat_df = pd.DataFrame(prediction_stat)
        return stat_df.reindex(sorted(stat_df.columns), axis=1)

    def plot_confusion_matrix(self, ax, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.

        :param ax: axes on which plot is to be generated
        :param cmap: color map for plotting
        :return:
        """
        title = ""

        # Compute confusion matrix
        cm = metrics.confusion_matrix(self.test_ob.features_df['y_true'], self.test_ob.features_df['y_pred'])
        # Only use the labels that appear in the data
        classes = [0,1]

        # fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes)
        ax.set_xlabel('True label', fontsize =20)
        ax.set_ylabel('Predicted label', fontsize =20 )

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=15)
        # fig.tight_layout()
        ax.axis("equal")
        return ax

    def get_ignored_events(self):
        """
        gets event types that are ignored for during training and testing
        """
        unknown = ""
        dropped = ""
        for key in self.test_sample_numbers:
            if (key in self.train_sample_numbers.keys()) & (self.test_sample_numbers[key] != 0):
                continue
            else:
                if self.test_sample_numbers[key] == 0:
                    if dropped == "":
                        dropped = ztf_ob_type_name(key)
                    else:
                        dropped = dropped + ", " + ztf_ob_type_name(key)
                else:
                    if unknown == "":
                        unknown = ztf_ob_type_name(key)
                    else:
                        unknown = unknown + ", " + ztf_ob_type_name(key)
        if unknown == "":
            unknown = 'None'
        if dropped == "":
            dropped = 'None'
        return unknown, dropped

    def plot_roc_curve(self, fpr, tpr, roc_auc, ax=None):
        """
        function to plot roc auc

        :param fpr: false positive rate
        :param tpr: true positive rate
        :param roc_auc: roc auc
        :param ax: axes object
        :return:
        """

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.gca()
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1],'r--')

        plt.ylim([0, 1])
        plt.xlim([0, .1])
        # plt.axis("square")
        plt.ylabel('True Positive Rate',fontsize= 20)
        plt.xlabel('False Positive Rate',fontsize=20)
        # plt.gca().set_aspect("equal")

    def prediction_type_names(self):
        """
        :return: for each type number, returns the anme TODO:?????
        """
        name = ""
        for type_no in self.prediction_type_nos:
            if name == "":
                name = ztf_ob_type_name(type_no)
            else:
                name = name + ", " + ztf_ob_type_name(type_no)
        return name

    def plot_performance_statistics(self, y_score):
        """
        makes a plot with contamination statistics, roc curve, confusion matrix and train-test set metadata

        :param y_score: prediction values for test dataset
        :return: figure with the plot
        """
        fig = plt.figure(figsize=(24, 12))
        # plt.subplot2grid((12,25), (0,0), colspan=25, rowspan=1, fig = fig)
        # plt.title("performance statistics [Predict:"+self.prediction_type_names()+"]", loc = "center")
        plt.subplot2grid((12, 24), (0, 5), colspan=9, rowspan=11, fig=fig)
        self.plot_contamination_statistics(ax=plt.gca())

        fpr, tpr, thresholds = metrics.roc_curve(self.test_ob.features_df['y_true'].values, y_score[:, 1])
        roc_auc = metrics.auc(fpr, tpr)

        plt.subplot2grid((12, 24), (0, 18), rowspan=3, colspan=5, fig=fig)
        self.plot_roc_curve(fpr=fpr, tpr=tpr, roc_auc=roc_auc, ax=plt.gca())
        plt.subplot2grid((12, 24), (4, 19), rowspan=3, colspan=3, fig=fig)
        self.plot_confusion_matrix(ax=plt.gca())
        cm = metrics.confusion_matrix(self.test_ob.features_df['y_true'], self.test_ob.features_df['y_pred'])
        plt.annotate('KN correctly identified = ' + str(100 * cm[1][1] / (cm[1][0] + cm[1][1]))[:5] + "%",
                     xy=(.79, .355), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=20)
        plt.annotate('Purity = ' + str(100 * cm[1][1] / (cm[0][1] + cm[1][1]))[:5] + "%",
                     xy=(.79, .315), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=20)
        plt.annotate('Training data [class 1: ' + str(
            np.sum(self.performance_statistics_df[self.prediction_type_nos].loc[2].values)) + " | class 0:"
                     + str(np.sum(self.performance_statistics_df.loc[2].values) - np.sum(
            self.performance_statistics_df[self.prediction_type_nos].loc[2].values)) + "]",
                     xy=(.79, .275), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=20)
        # plt.annotate("Num trees ="+, )
        plt.annotate('Performance statistics [Predict:' + self.prediction_type_names() + ']',
                     xy=(.5, .1), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=30)
        unknown_events, dropped_events = self.get_ignored_events()
        plt.annotate('Dropped events: ' + dropped_events,
                     xy=(.79, .23), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=10)
        plt.annotate('Unknown events: ' + unknown_events,
                     xy=(.79, .20), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=10)
        fig.tight_layout()
        return fig
