import numpy as np
import pandas as pd
import seaborn as sns
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from sklearn.metrics import multilabel_confusion_matrix, precision_recall_curve, roc_curve, auc


class RecoderX:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(logdir=log_dir)
        self.log = ''

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.add_scalar(
            tag=tag, scalar_value=scalar_value, global_step=global_step, walltime=walltime
        )

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.writer.add_scalars(
            main_tag=main_tag,
            tag_scalar_dict=tag_scalar_dict,
            global_step=global_step,
            walltime=walltime,
        )

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        self.writer.add_image(
            tag=tag,
            img_tensor=img_tensor,
            global_step=global_step,
            walltime=walltime,
            dataformats=dataformats,
        )

    def add_graph(self, graph_profile, walltime=None):
        self.writer.add_graph(graph_profile, walltime=walltime)

    def add_histogram(self, tag, values, global_step=None):
        self.writer.add_histogram(tag, values, global_step)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        self.writer.add_figure(tag, figure, global_step=global_step, close=close, walltime=walltime)

    def export_json(self, out_file):
        self.writer.export_scalars_to_json(out_file)

    def plot_multi_confusion_matrices(
        self,
        tag,
        true_labels,
        predicted_labels,
        naming_labels,
        global_step=None,
        figsize=(20, 20),
    ):
        def _plot_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=12):
            df_cm = pd.DataFrame(
                confusion_matrix,
                index=class_names,
                columns=class_names,
            )

            true_positive = df_cm.iloc[0, 0]
            true_negative = df_cm.iloc[1, 1]
            total_samples = df_cm.values.sum()
            accuracy = (true_positive + true_negative) / total_samples

            heat_map = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
            heat_map.yaxis.set_ticklabels(
                heat_map.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize
            )
            heat_map.xaxis.set_ticklabels(
                heat_map.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize
            )
            axes.set_ylabel('True label')
            axes.set_xlabel('Predicted label')
            axes.set_title('{}: {:.2f}'.format(class_label, accuracy))

        cm = multilabel_confusion_matrix(true_labels, predicted_labels)
        fig, ax = plt.subplots(int(np.ceil(len(naming_labels) / 4)), 4, figsize=figsize, dpi=200)
        canvas = FigureCanvasAgg(fig)

        for axes, cfs_matrix, label in zip(ax.flatten(), cm, naming_labels):
            _plot_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
            fig.tight_layout()

        canvas.draw()

        image_as_string = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        target_shape = canvas.get_width_height()[::-1] + (3,)
        reshaped_image = image_as_string.reshape(target_shape)
        self.writer.add_image(tag, reshaped_image, dataformats="HWC", global_step=global_step)

    def plot_multi_precision_recall_curves(
        self,
        tag,
        true_labels,
        predicted_probabilities,
        naming_labels,
        global_step=None,
        figsize=(20, 20),
    ):
        def _plot_precision_recall_curve(prc, rec, axes, class_label):
            f1 = 2 * prc * rec / (prc + rec)
            f1_idx = np.nanargmax(f1)

            axes.plot(rec, prc)
            axes.set_ylabel('Precision')
            axes.set_xlabel('Recall')
            axes.set_title('{} - F1: {:.2f}'.format(class_label, f1[f1_idx]))

        fig, ax = plt.subplots(int(np.ceil(len(naming_labels) / 4)), 4, figsize=figsize)
        canvas = FigureCanvasAgg(fig)
        for axes, label, idx in zip(ax.flatten(), naming_labels, np.arange(len(naming_labels))):
            prc, rec, _ = precision_recall_curve(
                true_labels[:, idx], predicted_probabilities[:, idx]
            )
            _plot_precision_recall_curve(prc, rec, axes, label)
            fig.tight_layout()

        canvas.draw()

        image_as_string = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        target_shape = canvas.get_width_height()[::-1] + (3,)
        reshaped_image = image_as_string.reshape(target_shape)
        self.writer.add_image(tag, reshaped_image, dataformats="HWC", global_step=global_step)

    def plot_multi_roc_curves(
        self,
        tag,
        true_labels,
        predicted_probabilities,
        naming_labels,
        global_step=None,
        figsize=(20, 20),
    ):
        def _plot_roc_curve(fpr, tpr, axes, class_label):
            axes.plot(fpr, tpr)
            axes.set_ylabel('True Positive')
            axes.set_xlabel('False Positive')
            axes.set_title('{} - AUC: {:.2f}'.format(class_label, auc(fpr, tpr)))

        fig, ax = plt.subplots(int(np.ceil(len(naming_labels) / 4)), 4, figsize=figsize)
        canvas = FigureCanvasAgg(fig)
        for axes, label, idx in zip(ax.flatten(), naming_labels, np.arange(len(naming_labels))):
            fpr, tpr, thr = roc_curve(true_labels[:, idx], predicted_probabilities[:, idx])
            _plot_roc_curve(fpr, tpr, axes, label)
            fig.tight_layout()

        canvas.draw()

        image_as_string = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        target_shape = canvas.get_width_height()[::-1] + (3,)
        reshaped_image = image_as_string.reshape(target_shape)
        self.writer.add_image(tag, reshaped_image, dataformats="HWC", global_step=global_step)

    def close(self):
        self.writer.close()
