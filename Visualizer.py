import numpy as np
import seaborn as sns
import matplotlib as mpl
from statistics import median
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from Processor import Processor


class Visualizer(Processor):

    def __init__(self, sec_hz, *args, sec_pre_trial=None, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.sec_hz = sec_hz
        if sec_pre_trial:
            self.sec_pre_trial = sec_pre_trial
        try:
            self.frames_pre_trial = int(self.sec_hz * self.sec_pre_trial)
            self.cue_duration = int(self.sec_hz * self.cue_duration)
        except TypeError:
            print('No value for sec_hz')

    def visualize_heatmaps(
            self, rows, columns, responses, names, title, fig_location, x_lines=None, y_lines=None, wide_edge=False,
            intervals=2):
        # heatmap_max = min([plot.max().max() for plot in responses if len(plot) > 0]) + \
        #               (min([plot.max().max() for plot in responses if len(plot) > 0])) * 0.2
        # heatmap_min = max([plot.min().min() for plot in responses if len(plot) > 0]) * 0.8

        heatmap_max = 0.8
        heatmap_min = -0.8

        fig = plt.figure(figsize=(18, 12))
        heatmap = gridspec.GridSpec(rows, columns)
        plt.suptitle(title, fontsize=18)

        for i in range(rows * columns):
            if i < len(responses):
                fig.add_subplot(heatmap[i])
                if i in [columns * j - 1 for j in range(1, rows + 1)]:
                    self.visualize_mean_cells_response(
                        responses[i], names[i], heatmap_min, heatmap_max, True, x_lines=x_lines, y_lines=y_lines,
                        intervals=intervals)
                else:
                    self.visualize_mean_cells_response(
                        responses[i], names[i], heatmap_min, heatmap_max, x_lines=x_lines, y_lines=y_lines,
                        intervals=intervals)

        plt.show()
        # plt.savefig(fig_location)
        # plt.close()

    def plot_trials_activity(self, plot, title):
        if len(plot) == 0:
            pass
        else:
            len_trials = median([sum(~np.isnan(plot.iloc[i])) for i in range(len(plot))])
            plot_mean = plot.loc[:, :len_trials].mean(axis=0)
            plot_sd = plot.loc[:, :len_trials].std(axis=0) / np.sqrt(len(plot_mean))
            plt.plot(plot_mean)
            plt.fill_between(
                range(len(plot_mean)), plot_mean - plot_sd, plot_mean + plot_sd, alpha=0.1)
            plot_max_ate = plot_mean.min() - abs(plot_mean.min() * 0.2)
            plot_min_ate = plot_mean.max() + abs(plot_mean.max() * 0.2)
            plt.ylim(plot_max_ate, plot_min_ate)
            self.set_xticks(len_trials)
            plt.title(title)

    def set_xticks(self, len_plot, x_lines=None, intervals=60):
        x_spaces = int(intervals / self.sec_hz)
        if x_lines:
            zero_sec = int(x_lines[0] / self.sec_hz)
        elif type(x_lines) == list:
            zero_sec = 0
        else:
            zero_sec = self.sec_pre_trial
        number_of_steps = np.arange(0, len_plot + 1, intervals)
        plt.xticks(number_of_steps, np.arange(-zero_sec, len(number_of_steps) * x_spaces - zero_sec, x_spaces))

    def visualize_mean_cells_response(
            self, plot, title, heatmap_min, heatmap_max, special=False, x_lines=None, y_lines=None, intervals=60):
        plt.title(title, fontsize=10)
        if len(plot) > 0:
            color = sns.diverging_palette(255, 10, sep=100, n=50, as_cmap=True)
            norm = TwoSlopeNorm(vmin=heatmap_min, vcenter=0, vmax=heatmap_max)
            median_length = int(plot.notna().sum(axis=1).median())
            plot = plot.iloc[:, :median_length]

            if special:
                # sns.heatmap(plot, norm=norm, cmap="coolwarm", vmin=heatmap_min, vmax=heatmap_max)
                sns.heatmap(plot, norm=norm, cmap=color, vmin=heatmap_min, vmax=heatmap_max)
            else:
                # sns.heatmap(plot, norm=norm, cmap="coolwarm", cbar=False, vmin=heatmap_min, vmax=heatmap_max)
                sns.heatmap(plot, norm=norm, cmap=color, cbar=False, vmin=heatmap_min, vmax=heatmap_max)
            plt.yticks(np.arange(0, len(plot), 50), np.arange(0, len(plot), 50))
            self.set_xticks(len(plot.iloc[0]), x_lines, intervals)

            if not x_lines and type(x_lines) != list:
                plt.axvline(x=int(self.sec_hz * 2), linewidth=1, color='black', linestyle='--')
                plt.axvline(x=int(self.sec_hz * 4), linewidth=1, color='black', linestyle='--')
            else:
                for i in x_lines:
                    plt.axvline(x=i, linewidth=1, color='black', linestyle='--')
            if y_lines is not None:
                for i in y_lines:
                    plt.axhline(y=i, linewidth=1, color='black', linestyle='--')

    def scatter_plot_config(self, pair, days_names, title, colors='black', additional_title=None, include_slope=None):
        if len(pair[0]) > 0 and len(pair[1]) > 0:
            corr = round(np.corrcoef(pair[0], pair[1])[0, 1], 3)
            plt.scatter(pair[0], pair[1], c=colors)
            x_array = np.array(pair[0])
            y_array = np.array(pair[1])
            slope, intercept = np.polyfit(x_array, y_array, 1)
            plt.plot(x_array, slope * x_array + intercept, '--', color='red')

            plt.xlabel(days_names[0], size=20)
            plt.ylabel(days_names[1], size=20)
            max_value = max([max(pair[0]), max(pair[1]), abs(min(pair[0])), abs(min(pair[1]))])
            plt.xlim(-max_value, max_value)
            plt.ylim(-max_value, max_value)
            plt.axvline(x=0, linewidth=1, color='black', linestyle='-')
            plt.axhline(y=0, linewidth=1, color='black', linestyle='-')
            plt.gca().set_aspect('equal', adjustable='box')
            if additional_title:
                plt.title(f'{title}: {corr} {additional_title}', size=20)
            elif include_slope:
                plt.title(f'{title}: {corr}, slope: {round(slope, 2)}', size=20)
            else:
                plt.title(f'{title}: {corr}', size=20)
        else:
            plt.title(title, size=20)

    def config_hist_percentages(self, mi, label, color, y_value, n_cells, n_bins=20):
        if len(mi) == 0:
            return None
        stable_bins = np.linspace(-1, 1, n_bins + 1, endpoint=True)
        counts, bins = np.histogram(mi, bins=stable_bins)
        percentages = counts / n_cells * 100
        plt.bar(np.linspace(-1, 1, n_bins), percentages, width=np.diff(bins), align="edge", alpha=0.5, label=label,
                color=color)
        if y_value is not None:
            plt.ylim(0, y_value)
        plt.xlim(-1.1, 1.1)
        return percentages

    def highest_hist(self, arrays, n_cells=False, n_bins=20):
        stable_bins = np.linspace(-1, 1, n_bins + 1)
        max_perc = []

        for array in arrays:
            if len(array):
                counts, bins = np.histogram(array, bins=stable_bins)
                if not n_cells:
                    percentages = counts / len(array) * 100
                else:
                    percentages = counts / n_cells * 100
                max_perc.append(percentages.max())

        max_value = max(max_perc)
        return max_value + 5

    @staticmethod
    def cbar_set_up(fig, max_value, min_value, color=sns.cm.rocket, title=None, location=[0.03, 0.11, 0.02, 0.77]):
        ax_cb = fig.add_axes(location)
        cb = mpl.colorbar.ColorbarBase(ax_cb,
                                       cmap=color,
                                       norm=mpl.colors.Normalize(vmin=min_value, vmax=max_value),
                                       orientation='vertical')
        if title:
            plt.ylabel('Normalized dF/F (Z-scores)')
        cb.outline.set_visible(False)


