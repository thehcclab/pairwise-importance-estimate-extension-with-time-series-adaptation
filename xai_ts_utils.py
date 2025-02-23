import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
from sklearn.metrics import mean_squared_error
from math import sqrt
from captum.attr import DeepLift, IntegratedGradients, InputXGradient, FeatureAblation
from captum.attr import DeepLift, IntegratedGradients, InputXGradient, FeatureAblation
import seaborn as sns


def plot_timeseries(in_timeseries, title=None, channel_names = None):
    """
    Plots univariate or multivariate time series.

    Parameters:
    - in_timeseries (array_like): Time series of shape (n_channels, timeseries_length)
    - title (string, optional): Optional title of the plot
    - channel_names (list, optional): Optional list of channel names. If provided, it has to be of length n_channels

    Returns:
    no return
    """
    if len(in_timeseries.shape) == 3:
        if in_timeseries.shape[0] > 1:
            warnings.warn('This function supports only plotting of individual time series! Only the first time series will be plotted')
        timeseries = in_timeseries[0]
    else:
        timeseries = in_timeseries

    n_channels = timeseries.shape[0]
    if channel_names is not None:
        assert len(channel_names) == n_channels, 'number of channels names has to match number of channels of provided time series'

    fig_w = 20
    fig_h = n_channels * 2
    fig, axs = plt.subplots(nrows=n_channels, ncols=1, figsize=(fig_w,fig_h))
    if title is not None:
        plt.suptitle(title)

    for ch_idx in range(len(timeseries)):
        ax = axs[ch_idx]
        if channel_names is not None:
            ax.set_title(channel_names[ch_idx])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.plot(timeseries[ch_idx], color='black', linewidth=1)

    plt.tight_layout()


def plot_timeseries_with_explanation(in_timeseries, in_explanation, title=None, channel_names = None):
    """
    Plots univariate or multivariate time series with its explanation.

    Parameters:
    - in_timeseries (array_like): Time series of shape (n_channels, timeseries_length)
    - in_explanation (array_like): Explanation of shape (n_channels, timeseries_length)
    - title (string, optional): Optional title of the plot
    - channel_names (list, optional): Optional list of channel names. If provided, it has to be of length n_channels

    Returns:
    no return
    """
    if len(in_timeseries.shape) == 3:
        if in_timeseries.shape[0] > 1:
            warnings.warn('This function supports only plotting of individual time series and explanations! Only the first time series will be plotted')
        timeseries = in_timeseries[0]
    else:
        timeseries = in_timeseries

    if len(in_explanation.shape) == 3:
        if in_explanation.shape[0] > 1:
            warnings.warn('This function supports only plotting of individual time series and explanations! Only the first time series will be plotted')
        explanation = in_explanation[0]
    else:
        explanation = in_explanation

    n_channels = timeseries.shape[0]
    if channel_names is not None:
        assert len(channel_names) == n_channels, 'number of channels names has to match number of channels of provided time series'

    fig_w = 20
    fig_h = n_channels * 2

    fig, axs = plt.subplots(nrows=n_channels, ncols=1, figsize=(fig_w,fig_h))
    if title is not None:
        plt.suptitle(title, fontsize=16)

    minValue = explanation.min()
    valueExtent = abs(max(minValue, explanation.max()))
    for ch_idx in range(n_channels):
        channel_timeseries = timeseries[ch_idx]
        channel_explanation = explanation[ch_idx]
        if n_channels > 1:
            ax = axs[ch_idx]
        else:
            ax = axs

        if channel_names is not None:
            ax.set_title(channel_names[ch_idx])
        ax.tick_params(axis='both', which='major', labelsize=12)

        x_min = 0
        x_max = len(channel_timeseries)
        y_min = channel_timeseries.min() * .9
        y_max = channel_timeseries.max() * 1.1

        relevance_img = np.tile(channel_explanation, (len(channel_explanation),1) )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.plot(channel_timeseries, color='black', linewidth=1)
        ax.imshow(relevance_img, extent=[x_min-0.5, x_max-0.5, y_min, y_max], cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=-valueExtent, vmax=valueExtent)

    plt.tight_layout()



def multi_plot_timeseries(timeseries, title=None, channel_names = None):
    """
    Plots a group of univariate or multivariate time series.
    Each time series is plotted individually and transparent.
    Additioanlly, the mean time series of the group is plotted with the standard deviation.

    Parameters:
    - timeseries (array_like): Time series of shape (n_samples, n_channels, timeseries_length)
    - title (string, optional): Optional title of the plot
    - channel_names (list, optional): Optional list of channel names. If provided, it has to be of length n_channels

    Returns:
    no return
    """
    n_samples = timeseries.shape[0]
    n_channels = timeseries.shape[1]
    if channel_names is not None:
        assert len(channel_names) == n_channels, 'number of channels names has to match number of channels of provided time series'

    fig_w = 20
    fig_h = n_channels * 2
    fig, axs = plt.subplots(nrows=n_channels, ncols=1, figsize=(fig_w,fig_h))

    if title is not None:
        plt.suptitle(title, fontsize=16)

    for i in range(n_samples):
        sample = timeseries[i]
        for ch_idx, channel_data in enumerate(sample):
            ax = axs[ch_idx]
            ax.plot(channel_data, color='black', linewidth=1, alpha=0.15)
            

    for i in range(n_channels):
        ax = axs[i]
        if channel_names is not None:
            ax.set_title(channel_names[i])

        ax.tick_params(axis='both', which='major', labelsize=12)
        mean_sample = timeseries[:,i].mean(axis=0)
        std = timeseries[:,i].std(axis=0)
        std_lower = mean_sample - std
        std_upper = mean_sample + std
        ax.plot(mean_sample, color='red', linewidth=1, label='mean')
        ax.plot(std_lower, color='skyblue', linewidth=1)
        ax.plot(std_upper, color='skyblue', linewidth=1)
        
        x = np.arange(0, len(mean_sample))
        ax.fill_between(x, std_lower, std_upper, color='skyblue', alpha=0.3, label='std')
        ax.set_ylim(0,1)
    
    plt.legend()
    plt.tight_layout()




def perform_ts_xai_routine(data_x, data_y, channel_names, model, dataset_name, save_path, show_plots = True, transpose_x_predict = False):
    """
    Plots a group of univariate or multivariate time series.
    Each time series is plotted individually and transparent.
    Additioanlly, the mean time series of the group is plotted with the standard deviation.

    Parameters:
    - data_x (array_like): Set of timeseries of shape (n_samples, n_channels, timeseries_length)
    - data_y (array_like): Ground truth for data_x (n_samples, n_channels, timeseries_length)
    - channel_names (list of str): list of channel names - necessary for plotting
    - model (object): Model for which the explanations should be generated
    - dataset_name (string): name of dataset - necessary for plotting
    - save_path (string): location where the plots will be stored
    - show_plots (boolean, optional): default is True. If False, plots are not shown in notebook.
    - transpose_x_predict (boolean, optional): Relevant for LSTMs, transposes last two axis of data_x before predicting.

    Returns:
    no return
    """
    assert len(channel_names) == data_x.shape[1], 'number of channels names has to match number of channels in provided time series (data_x.shape[1])'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    data_x_tensor = torch.Tensor(data_x)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Plot distribution of ground truth
    plt.hist(data_y.reshape(-1), bins=11)
    plt.title(f'{dataset_name}\nDistribution of ground truth')
    plt.savefig(os.path.join(save_path, 'distribution - ground truth.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Plot distributions of each channel
    for ch_idx, ch_name in enumerate(channel_names):
        plt.hist(data_x[:,ch_idx,:].reshape(-1), bins=11)
        plt.title(f'{dataset_name}\nDistribution of {ch_name}')
        plt.savefig(os.path.join(save_path, 'distribution - {}.png'.format(ch_name)))
        if show_plots:
            plt.show()
        else:
            plt.close()


    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Plot groups of samples binned by ground truth (including mean+std of group)
    y_histogram_ranges = np.histogram(data_y)[1]
    data_y_flat = data_y.reshape(-1)
    for i in range(len(y_histogram_ranges)-1):
        min_val = y_histogram_ranges[i]
        max_val = y_histogram_ranges[i+1]
        if i == len(y_histogram_ranges)-2:
            max_val += .1 # just increase it by an arbitrary small value, so that all values are included
        samples_subset = data_x[(data_y_flat  >= min_val) & (data_y_flat  < max_val)]
        multi_plot_timeseries(samples_subset, '{}\n#Samples: {} - SOH range [{:.3f},{:.3f})'.format(dataset_name, len(samples_subset),min_val,max_val), channel_names)
        plt.savefig(os.path.join(save_path, 'ground_truth_range_{:.3f}-{:.3f}.png'.format(min_val, max_val)))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Make all model predictions
    if transpose_x_predict:
        predictions = model(data_x_tensor.transpose(1,2)).detach().numpy()
    else:
        predictions = model(data_x_tensor).detach().numpy()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Plot MAE 
    maes = abs(predictions - data_y).reshape(-1)
    mae_sorted_idxs = maes.argsort()

    mse = mean_squared_error(predictions, data_y)
    rmse = sqrt(mse)

    plt.hist(maes)
    plt.title(f'{dataset_name}\nDistribution of MAEs | RMSE: {rmse:.4f}')
    plt.savefig(os.path.join(save_path, 'distribution of MAEs.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Generate explanations 
    # explainer = DeepLift(model)
    # explainer = IntegratedGradients(model)
    explainer = InputXGradient(model)
    
    if transpose_x_predict:
        raw_explanations = explainer.attribute(data_x_tensor.transpose(1,2)).transpose(1,2)
    else:
        raw_explanations = explainer.attribute(data_x_tensor)
    raw_explanations = raw_explanations.detach().numpy()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Plot mean/max explanations over all samples
    mean_timeseries = data_x.mean(axis=0)
    mean_explanation = raw_explanations.mean(axis=0)
    max_explanation = raw_explanations.max(axis=0)

    explainer_name = explainer.__class__.__name__

    plot_timeseries_with_explanation(mean_timeseries, mean_explanation, f'{dataset_name}\n{explainer_name} - Mean time series and mean explanation over all samples', channel_names)
    plt.savefig(os.path.join(save_path, f'mean time series - mean explanation (all) - {explainer_name}'))
    if show_plots:
        plt.show()
    else:
        plt.close()

    plot_timeseries_with_explanation(mean_timeseries, max_explanation, f'{dataset_name}\n{explainer_name} - Mean time series and max explanation over all samples', channel_names)
    plt.savefig(os.path.join(save_path, f'mean time series - max explanation (all) - {explainer_name}'))
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Compute channel importance
    channel_importance = raw_explanations.sum(axis=(0,2))
    minValue = channel_importance.min()
    valueExtent = abs(max(minValue, channel_importance.max()))
    channel_importance = channel_importance / valueExtent
    
    plt.barh(channel_names[::-1], channel_importance[::-1])
    plt.title(f'{dataset_name}\nChannel Importance')
    plt.savefig(os.path.join(save_path, 'channel importance.png'), bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Bin explanations based on SOH (5 bins)
    y_histogram_ranges = np.histogram(data_y, bins=5)[1]
    data_y_flat = data_y.reshape(-1)
    for i in range(len(y_histogram_ranges)-1):
        min_val = y_histogram_ranges[i]
        max_val = y_histogram_ranges[i+1]
        if i == len(y_histogram_ranges)-2:
            max_val += .1 # just increase it by an arbitrary small value, so that all values are included

        samples_subset_mean = data_x[(data_y_flat  >= min_val) & (data_y_flat  < max_val)].mean(axis=0)
        raw_explanations_subset = raw_explanations[(data_y_flat  >= min_val) & (data_y_flat  < max_val)]
        subset_mean_explanation = raw_explanations_subset.mean(axis=0)
        subset_max_explanation = raw_explanations_subset.max(axis=0)

        plot_timeseries_with_explanation(samples_subset_mean, subset_mean_explanation, f'{dataset_name}\n{explainer_name} - Mean time series and mean explanation - #Samples: {len(raw_explanations_subset)} - SOH range [{min_val:.3f},{max_val:.3f})', channel_names)
        plt.savefig(os.path.join(save_path, f'mean time series - mean explanation (SOH range [{min_val:.3f},{max_val:.3f})) - {explainer_name}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

        plot_timeseries_with_explanation(samples_subset_mean, subset_max_explanation, f'{dataset_name}\n{explainer_name} - Mean time series and max explanation - #Samples: {len(raw_explanations_subset)} - SOH range [{min_val:.3f},{max_val:.3f})', channel_names)
        plt.savefig(os.path.join(save_path, f'mean time series - max explanation (SOH range [{min_val:.3f},{max_val:.3f})) - {explainer_name}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()







################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
            
def create_explanation_heatmap(explanation, channel_labels, title, savepath, xticklabels=None, show_plots=True):
    minValue = explanation.min()
    valueExtent = abs(max(minValue, explanation.max()))

    plt.figure(figsize=(24, len(channel_labels) // 4))
    # sns.heatmap(explanation, cmap='coolwarm', cbar=True, xticklabels=np.arange(explanation.shape[1]), yticklabels=channel_labels, vmin=-valueExtent, vmax=valueExtent)
    if xticklabels is None:
        xticklabels = np.arange(45,71)

    if minValue >= 0:
        sns.heatmap(explanation, cmap='Reds', cbar=True, xticklabels=xticklabels, yticklabels=channel_labels, vmin=0, vmax=valueExtent)
    else:
        sns.heatmap(explanation, cmap='coolwarm', cbar=True, xticklabels=xticklabels, yticklabels=channel_labels, vmin=-valueExtent, vmax=valueExtent)
        
    plt.xlabel('Time points')
    plt.ylabel('Channels')
    plt.title(title)
    plt.gcf().set_facecolor('white')
    plt.savefig(savepath, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
            
            
def create_timepoint_relevance_plot(explanation, title, savepath, xticklabels=None, show_plots=True, supporting_timeseries = None):
    plt.figure(figsize=(15, 4))
    timepoint_mean = explanation.mean(axis=0)
    # timepoint_mean = np.median(explanation, axis=0)
    # minValue = timepoint_mean.min()
    # valueExtent = abs(max(minValue, timepoint_mean.max()))
    timepoint_mean -= timepoint_mean.min()
    timepoint_mean /= timepoint_mean.max()

    # plt.plot(timepoint_mean)
    # plt.bar(np.arange(len(timepoint_mean)), timepoint_mean)
    if xticklabels is None:
        xticklabels = np.arange(45,71)

    xticklabels = [str(num) for num in xticklabels]
    
    plt.bar(xticklabels, timepoint_mean, width=.9, label='time point importance', color='C0')
    plt.xticks(xticklabels[::7], rotation=90, fontsize=18)
    plt.yticks(fontsize=14)
    # plt.xticks(rotation=90, fontsize=14)
    plt.xlabel('time (s)', fontsize=18)
    # plt.ylabel('Mean relevance')
    plt.xlim(-1, len(xticklabels)+1)
    plt.ylabel('normalized relevance', fontsize=18, color='C0')

    plt.title(title, fontsize=20)

    lines, labels = plt.gca().get_legend_handles_labels()

    if supporting_timeseries is not None:
        plt.ylim(0,1.05)
        plt.yticks(color='C0')

        plt.twinx()
        plt.yticks(color='orange', fontsize=14)
        plt.ylabel('amplitude ($\mu$V)', fontsize=18, color='orange')
        plt.plot(supporting_timeseries, color='orange', label='error-related potential', linewidth=3)
        
        if '0.0' in xticklabels:
            plt.axvline(x=xticklabels.index('0.0'), color='red', label='error onset')
        plt.ylim(-2,2)
        lines2, labels2 = plt.gca().get_legend_handles_labels()
        lines += lines2
        labels += labels2

    plt.legend(lines, labels, loc='upper left', fontsize=16)
    plt.gcf().set_facecolor('white')
    plt.savefig(savepath, bbox_inches='tight')
    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.close()

    return timepoint_mean


def get_mean_class_probability(data_x, data_y, class_id, model, device):
    target_set_x = data_x[data_y == class_id].copy()
    data_x_tensor = torch.tensor(target_set_x).to(device)

    predictions = model(data_x_tensor)
    ps = torch.exp(predictions)
    mean_probability = float(ps[:,class_id].mean())
    return mean_probability


def get_xai_summary_per_subset(data_x, data_y, channel_labels, class_id, subset_name, model, results_directory, device, xticklabels = None, filter_samples=True, cut_start=0, cut_end=None, show_plots=True, supporting_timeseries=None):
    os.makedirs(results_directory, exist_ok=True)

    subset_x = data_x[data_y == class_id]

    # ------------------------------  Get data, make predictions and generate explanations ------------------------------
    target_set_x = subset_x.copy()
    data_x_tensor = torch.tensor(target_set_x).to(device)

    predictions = model(data_x_tensor)
    ps = torch.exp(predictions)
    top_ps, top_class = ps.topk(1, dim=1)
    all_predictions = top_class.cpu().detach().numpy().reshape(-1)
    # print(f'  - Mean probability for class {class_id}: {ps[:,class_id].mean():.3f}')

    data_x_tensor = data_x_tensor[all_predictions == class_id]

    # print('data_x_tensor.shape: ', data_x_tensor.shape)
    # explainer = DeepLift(model)
    explainer = InputXGradient(model)
    explainer_name = 'DeepLift'

    tpr = len(data_x_tensor)/len(subset_x)
    print(f'  - TPR: {tpr:.3f}')
    # if tpr < .5:
    #     raise Exception
    # return np.zeros_like(subset_x), np.zeros_like(subset_x), np.zeros_like(subset_x)

    class_explanations = np.zeros(data_x_tensor.shape)
    bs = 1000
    for i in range(0, len(data_x_tensor), bs):
        class_explanations[i:i+bs] = explainer.attribute(data_x_tensor[i:i+bs], target=class_id).cpu().detach().numpy()
    # print('class_explanations.shape: ', class_explanations.shape)

    class_label = f'class_{class_id}'
    # class_samples = target_set_x[all_predictions == class_id]
    # class_explanations = raw_explanations[all_predictions == class_id]

    
    # print(f'#explanations: {len(class_explanations)}')

    # Compute aggregated time series and explanations
    # mean_explanation = class_explanations.mean(axis=0)
    # max_explanation = class_explanations.max(axis=0)

    pos_explanation = class_explanations.copy()
    pos_explanation = pos_explanation[:,:,cut_start:cut_end]
    xticklabels = xticklabels[cut_start:cut_end]
    # TODO: Normalize per explanation!
    pos_explanation[pos_explanation < 0] = 0
    for i in range(len(pos_explanation)):
        pos_explanation[i] -= pos_explanation[i].min()
        pos_explanation[i] /= pos_explanation[i].max()


    meanpos_explanation = pos_explanation.mean(axis=0)
    #minmax
    meanpos_explanation -= meanpos_explanation.min() # min max
    meanpos_explanation /= meanpos_explanation.max()


    # # ------------------------------  Mean explanation ------------------------------
    # fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations mean"
    # fig_savepath = os.path.join(results_directory, f'explanations mean - {subset_name} - {class_label} - {explainer_name}.png')
    # create_explanation_heatmap(mean_explanation, channel_labels, fig_title, fig_savepath, xticklabels=xticklabels)

    # # ------------------------------  Mean explanation - overall time point relevance ------------------------------
    # fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations mean - time point relevace"
    # fig_savepath = os.path.join(results_directory, f'explanations mean - time point relevace - {subset_name} - {class_label} - {explainer_name}.png')
    # create_timepoint_relevance_plot(mean_explanation, fig_title, fig_savepath, xticklabels=xticklabels)

    # ------------------------------  Mean pos explanation ------------------------------
    fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations mean pos"
    fig_savepath = os.path.join(results_directory, f'explanations mean pos - {subset_name} - {class_label} - {explainer_name}.png')
    create_explanation_heatmap(meanpos_explanation, channel_labels, fig_title, fig_savepath, xticklabels=xticklabels, show_plots=show_plots)

    # ------------------------------  Mean pos  explanation - overall time point relevance ------------------------------
    fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations mean pos- time point relevace"
    fig_savepath = os.path.join(results_directory, f'explanations mean pos - time point relevace - {subset_name} - {class_label} - {explainer_name}.png')
    time_point_relevance = create_timepoint_relevance_plot(meanpos_explanation, fig_title, fig_savepath, xticklabels=xticklabels, show_plots=show_plots, supporting_timeseries=supporting_timeseries[cut_start:cut_end])

    # # ------------------------------  Max explanation ------------------------------
    # fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations max"
    # fig_savepath = os.path.join(results_directory, f'explanations max - {subset_name} - {class_label} - {explainer_name}.png')
    # create_explanation_heatmap(max_explanation, channel_labels, fig_title, fig_savepath, xticklabels=xticklabels)

    # fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations max - time point relevace"
    # fig_savepath = os.path.join(results_directory, f'explanations max - time point relevace - {subset_name} - {class_label} - {explainer_name}.png')
    # create_timepoint_relevance_plot(max_explanation, fig_title, fig_savepath, xticklabels=xticklabels)

    
    # ------------------------------  channel importance  ------------------------------
    # channel_importance = class_explanations.sum(axis=(0,2))
    channel_importance = meanpos_explanation.sum(axis=1)
    # minValue = channel_importance.min()
    # valueExtent = abs(max(minValue, channel_importance.max()))

    # channel_importance = channel_importance / valueExtent
    channel_importance -= channel_importance.min()
    channel_importance /= channel_importance.max()

    pos_explanation = class_explanations.copy()
    pos_explanation = pos_explanation[:,:,cut_start:cut_end]
    pos_explanation[pos_explanation < 0] = 0
    channel_importances = []
    for i in range(len(pos_explanation)):
        # pos_explanation[i] /= pos_explanation[i].sum()
        ch_importance = pos_explanation[i].sum(axis=1)
        # ch_importance -= ch_importance.min()
        # ch_importance /= ch_importance.max()
        channel_importances.append(ch_importance)
    channel_importances = np.array(channel_importances)
    channel_importance = channel_importances.mean(axis=0)
    channel_importance -= channel_importance.min()
    channel_importance /= channel_importance.max()

    plt.figure(figsize=(5, len(channel_labels) // 5))
    plt.barh(channel_labels[::-1], channel_importance[::-1])
    plt.title(f'{subset_name} - {class_label} - {explainer_name}\n Channel Importance')
    plt.gcf().set_facecolor('white')
    plt.savefig(os.path.join(results_directory, f'channel importance - {subset_name} - {class_label} - {explainer_name}.png'), bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    return meanpos_explanation, time_point_relevance, channel_importance



def get_xai_summary_per_subset_test(data_x, data_y, channel_labels, class_id, subset_name, model, results_directory, device, xticklabels = None, filter_samples=True, cut_start=0, cut_end=None, show_plots=True):
    os.makedirs(results_directory, exist_ok=True)

    # if filter_samples:
    subset_x = data_x[data_y == class_id]
    # else:
    #     subset_x = data_x

    # ------------------------------  Get data, make predictions and generate explanations ------------------------------
    target_set_x = subset_x.copy()
    data_x_tensor = torch.tensor(target_set_x).to(device)

    predictions = model(data_x_tensor)
    ps = torch.exp(predictions)
    top_ps, top_class = ps.topk(1, dim=1)
    all_predictions = top_class.cpu().detach().numpy().reshape(-1)
    # print(f'  - Mean probability for class {class_id}: {ps[:,class_id].mean():.3f}')

    if filter_samples:
        data_x_tensor = data_x_tensor[all_predictions == class_id]

    # print('data_x_tensor.shape: ', data_x_tensor.shape)
    explainer = DeepLift(model)
    explainer_name = explainer.__class__.__name__

    tpr = len(data_x_tensor)/len(subset_x)
    # print(f'  - TPR: {tpr:.3f}')
    # if tpr < .5:
    #     raise Exception
    # return np.zeros_like(subset_x), np.zeros_like(subset_x), np.zeros_like(subset_x)

    class_explanations = np.zeros(data_x_tensor.shape)
    bs = 1000
    for i in range(0, len(data_x_tensor), bs):
        class_explanations[i:i+bs] = explainer.attribute(data_x_tensor[i:i+bs], target=class_id).cpu().detach().numpy()
    # print('class_explanations.shape: ', class_explanations.shape)

    class_label = f'class_{class_id}'
    # class_samples = target_set_x[all_predictions == class_id]
    # class_explanations = raw_explanations[all_predictions == class_id]

    
    # print(f'#explanations: {len(class_explanations)}')

    # Compute aggregated time series and explanations
    # mean_explanation = class_explanations.mean(axis=0)
    # max_explanation = class_explanations.max(axis=0)

    pos_explanation = class_explanations.copy()
    pos_explanation = pos_explanation[:,:,cut_start:cut_end]
    xticklabels = xticklabels[cut_start:cut_end]
    # TODO: Normalize per explanation!
    pos_explanation[pos_explanation < 0] = 0
    ##################################### STOP AFTER THIS APPEARS #######################################
    channel_importances = []
    for i in range(len(pos_explanation)):
        pos_explanation[i] /= pos_explanation[i].sum()
        ch_importance = pos_explanation[i].sum(axis=1)
        # ch_importance -= ch_importance.min()
        # ch_importance /= ch_importance.max()
        channel_importances.append(ch_importance)
    channel_importances = np.array(channel_importances)

    return None, None, channel_importances.mean(axis=0)


    #     # print(i)
    #     # print(f' - Probability                   : {ps[i,class_id]:.2f}')
    #     # print(f' - Explanation sum               : {class_explanations[i].sum():.2f}')
    #     # print(f' - Explanation pos sum           : {pos_explanation[i].sum():.2f}')
    #     pos_explanation[i] /= pos_explanation[i].sum() # normalize total relevance to 1
    #     # pos_explanation[i] *= float(ps[i,class_id])



        # print(f' - Explanation pos sum after norm: {pos_explanation[i].sum():.2f}')
    # for i in range(len(pos_explanation)):
    #     pos_explanation
        # pos_explanation[i] -= pos_explanation[i].min()
        # pos_explanation[i] /= pos_explanation[i].max()

    # meanpos_explanation = pos_explanation.mean(axis=0)
    #minmax
    # meanpos_explanation -= meanpos_explanation.min() # min max
    # meanpos_explanation /= meanpos_explanation.max()


    # # ------------------------------  Mean explanation ------------------------------
    # fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations mean"
    # fig_savepath = os.path.join(results_directory, f'explanations mean - {subset_name} - {class_label} - {explainer_name}.png')
    # create_explanation_heatmap(mean_explanation, channel_labels, fig_title, fig_savepath, xticklabels=xticklabels)

    # # ------------------------------  Mean explanation - overall time point relevance ------------------------------
    # fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations mean - time point relevace"
    # fig_savepath = os.path.join(results_directory, f'explanations mean - time point relevace - {subset_name} - {class_label} - {explainer_name}.png')
    # create_timepoint_relevance_plot(mean_explanation, fig_title, fig_savepath, xticklabels=xticklabels)

    # ------------------------------  Mean pos explanation ------------------------------
    # fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations mean pos"
    # fig_savepath = os.path.join(results_directory, f'explanations mean pos - {subset_name} - {class_label} - {explainer_name}.png')
    # create_explanation_heatmap(meanpos_explanation, channel_labels, fig_title, fig_savepath, xticklabels=xticklabels, show_plots=True)

    # ------------------------------  Mean pos  explanation - overall time point relevance ------------------------------
    # fig_title = f"{subset_name} - {class_label} - {explainer_name} - explanations mean pos- time point relevace"
    # fig_savepath = os.path.join(results_directory, f'explanations mean pos - time point relevace - {subset_name} - {class_label} - {explainer_name}.png')
    # time_point_relevance = create_timepoint_relevance_plot(meanpos_explanation, fig_title, fig_savepath, xticklabels=xticklabels, show_plots=True)


    
    # ------------------------------  channel importance  ------------------------------
    # channel_importance = class_explanations.sum(axis=(0,2))
    channel_importance = meanpos_explanation.sum(axis=1)
    channel_importance -= channel_importance.min()
    channel_importance /= channel_importance.max()

    # minValue = channel_importance.min()
    # valueExtent = abs(max(minValue, channel_importance.max()))

    # channel_importance = channel_importance / valueExtent
    # channel_importance -= channel_importance.min()
    # channel_importance /= channel_importance.max()

    # plt.figure(figsize=(5, len(channel_labels) // 5))
    # plt.barh(channel_labels[::-1], channel_importance[::-1])
    # plt.title(f'{subset_name} - {class_label} - {explainer_name}\n Channel Importance')
    # plt.gcf().set_facecolor('white')
    # plt.savefig(os.path.join(results_directory, f'channel importance - {subset_name} - {class_label} - {explainer_name}.png'), bbox_inches='tight')
    # plt.show()

    return None, None, channel_importance