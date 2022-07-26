"""
Utility functions to benchmark resources on known data
"""

from statistics import mean
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
import decoupler as dc
from .utils_calibrated_metrics import average_precision as calibrated_average_precision

from sklearn.metrics import roc_auc_score, average_precision_score
from numpy.random import default_rng
from tqdm import tqdm
import matplotlib.pyplot as plt

def random_scores_GT(nexp=50, ncol = 4):
    """
    Generate random scores and groud-truth matrix, for testing

    Args:
        nexp (int, optional): Number of rows/experiments. Defaults to 50.
        ncol (int, optional): Number of classes/TFs/pathways. Defaults to 4.

    Returns:
        _type_: (DataFrame): Dataframe with scores and associated ground truth
    """
    df = np.random.randn(nexp, ncol)
    ind = np.random.randint(0, df.shape[1] ,size=df.shape[0])
    gt = np.zeros(df.shape)
    gt[range(gt.shape[0]),ind] = 1

    return pd.DataFrame(np.column_stack((df.flatten(), gt.flatten())), columns = ['score', 'GT'])

"""
Downsample ground truth vector 
"""

def down_sampling(y, seed=7, n = 100):
    """
    Downsampling of ground truth
    
    Parameters
    ----------
    
    y: array
        binary groundtruth vector 
        
    seed: arbitrary seed for random sampling
    
    n: number of iterations
        
    Returns
    -------
    ds_msk: list of downsampling masks for input vectors
    """
    
    msk = []
    rng = default_rng(seed)
    
    # Downsampling
    for i in range(n):
        tn_index = np.where(y == 0)[0]
        tp_index = np.where(y != 0)[0]
        ds_array = rng.choice(tn_index, size=len(tp_index), replace=True)
        ds_msk = np.hstack([ds_array, tp_index])

        msk.append(ds_msk)

    return msk

"""
Compute AUC of ROC or PRC
"""

def get_auc(x, y, mode, pi0 = None):
    """
    Computes AUROC for each label
    
    Parameters
    ----------
    
    x: array
        binary groundtruth vector
        
    y: array (flattened)
        vector of continuous values

    pi0: float
        Reference ratio for calibrated metrics. Corresponds to the baseline/reference class inbalance to which to set the metric. Defaults to None.
        
        
    Returns
    -------
    auc: value of auc
    """

    if mode == "roc":
        auc = roc_auc_score(x, y) 
    elif mode == "prc":
        auc = average_precision_score(x, y)
    elif mode == 'calprc':
        auc = calibrated_average_precision(x, y_pred= y, pi0=pi0)
    elif mode == 'ci':
        auc = x.sum()/x.size
    else: 
        raise ValueError("mode can only be roc or prc")
    return auc

def get_source_masks(long_data, sources, subset = None, min_exp = 5):
    """
    Generates a list of indices of a DataFrame that correspond to prediction scores and associated ground-truth for each source

    Args:
        long_data (DataFrame): DataFrame with a 'score' and 'GT' column.
        sources (list): List of sources (have to be in correct order) for which there are entries in long_data.
        subset (list, optional): A subset of the sources for which to make masks. If None, then the masks will be made for all targets. Defaults to None.
        min_exp (int, optional): The minimum number of perturbation experiments required to compute an individual source performance score. Defaults to 5.

    Returns:
        target_ind: List of data indices for each target in the targets object
        target_names : Target name corresponding to the elements in target_ind. Elements in subset are filtered and put in same order as in targets.
    """

    if long_data.shape[0] < len(sources):
        raise ValueError('The data given is smaller than the number of targets')
    elif long_data.shape[0] % len(sources) != 0:
        raise ValueError('The data is likely misshapen: the number of rows cannot be divided by the number of targets')

    if subset is not None:
        iterateover = np.argwhere(np.in1d(sources, subset)).flatten().tolist()
        source_names = [sources[i] for i in iterateover]
    else:
        iterateover = range(len(sources))
        source_names = sources

    source_ind = []
    names = []
    #create indexes
    for id, name in zip(iterateover, source_names):
        ind = np.arange(id, long_data.shape[0], len(sources))
        # check that there is at least min_exp perturbations
        if long_data.iloc[ind,:]['GT'].sum() >= min_exp:
            source_ind.append(ind)
            names.append(name)

    return source_ind, names

def get_performance(data, metric = 'roc', n_iter = 100, seed = 42, prefix = None, pi0 = 0.5):
    """
    Compute binary classifier performance

    Args:
        data (DataFrame): DataFrame with a 'score' and 'GT' column. 'GT' column contains groud-truth ( e.g. 0, 1). 'score' can be continuous or same as 'GT'
        metric (str or list of str, optional): Which metric(s) to use. Currently implemeted methods are: 'mcroc', 'mcprc', 'roc', 'prc', 'calprc' and 'ci'. Defaults to 'roc'.
        n_iter (int, optional): Number of iterations for the undersampling procedures for the 'mcroc' and 'mcprc' metrics. Defaults to 100.
        seed (int, optional): Seed used to generate the undersampling for the 'mcroc' and 'mcprc' metrics. Defaults to 42.
        prefix (str, optional): Added as prefix to the performance metric key in the output dictionary e.g. 'mlm_roc' if equal to 'mlm'. Defaults to None.
        pi0 (float, optional): Reference ratio used to calculate calibrated metrics. Must be between 0 and 1. Defaults to 0.5.

    Returns:
        perf: Dict of prediction performance(s) on the given data. 'mcroc' and 'mcprc' metrics will return the values for each sampling. Other methods return a single value.
    """

    available_metrics = ['mcroc', 'mcprc', 'roc', 'prc', 'calprc', 'ci']
    metrics = [available_metrics[i] for i in np.argwhere(np.in1d(available_metrics, metric)).flatten().tolist()]

    if len(metrics) == 0:
        raise ValueError('None of the performance metrics given as parameter have been implemented')

    if 'mcroc' in metrics or 'mcprc' in metrics:
        masks = down_sampling(y = data['GT'].values, seed=seed, n=n_iter)

    perf = {}
    for met in metrics:
        if met == 'mcroc' or met == 'mcprc':
            # Compute AUC for each mask (with equalised class priors)
            aucs = []
            for mask in masks:
                auc = get_auc(x = data['GT'][mask],
                            y = data['score'][mask],
                            mode = met[2:])
                aucs.append(auc)

        elif met == 'roc' or met == 'prc' or met == 'calprc' or met == 'ci':
            # Compute AUC on the whole (unequalised class priors) data
            aucs = get_auc(x = data['GT'], y = data['score'], mode = met, pi0 = pi0)
        
        if prefix is None:
            perf[met] = aucs
        else:
            perf[prefix + '_' + met] = aucs

    return perf

def get_source_performance(data, sources, metric='roc', subset = None, n_iter = 100, seed = 42, prefix = None, pi0 = 0.5, min_exp = 5):
    """
    Compute binary classifier performance for each source or susbet of sources

    Args:
        data (DataFrame): DataFrame with a 'score' and 'GT' column. 'GT' column contains groud-truth ( e.g. 0, 1). 'score' can be continuous or same as 'GT'
        targets (list of str): List of targets (have to be in correct order) for which there are entries in data.
        metric (str, or list of str optional): Which metrics to use. Currently implemeted methods are: 'mcroc', 'mcprc', 'roc', 'prc'. Defaults to 'mcroc'.
        subset (list of str, optional): A subset of the targets for which to compute performance. If None, then the performance will be calculated for all targets. Defaults to None.
        n_iter (int, optional): Number of iterations for the undersampling procedures for the 'mcroc' and 'mcprc' metrics. Defaults to 100.
        seed (int, optional):  Seed used to generate the undersampling for the 'mcroc' and 'mcprc' metrics. Defaults to 42.
        prefix (str, optional): Added as prefix to the output dictionary. Defaults to 100.
        pi0 (float, optional): Reference ratio for calibrated metrics. Corresponds to the baseline/reference class inbalance to which to set the metric. Should be between 0 and 1. Defaults to 0.5.
        min_exp (int, optional): Minium number of perturbation experiments required to compute performance scores. Defaults to 5.

    Returns:
        perf : Dict of prediction performance(s) for each target or subset of targets. 'mcroc' and 'mcprc' metrics will return the values for each sampling. Other methods return a single value.
    """

    masks, source_names = get_source_masks(data, sources, subset = subset, min_exp = min_exp)

    perf = {}
    for src, name in zip(masks, source_names):
        if prefix is None:
            p = name
        else:
            p = prefix  + '_' + name

        perf.update(get_performance(data.iloc[src.tolist()].reset_index(), metric, n_iter, seed, prefix = p, pi0=pi0))

    return perf

def get_meta_masks(flat_data, metadata, column, min_exp = 5):

    items = np.sort(metadata[column].unique())
    n_features = flat_data.shape[0] / metadata.shape[0]

    indexes = []
    names = []
    for item in items:
        #get row numbers of all experiments corresponding to this level
        ids = np.flatnonzero(metadata[column] == item)
        if len(ids) >= min_exp:
            #find indexes of flattened array from row number in original array
            indexes.append(np.concatenate([np.arange(id * n_features, (id * n_features) + n_features, step = 1) for id in ids]))
            names.append(str(item))
        
    return indexes, names

def get_meta_performance(flat_data, metadata, columns, metric= 'roc', n_iter = 100, seed = 7, prefix= None, pi0=0.5, min_exp = 5):

    if type(columns) != list:
        columns = [columns]

    #check that columns are in metadata
    columns = list(set(metadata.columns).intersection(set(columns)))

    if len(columns) == 0:
        raise ValueError('None of the columns are in the metadata')

    perf = {}
    for column in columns:

        masks, names = get_meta_masks(flat_data, metadata, column = column, min_exp = min_exp)

        for slice, name in zip(masks, names):
            name = column + ':' + name
            if prefix is None:
                p = name
            else:
                p = prefix  + '_' + name

            perf.update(get_performance(flat_data.iloc[slice.tolist()].reset_index(), metric, n_iter, seed, prefix = p, pi0=pi0))

    return perf

def get_scores_GT(decoupler_results, metadata, by = None, min_exp = 5):
    """

    Convert decouple output to flattenend vectors and combine with GT information

    Args:
        decoupler_results (dict): Output of decouple
        metadata (DataFrame): Metadata of the perturbation experiment containing the activated/inhibited targets and the sign of the perturbation
        meta_perturbation_col (str, optional): Column name in the metadata with perturbation targets. Defaults to 'target'.
        by (str or list of str, optional): How to decompose performance. By 'sign' will also subselect scores of activating/inhibiting perturbations specifically, in addition to the overall scores. Defaults to None.
        min_exp (int, optional): Min number of perturbation experiments in order to compute score. Defaults to 5.


    Returns:
        scores_gt: dict of flattenend dataframes for each method
        targets: dict with target names for which activities were inferred for each method respectively
        meta: filtered metadata, based on decoupler output
    """
    computed_methods = list(set([i.split('_')[0] for i in decoupler_results.keys()])) # get the methods that were able to be computed (filtering of methods done by decouple)
    scores_gt = {}
    targets = {}
    metadatas = {}

    for m in computed_methods:
        # estimates = res[m + 'estimate']
        # pvals = res[m + 'pvals']

        # remove experiments with no prediction for the perturbed TFs
        missing = list(set(metadata['source']) - set(decoupler_results[m + '_estimate'].columns))
        keep = [trgt not in missing for trgt in metadata['source'].to_list()]
        meta = metadata[keep]
        estimates = decoupler_results[m + '_estimate'][keep]
        # pvals = res[m + '_pvals'][keep]

        # mirror estimates
        estimates = estimates.mul(meta['sign'], axis = 0)
        gt = meta.pivot(columns = 'source', values = 'sign').fillna(0)

        # add 0s in the ground-truth array for targets predicted by decoupler
        # for which there is no ground truth in the provided metadata (assumed 0)
        missing = list(set(estimates.columns) - set(meta['source']))
        gt = pd.concat([gt, pd.DataFrame(0, index= gt.index, columns=missing)], axis = 1, join = 'inner').sort_index(axis=1)

        flat_scores = []
        scores_names = [m]

        # flatten and then combine estimates and GT vectors
        # set ground truth to be either 0 or 1
        df_scores = pd.DataFrame({'score': estimates.to_numpy().flatten(), 'GT': gt.to_numpy().flatten()})
        flat_scores.append(df_scores)

        if by is not None and 'sign' in by:
            keep = [meta['sign'] == 1, meta['sign'] == - 1]
            sign = ['_positive', '_negative']

            for ind, s in zip(keep, sign):
                df = pd.DataFrame({'score': estimates[ind].to_numpy().flatten(), 'GT': gt[ind].to_numpy().flatten()})
                flat_scores.append(df)
                scores_names.append(m + s)

        for score, name in zip(flat_scores, scores_names):
            score['GT'] = abs(score['GT'])
            if score['GT'].sum() > min_exp:
                scores_gt[name] = score.reset_index()
                targets[name] = list(estimates.columns)
                metadatas[name] = meta


    return scores_gt, targets, metadatas

def format_benchmark_data(data, metadata, network, columns = None, meta_perturbation_col = 'treatment', meta_sign_col = 'sign', net_source_col = 'source', net_weight_col = 'weight', filter_experiments = True, filter_sources = False):

    if metadata.shape[0] != data.shape[0]:
        raise ValueError('The data and metadata do not have the same number of rows! ({0} vs. {1})\n'.format(data.shape[0], metadata.shape[0]))

    if not all(item in network.columns for item in [net_source_col, net_weight_col]):
        missing = list(set([net_source_col, net_weight_col]) - set(network.columns))
        raise ValueError('{0} column(s) are missing from the input network'.format(str(missing)))

    if not all(item in metadata.columns for item in [meta_perturbation_col, meta_sign_col]):
        missing = list(set([meta_perturbation_col, meta_sign_col]) - set(metadata.columns))
        raise ValueError('{0} column(s) are missing from the input metadata'.format(str(missing)))

    network = network.rename(columns={net_source_col:'source', net_weight_col:'weight'})
    metadata = metadata.rename(columns={meta_perturbation_col:'source', meta_sign_col:'sign'})

    #subset by TFs with GT available
    if filter_sources:
        keep = [src in metadata['source'].to_list() for src in network['source'].to_list()]
        network = network[keep]

    # filter out experiments without predictions available
    if filter_experiments:
        keep = [src in network['source'].to_list() for src in metadata['source'].to_list()]
        data = data[keep]
        metadata = metadata[keep]

    if columns is not None:
        if type(columns) != list:
            columns = [columns]

        if meta_perturbation_col in columns:
            columns.append('source')

        if meta_sign_col in columns:
            columns.append('sign')

        #check that columns are in metadata
        columns = list(set(metadata.columns).intersection(set(columns)))

        if len(columns) == 0:
            raise ValueError('None of the columns are in the metadata')

        for c in columns:
            if type(metadata[c][0]) == 'str':
                metadata[c] = metadata[c].str.replace('[_,:]', ' ')

    return data, metadata, network, columns

def performances(flat_data, sources, metadatas, columns = None, metric = 'roc', by = 'method', subset = None, n_iter = 100, seed = 7, pi0 = 0.5, min_exp = 5, verbose = True):

    bench = {}
    for method in flat_data.keys():
        if verbose: print('Calculating performance metrics for', method)
        if 'method' in by or 'sign' in by or 'all' in by:
            perf = get_performance(flat_data[method], metric, n_iter = n_iter, seed = seed, prefix= method, pi0=pi0)
            bench.update(perf)
        if('source' in by or 'all' in by) and ('_positive' not in method and '_negative' not in method):
            perf = get_source_performance(flat_data[method], sources[method], metric, subset = subset, n_iter = n_iter, seed = seed, prefix = method, pi0 = pi0, min_exp = min_exp)
            bench[method + '_bySource'] = perf
        if(columns is not None) and ('_positive' not in method and '_negative' not in method):
            perf = get_meta_performance(flat_data[method], metadatas[method], columns, metric, n_iter = n_iter, seed = seed, prefix= method, pi0=pi0, min_exp = min_exp)
            bench[method + '_byMeta'] = perf


    return bench

def get_mean_performances(benchmark_dict):
    #make dataframes with mean perfomances
    perf_method = {}
    perf_bySource = {}
    perf_bySign = {}
    perf_byMeta = {}
    for topkey, topvalue in benchmark_dict.items():
        if '_bySource' in topkey:
            for key, value in topvalue.items():
                perf_bySource[key] = np.mean(value)
        elif '_byMeta' in topkey:
            for key, value in topvalue.items():
                perf_byMeta[key] = np.mean(value)
        elif '_negative' in topkey or '_positive' in topkey:
            perf_bySign[topkey] = np.mean(topvalue)
        else:
            perf_method[topkey] = np.mean(topvalue)

    perfs = []

    if len(perf_method) > 0:
        perf_method = pd.DataFrame.from_dict(perf_method, orient='index').reset_index()
        perf_method.columns = ['id','value']
        perf_method[['method','metric']] = perf_method['id'].str.split('_', expand=True)
        perf_method = perf_method.pivot(index='method', columns='metric', values='value').reset_index()
        perfs.append(perf_method)

    if len(perf_bySign) > 0:
        perf_bySign = pd.DataFrame.from_dict(perf_bySign, orient='index').reset_index()
        perf_bySign.columns = ['id','value']
        perf_bySign[['method', 'sign','metric']] = perf_bySign['id'].str.split('_', expand=True)
        perf_bySign = perf_bySign.pivot(index=['method','sign'], columns='metric', values='value').reset_index()
        perfs.append(perf_bySign)

    if len(perf_bySource) > 0:
        perf_bySource = pd.DataFrame.from_dict(perf_bySource, orient='index').reset_index()
        perf_bySource.columns = ['id','value']
        perf_bySource[['method','source','metric']] = perf_bySource['id'].str.split('_', expand=True)
        perf_bySource = perf_bySource.pivot(index=['method','source'], columns='metric', values='value').reset_index()
        perfs.append(perf_bySource)

    if len(perf_byMeta) > 0:
        perf_byMeta = pd.DataFrame.from_dict(perf_byMeta, orient='index').reset_index()
        perf_byMeta.columns = ['id','value']
        perf_byMeta[['method','meta','metric']] = perf_byMeta['id'].str.split('_', expand=True)

        perf_byMeta = perf_byMeta.pivot(index = ['method', 'meta'], columns = 'metric', values = 'value').reset_index()

        perf_byMeta[['meta', 'level']] = perf_byMeta['meta'].str.split(':', expand=True)
        factors = perf_byMeta['meta'].unique()

        perf_byMeta = perf_byMeta.pivot(index=list(set(perf_byMeta.columns) -set(['meta', 'level'])), columns='meta', values='level').reset_index()

        perf_byMeta = perf_byMeta.rename(dict(zip(factors ,'meta_' + factors)), axis = 1)
        perfs.append(perf_byMeta)


    mean_perf = pd.concat(perfs)
    mean_perf = mean_perf.fillna('').reset_index()

    return mean_perf.drop(labels='index', axis = 1)


def run_benchmark(data, metadata, network, methods = None, metric = ['roc', 'calprc'], columns = None, meta_perturbation_col = 'target', meta_sign_col = 'sign', net_source_col = 'source', net_weight_col = 'weight',
filter_experiments= True, filter_sources = False, by = 'method', subset = None, min_exp = 5, pi0 = 0.5, n_iter = 100, seed = 7, verbose = True, **kwargs):
    """
    Benchmark methods or networks on a given set of perturbation experiments using activity inference with decoupler.

    Args:
        data (DataFrame): Gene expression data where each row is a perturbation experiment and each column a gene
        metadata (DataFrame): Metadata of the perturbation experiment containing the activated/inhibited targets and the sign of the perturbation
        network (DataFrame): Network in long format passed on to the decouple function
        methods (str or list of str, optional): List of methods to run. If none are provided use weighted top performers (mlm, ulm and wsum). To benchmark all methods set to "all". Defaults to None.
        metric (str or list of str, optional): Performance metric(s) to compute. See the description of get_performance for more details. Defaults to ['roc', 'calprc'].
        column (str or list of str, optional): Metadata columns that contain the levels for which you want individual performance decomposition. Defaults to None.
        meta_perturbation_col (str, optional): Column name in the metadata with perturbation targets. Defaults to 'target'.
        meta_sign_col (str, optional): Column name in the metadata with sign of perturbation. Defaults to 'sign'.
        net_source_col (str, optional): Column name in network with source nodes. Defaults to 'source'.
        net_weight_col (str, optional): Column name in network with interaction weight. Defaults to 'weight'.
        filter_experiments (bool, optional): Whether to filter out experiments whose perturbed targets cannot be infered from the given network. Defaults to True.
        filter_sources (bool, optional): Whether to fitler out sources in the network for which there are not perturbation experiments (reduces the number of predictions made by decouple). Defaults to False.
        by (str or list of str, optional): How to compute/decompose the performance score: any of 'method' and/or 'source'. Defaults to 'method'.
        subset (str or list of str, optional): Subset of sources for which to compute an individual performance score. Requires by to contain 'source'. Sources with fewer than 'min_exp' experiments in the dataset are ignored Defaults to None
        min_exp (int, optional): Minium number of perturbation experiments required to compute performance scores. Defaults to 5.
        pi0 (float, optional): Reference ratio for calibrated metrics. Corresponds to the baseline/reference class inbalance to which to set the metric. Defaults to 0.5.
        n_iter (int, optional): Number of iterations/subsampling used for the 'mcroc' and 'mcprc' metrics. Defaults to 100.
        seed (int, otional): Random seed to use for subsampling for the 'mcroc' and 'mcprc' metrics. Defaults to 7.
        verbose (bool, optional): Whether to print progession. Defaults to True.
        **kwargs: Other arguments to pass on to decouple

    Returns:
        mean_perf: DataFrame containing the mean performance for each metric and for each method (mean has to be done for the mcroc and mcprc metrics)
        bench: dict containing the whole data for each method and metric. Useful if you want to see the distribution for each subsampling for the mcroc and mcprc methods
    """

    #format and filter the data, metadata and networks
    data, metadata, network, columns = format_benchmark_data(data, metadata, network, columns, meta_perturbation_col=meta_perturbation_col, meta_sign_col = meta_sign_col,
                                                    net_source_col=net_source_col, net_weight_col = net_weight_col, filter_experiments=filter_experiments, filter_sources=filter_sources)

    
    #run prediction
    res = dc.decouple(data, network, methods=methods, verbose = verbose, **kwargs)

    #flatten and select predicitons before computing performance measures
    scores_gt, sources, metadatas = get_scores_GT(res, metadata, by= by, min_exp = min_exp)

    bench = performances(scores_gt, sources, metadatas, columns = columns, metric = metric, by = by, subset = subset, n_iter = n_iter, seed = seed, pi0 = pi0, min_exp = min_exp, verbose = verbose)

    mean_perfs = get_mean_performances(bench)

    return mean_perfs, bench

def benchmark_scatterplot(mean_perf, x = 'mcroc', y = 'mcprc', ax = None, label_col=None, ann_fontsize = None):
    """
    Creates a scatter plot for each given method for two performance metrics

    Args:
        mean_perf (DataFrame): Mean performance of each method output by run_benchmark()
        x (str, optional): Which metric to plot on the x axis. Defaults to 'mcroc'.
        y (str, optional): Which metric to plot on the y axis. Defaults to 'mcprc'.

    Returns:
        ax: Axes of a scatter plot
    """
    mean_perf = mean_perf.reset_index()

    if ax is None: ax = plt.subplot(111)
    ax.scatter(x = mean_perf[x], y = mean_perf[y])
    ax.set_aspect('equal')

    min_v = mean_perf[[x,y]].min().min()
    max_v = mean_perf[[x,y]].max().max()
    border = (max_v - min_v)/15

    ax.set_xlim(min_v - border, max_v + border)
    ax.set_ylim(min_v - border, max_v + border)

    if (x in ['roc','mcroc'] and y in ['roc','mcroc']) or (x in ['prc','mcprc', 'calprc'] and y in ['prc','mcprc', 'calprc']):
        ax.axline((0,0),slope=1, color = 'black', linestyle = ':')

    if label_col is not None and label_col in mean_perf.columns:
        for i, label in enumerate(mean_perf[label_col]):
            if ann_fontsize is None:
                ax.annotate(label.capitalize(), (mean_perf[x][i], mean_perf[y][i]))
            else:
                ax.annotate(label.capitalize(), (mean_perf[x][i], mean_perf[y][i]), fontsize = ann_fontsize)

    if x in ['mcroc', 'mcprc', 'roc', 'prc', 'calprc']:
        x = x + ' AUC'

    if y in ['mcroc', 'mcprc', 'roc', 'prc', 'calprc']:
        y = y + ' AUC'

    ax.set_xlabel(x.upper())
    ax.set_ylabel(y.upper())

    return ax

def benchmark_boxplot(benchmark_data, metric = 'mcroc', ax = None):
    """
    Creates boxplots for an iterative performance metric (i.e. mcroc and mcprc)

    Args:
        benchmark_data (dict): dict containing complete output from run_benchmark()
        metric (str, optional): Metric to plot a distribution for. Either mcroc or mcprc. Defaults to 'mcroc'.

    Returns:
        ax: Axes of a boxplot
    """

    if not (metric == 'mcprc' or metric == 'mcroc'):
        raise ValueError('Plotting of boxplots only possible for the \'mcprc\' and \'mcroc\' methods')

    #TODO: change so that input format corresponds again. Since mc is not that useful anymore, repurpose for target by target boxplots ?
    keys = [key for key in benchmark_data.keys() if metric in key.split('_')[1]]
    methods = [key.split('_')[0] for key in keys]

    if len(keys) == 0:
        raise ValueError('The given metric was not found in the benchmark data')

    if ax is None: ax = plt.subplot(111)
    for i, key in enumerate(keys):
        ax.boxplot(benchmark_data[key], positions = [i])
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylabel(metric.upper() + ' AUC')
    ax.set_xticklabels([m.capitalize() for m in methods])
    
    return ax