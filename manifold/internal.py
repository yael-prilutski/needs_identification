import numpy as np
import os
from manifold import reduce_internal
import pickle

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


os.environ["OMP_NUM_THREADS"] = '4' # for kmeans

BASE_DIRECTORY_NAME = "data"
BASE_DATASET = "base_mat"
BASE_DATASET_NAME = "%s.npy" % BASE_DATASET
CLUSTER_MAP_BASE_DIRECTORY = "cluster_map"
STOCHASTIC_CLUSTERING_LABELS_DIRECTORY = "stochastic_clustering"


BIN = 0
SCALE = 1
FILTER = 2
REDUCE = 3
CUSTOM = 4

required_operation_parameters = ['opcode', 'params']

internal_validation = {BIN: {'expected_params' : ["bin_size"],  'operation' : "binning matrix"},
                       SCALE: {'expected_params' : ["scale"],     'operation' : "scaling matrix"},
                       FILTER: {'expected_params' : ["threshold"], 'operation' : "filtering matrix"},
                       CUSTOM: {'expected_params' : ["function", "prefix"]}}


scaling_methods = {"perc": lambda vec: (vec - np.percentile(vec, 1))/(np.percentile(vec, 99) - np.percentile(vec, 1)),
                   "zscore": lambda vec: (vec - np.mean(vec))/(np.std(vec)),
                   "minmax": lambda vec: (vec - np.min(vec))/(np.max(vec) - np.min(vec))}

global _base_directory_internal
global _verbose
_verbose = False


def _euc_distance(centroid, tp):
    return np.linalg.norm(np.array(tp) - np.array(centroid))


def _raise_error(string_error, raise_exception=True):            
    global _verbose 
    
    if _verbose:
        #cf = currentframe()    
        #print(str(cf.f_back.f_code))
        print(string_error)
        
    if raise_exception:
        raise BaseException(string_error)


def _set_base_directory(base_directory):
    global _base_directory_internal
    _base_directory_internal = base_directory
    return _base_directory_internal 


def _get_base_directory():
    global _base_directory_internal
    return _base_directory_internal
    

def initialize_environment(base_directory):
    
    base_directory_full = "%s\\%s" % (base_directory, BASE_DIRECTORY_NAME)
    
    if BASE_DIRECTORY_NAME in os.listdir(base_directory):
        
        if not os.path.isdir(base_directory_full):
            _raise_error("Invalid base directory provided. already contains a file (but not directory) named %s"\
                         % BASE_DIRECTORY_NAME)
        
        _set_base_directory(base_directory_full)
        return
    
    os.makedirs(base_directory_full)
    _set_base_directory(base_directory_full)
    

def create_dataset(matrix=None, dataset_name=None):
    dataset_directory = "%s\\%s" % (_get_base_directory(), dataset_name)
    base_dataset = "%s\\%s" % (dataset_directory, BASE_DATASET_NAME)
    
    if dataset_directory not in os.listdir(_get_base_directory()):
        os.makedirs(dataset_directory)
    else:        
        if not os.path.isdir(dataset_directory):
            _raise_error("Invalid dataset directory provided.  Environment alreayd contains a file (but not directory) named %s"\
                         % dataset_directory)
        
        if BASE_DATASET_NAME in os.listdir():
            _raise_error("Dataset already initialized (%s already exists)" % base_dataset)
        

    # Make sure dimensions are always cells over timepoints
    # if matrix.shape[0] > matrix.shape[1]:
        # matrix = np.transpose(matrix)
                    
    with open(base_dataset, 'wb') as f:    
        np.save(f, matrix)        
        
    return matrix


def _save_intermediate_dataset(matrix=None, dataset_name=None, base=None, prefix=None, alias=None):
    dataset_directory = "%s\\%s" % (_get_base_directory(), dataset_name)
    
    if base is None: base = BASE_DATASET
    if prefix is None: prefix = ""
    
        
    if dataset_name not in os.listdir(_get_base_directory()):
        _raise_error("Dataset %s does not exist" % dataset_name)
    
        
    dataset_file_name = "%s\\%s%s.npy" % (dataset_directory, base, prefix)

    # Make sure dimensions are always neurons/features over timepoints
    if matrix.shape[0] > matrix.shape[1]:
        matrix = np.transpose(matrix)
                    
    with open(dataset_file_name, 'wb') as f:        
        np.save(f, matrix) 
        
    if alias is not None:
        alias_file_name = "%s\\%s.npy" % (dataset_directory, alias)
        
        with open(alias_file_name, 'wb') as f:        
            np.save(f, matrix) 
        
    return matrix

    
def get_dataset(dataset_name, alias=None, prefix=None, raise_exception=True):
    dataset_directory = "%s\\%s" % (_get_base_directory(), dataset_name)
    
    matrix_name = BASE_DATASET
    
    if alias is not None:
        matrix_name = alias
    
    if prefix is not None:
        matrix_name = "%s%s" % (matrix_name, prefix)
    
    matrix_file_name_to_load = "%s.npy" % matrix_name
    
    if matrix_file_name_to_load not in os.listdir(dataset_directory):
        _raise_error("Processed dataset alias %s does not exist for %s"\
                     % (matrix_name, dataset_name),
                     raise_exception=raise_exception)
            
        return None
            
    final_path_to_load = "%s\\%s" % (dataset_directory, matrix_file_name_to_load)
    mat = np.load(final_path_to_load)
    
    return(mat)


def process_dataset(dataset_name=None, alias=None, prefix=None, op_list=list(), rebase_alias=False):
    global _verbose
    
    mat = get_dataset(dataset_name, alias=alias, prefix=prefix)
    base_intermediate_prefix = ""

    # Sanity
    if type(op_list) is not list:
        _raise_error("Invalid `op_list` provided, must be a list")
        

    # Loop over all processing operations fed for dataset
    for op in op_list:

        # Sanity validations
        if type(op) is not dict:
            _raise_error("Invalid operation format inserted operation must be a dictionary")            

        if sum([key in op.keys() for key in required_operation_parameters]) != len(required_operation_parameters):            
            print_str ="(%s)" % ", ".join([key for key in required_operation_parameters])                    

            _raise_error("Invalid operation format (%s) inserted. Make sure operation contains %s" % (str(op), print_str))
        
        _validate_opcode(mat, op["opcode"], op["params"])        

        intermediate_prefix = _get_intermediate_prefix_name(op["opcode"], op["params"], base_intermediate_prefix)
        intermediate_dataset = get_dataset(dataset_name, alias=alias, prefix=intermediate_prefix, raise_exception=False)

        if intermediate_dataset is None:

            if _verbose:
                print("intermediate %s does not exist for %s, creating." % (intermediate_prefix, dataset_name))

            intermediate_dataset = _execute_opcode(mat, op["opcode"], op["params"])
            intermediate_alias = op["alias"] if "alias" in op.keys() else None
            _save_intermediate_dataset(intermediate_dataset, dataset_name, alias, intermediate_prefix, intermediate_alias)

        base_intermediate_prefix = intermediate_prefix
        mat = intermediate_dataset  
                
    if intermediate_dataset.shape[0] > intermediate_dataset.shape[1]:
        intermediate_dataset = np.transpose(intermediate_dataset)

    return intermediate_dataset


def _validate_opcode(matrix, opcode, params):
    
    # Validate internaly
    if opcode in internal_validation.keys():
        expected_params = internal_validation[opcode]["expected_params"]
        operation = internal_validation[opcode]["operation"]

        for expected in expected_params:
            if expected not in params.keys():
                _raise_error("Parameter `%s` must be defined when %s" % (expected, operation))

    elif opcode == REDUCE:
        # too much logic -> migrate to an independent module
        reduce_internal._reduce_internal_validate_args(matrix, **params)
    else:
        _raise_error("Invalid opcode used %s" % opcode)


    # Specific internal validations
    if opcode == BIN:        
        bin_size = params["bin_size"]

        if bin_size <= 0:
            _raise_error("Invalid bin size %d" % bin_size)

    elif opcode == SCALE:
        scaling_method  = params["scale"].lower()

        if scaling_method not in scaling_methods.keys():
            defined_methods_str = "(%s)" % ", ".join([key for key in scaling_methods.keys()])
            
            _raise_error("Scaling method %s unknown, use defined scaling methods %s" % (scaling_method, defined_methods_str))
        
    elif opcode == FILTER:
        pass

def _get_intermediate_prefix_name(opcode, params, base_prefix=""):       
    if opcode not in internal_validation.keys() and opcode != REDUCE:
        _raise_error("Invalid opcode used %s" % opcode)
            
    if opcode not in  [CUSTOM, REDUCE]:
        prefix = "_".join(["%s%s" % (kw, params[kw]) for kw in internal_validation[opcode]["expected_params"]])
    elif opcode == REDUCE:
        prefix = reduce_internal._reduce_internal_get_prefix(**params)
    else:
        prefix = params["prefix"]

    return("%s_%s" % (base_prefix, prefix))
    

def _execute_opcode(matrix, opcode, params):    
    operation_functions = {BIN: _bin_matrix,
                           SCALE: _scale_matrix,
                           FILTER: _filter_matrix,
                           REDUCE: _reduce_matrix,
                           CUSTOM: _custom_operation_matrix}
    
    # At this point, no need to validate params as they should've already been validated by
    # `_validate_opcode`

    if opcode in operation_functions.keys():
        operation_func = operation_functions[opcode]
        processed_mat = operation_func(matrix, **params)
    else:
        _raise_error("Invalid opcode used %s" % opcode)

    return(processed_mat)


def _bin_matrix(matrix, **kwargs):    
    bin_size = kwargs["bin_size"]        

    bin_size = int(bin_size)

    if _verbose:
        print("Binning time points using %s", bin_size)

    # dispose last time point in case of misalignment with bin size
    chunks = np.arange(0, ((matrix.shape[1] - matrix.shape[1] % bin_size) / bin_size))
    list_of_indices_to_bin = [np.arange((int(chunk_idx) * bin_size),((int(chunk_idx) + 1) * bin_size)) for chunk_idx in chunks]
    binned_matrix = np.transpose(np.vstack([np.mean(matrix[:,indices_to_bin], axis=1) for indices_to_bin in list_of_indices_to_bin]))

    return(binned_matrix)


def _scale_matrix(matrix, **kwargs):
    scale = kwargs["scale"] 

    scale = scale.lower()
    
    if _verbose:
        print("Scaling features using %s", scale)

    # Get scaling method
    scale_func = scaling_methods[scale]
    scaled_matrix = np.apply_along_axis(scale_func, axis=1, arr=matrix)
    return(scaled_matrix)


def _filter_matrix(matrix, **kwargs):    
    threshold = kwargs["threshold"]
    
    if _verbose:
        print("Filtering features with mean >= %s", threshold)

    features_passing_threshold =  np.apply_along_axis(lambda vec: np.abs(np.mean(vec)) <= threshold , axis=1, arr=matrix)
    return(matrix[features_passing_threshold,:])


def _reduce_matrix(matrix, **kwargs):
    # Too much logic -> independent module
    reduced_matrix = reduce_internal._reduce_internal_main(matrix, **kwargs)
    return(reduced_matrix)


def _custom_operation_matrix(matrix, **kwargs):
    pass


def _get_stochastic_kmeans_cluster_labels_name(alias=None,
                                               prefix=None,
                                               number_of_clusters=None,
                                               timepoints_to_include=None,
                                               stochastic_kmeans_reps=None):
    
    # If this was oop implemented, I'd definitely consider privat`ing this
    if number_of_clusters is None\
        or timepoints_to_include is None\
        or stochastic_kmeans_reps is None:
        _raise_error("Please provide all parameters creating stochastic clustering prefix")
    
    matrix_name = BASE_DATASET
    
    if alias is not None:
        matrix_name = alias
    
    if prefix is not None:
        matrix_name = "%s%s" % (matrix_name, prefix)        
            
    
    cluster_labels_file_name = "tp%s_nc%s_skr%s_%s" % (timepoints_to_include,\
                                                       number_of_clusters,\
                                                       stochastic_kmeans_reps,\
                                                       matrix_name)
    return(cluster_labels_file_name)


def _get_stochastic_kmeans_cluster_labels(dataset_name,
                                          alias=None,
                                          prefix=None,
                                          number_of_clusters=None,
                                          timepoints_to_include=None,
                                          stochastic_kmeans_reps=None,
                                          raise_exception=True):
    cluster_labels_file_name = \
        _get_stochastic_kmeans_cluster_labels_name(alias=alias,
                                                   prefix=prefix,
                                                   number_of_clusters=number_of_clusters,
                                                   timepoints_to_include=timepoints_to_include,
                                                   stochastic_kmeans_reps=stochastic_kmeans_reps)            
    
    dataset_directory = "%s\\%s" % (_get_base_directory(), dataset_name)
    stochastic_clustering_directory = "%s\\%s" % (dataset_directory, STOCHASTIC_CLUSTERING_LABELS_DIRECTORY)
    
    if STOCHASTIC_CLUSTERING_LABELS_DIRECTORY not in os.listdir(dataset_directory):
        os.makedirs(stochastic_clustering_directory)
    else:        
        if not os.path.isdir(stochastic_clustering_directory):
            _raise_error("Invalid cluser map directory provided.  Environment alreayd contains a file (but not directory) named %s"\
                         % stochastic_clustering_directory)

    cluster_labels_file_name_to_load = "%s.pkl" % cluster_labels_file_name
    
    if cluster_labels_file_name_to_load not in os.listdir(stochastic_clustering_directory):
        _raise_error("Processed stochastic cluster labels %s do not exist for %s"\
                     % (cluster_labels_file_name, dataset_name),
                     raise_exception=raise_exception)
                
        return None
            
    final_path_to_load = "%s\\%s" % (stochastic_clustering_directory, cluster_labels_file_name_to_load)
    
    with open(final_path_to_load, 'rb') as f:
        cluster_map = pickle.load(f)
                
    return(cluster_map)    
    
        
def _stochastic_kmeans_clustering(dataset_name,
                                  alias=None,
                                  prefix=None,
                                  number_of_clusters=None,
                                  timepoints_to_include=None,
                                  stochastic_kmeans_reps=500):
    
    if number_of_clusters is None or timepoints_to_include is None:
        _raise_error("Cannot preform stochastic clustering without number of clusters / timepoints to include")
        
    if type(timepoints_to_include) != int:
        _raise_error("Timepoints to include must be an int")
        
            
    mat = get_dataset(dataset_name, alias=alias, prefix=prefix)
    
    cluster_labels = \
        _get_stochastic_kmeans_cluster_labels(dataset_name=dataset_name,
                                              alias=alias,
                                              prefix=prefix,
                                              number_of_clusters=number_of_clusters,
                                              timepoints_to_include=timepoints_to_include,
                                              stochastic_kmeans_reps=stochastic_kmeans_reps,
                                              raise_exception=False)
        
    # Redundant but still
    cluster_labels_name = \
        _get_stochastic_kmeans_cluster_labels_name(alias=alias,
                                                   prefix=prefix,
                                                   number_of_clusters=number_of_clusters,
                                                   timepoints_to_include=timepoints_to_include,
                                                   stochastic_kmeans_reps=stochastic_kmeans_reps)
    
    if cluster_labels is not None:
        print("Cluster map %s already exists" % cluster_labels_name)
        return(cluster_labels)
    else:
        print("Cluster map %s does not exist creating" % cluster_labels_name)

    centroid = np.mean(mat, axis=1)    
    distance_from_centroid = np.apply_along_axis(lambda tp: _euc_distance(centroid, tp), 0, mat)
    sorted_timepoints_indices_by_distance = np.argsort(-distance_from_centroid) # Decreasing order
    
    included_pointcloud_mat = mat[:,sorted_timepoints_indices_by_distance[0:timepoints_to_include]]
    
    labels_matrix = np.array([])
    
    for rep in range(0,stochastic_kmeans_reps):
        print(rep)
        kmeans = KMeans(n_clusters=number_of_clusters, n_init="auto") # what the fuck are these warnings
        kmeans.fit_predict(np.transpose(included_pointcloud_mat))                
        
        if rep == 0:
            labels_matrix = np.array([kmeans.labels_])
        else:
            labels_matrix = np.vstack([labels_matrix, np.array([kmeans.labels_])])
            
    distances = pdist(np.transpose(labels_matrix), metric='euclidean') # pretty heavy operation
    distance_matrix = squareform(distances ** 2)

    # equivalent of ward.d2
    heirarchial_tree = linkage(distance_matrix, method='ward')
        
    cluster_final_labels = fcluster(heirarchial_tree, number_of_clusters, criterion='maxclust')
    final_labels = np.repeat(-1, mat.shape[1])
    final_labels[sorted_timepoints_indices_by_distance[0:timepoints_to_include]] = cluster_final_labels
        
        
    print("\tStochastic k-means clustering %s of %s is done" % (cluster_labels_name, dataset_name))
    
    # In principle at this point, _get_stochastic_kmeans_cluster_labels should've raised an exception if dataset is invalid
    # no need to validate or create dir then    
        
    dataset_directory = "%s\\%s" % (_get_base_directory(), dataset_name)
    stochastic_clustering_directory = "%s\\%s" % (dataset_directory, STOCHASTIC_CLUSTERING_LABELS_DIRECTORY)
    cluster_labels_file_name_to_write = "%s.pkl" % cluster_labels_name
    final_path_to_write = "%s\\%s" % (stochastic_clustering_directory, cluster_labels_file_name_to_write)
        
    with open(final_path_to_write, 'wb') as f:
        pickle.dump(final_labels, f)
        
    return(final_labels)


def _get_cluster_map_name(alias=None,
                          prefix=None,
                          max_cluster_number=None,
                          min_num_of_frames=None,
                          timepoint_bins=None,
                          nreps=None):
    
    # If this was oop implemented, I'd definitely consider privat`ing this
    if max_cluster_number is None\
        or min_num_of_frames is None\
        or timepoint_bins is None\
        or nreps is None:
        _raise_error("Please provide all parameters when creating cluster map prefix")
    
    matrix_name = BASE_DATASET
    
    if alias is not None:
        matrix_name = alias
    
    if prefix is not None:
        matrix_name = "%s%s" % (matrix_name, prefix)        
            
    
    cluster_map_file_name = "nreps%s_tp%s_mf%s_mc%s_%s" % (nreps,\
                                                           timepoint_bins,\
                                                           min_num_of_frames,\
                                                           max_cluster_number,\
                                                           matrix_name)
    return(cluster_map_file_name)

def _get_cluster_map(dataset_name, 
                     alias=None,
                     prefix=None,
                     max_cluster_number=None, 
                     min_num_of_frames=None,
                     timepoint_bins=None,
                     nreps=None,
                     raise_exception=True):
    
    
    cluster_map_name = \
        _get_cluster_map_name(alias=alias,
                              prefix=prefix,
                              max_cluster_number=max_cluster_number,
                              min_num_of_frames=min_num_of_frames,
                              timepoint_bins=timepoint_bins,
                              nreps=nreps)
            
    
    dataset_directory = "%s\\%s" % (_get_base_directory(), dataset_name)
    cluster_map_directory = "%s\\%s" % (dataset_directory, CLUSTER_MAP_BASE_DIRECTORY)
    
    if CLUSTER_MAP_BASE_DIRECTORY not in os.listdir(dataset_directory):
        os.makedirs(cluster_map_directory)
    else:        
        if not os.path.isdir(cluster_map_directory):
            _raise_error("Invalid cluser map directory provided.  Environment alreayd contains a file (but not directory) named %s"\
                         % cluster_map_directory)

    cluster_map_file_name_to_load = "%s.pkl" % cluster_map_name
    
    if cluster_map_file_name_to_load not in os.listdir(cluster_map_directory):
        _raise_error("Processed cluster map %s does not exist for %s"\
                     % (cluster_map_name, dataset_name),
                     raise_exception=raise_exception)
                
        return None
            
    final_path_to_load = "%s\\%s" % (cluster_map_directory, cluster_map_file_name_to_load)
    
    with open(final_path_to_load, 'rb') as f:
        cluster_map = pickle.load(f)
                
    return(cluster_map)

def process_cluster_map(dataset_name,
                         alias=None,
                         prefix=None,
                         max_cluster_number=20, 
                         min_num_of_frames=500,
                         timepoint_bins=40,
                         nreps=20,
                         stochastic_kmeans_reps=500):
    
    
    # This should die if dataset doesn't even exist    
    mat = get_dataset(dataset_name, alias=alias, prefix=prefix)
    
    cluster_map = _get_cluster_map(dataset_name=dataset_name,
                                   alias=alias,
                                   prefix=prefix,
                                   max_cluster_number=max_cluster_number, 
                                   min_num_of_frames=min_num_of_frames,
                                   timepoint_bins=timepoint_bins,
                                   nreps=nreps,
                                   raise_exception=False)
        
    # Redundant but still
    cluster_map_name = _get_cluster_map_name(alias=alias,
                                             prefix=prefix,
                                             max_cluster_number=max_cluster_number, 
                                             min_num_of_frames=min_num_of_frames,
                                             timepoint_bins=timepoint_bins,
                                             nreps=nreps)
    
    if cluster_map is not None:
        print("Cluster map %s already exists" % cluster_map_name)
        return(cluster_map)
    else:
        print("Cluster map %s does not exist creating" % cluster_map_name)
    
    time_points_range = np.linspace(min_num_of_frames, mat.shape[1], timepoint_bins)
    number_of_time_points_to_include = [int(tp) for tp in np.round(time_points_range)]
    centroid = np.mean(mat, axis=1)    
    distance_from_centroid = np.apply_along_axis(lambda tp: _euc_distance(centroid, tp), 0, mat)
    sorted_timepoints_indices_by_distance = np.argsort(-distance_from_centroid) # Decreasing order
    
    mse_matrix = np.array([])
    
    for included in number_of_time_points_to_include:        
    
        included_pointcloud_mse = np.array([])    
       # Include only the #N farthest away timepoints
        included_pointcloud_mat = mat[:,sorted_timepoints_indices_by_distance[0:included]]
        
        for num_of_clusters in range(2,max_cluster_number + 1):            
            reps_mse = []            
            
            if _verbose:
                print("Calculating for %.3f outmost points, %s clusters" % (included / mat.shape[1], num_of_clusters))
                
            for rep_i in range(0, nreps):
                kmeans = KMeans(n_clusters=num_of_clusters, n_init="auto")
                kmeans.fit_predict(np.transpose(included_pointcloud_mat))                
                reps_mse.append(kmeans.inertia_)
            
            if num_of_clusters == 2:
                included_pointcloud_mse = np.array(reps_mse)
            else:            
                included_pointcloud_mse = np.vstack([included_pointcloud_mse, np.array(reps_mse)])
                  
        mean_mse_for_included_timepoints = np.mean(included_pointcloud_mse, axis=1) 
        
        if (mse_matrix.size == 0):
            mse_matrix = mean_mse_for_included_timepoints
        else:
            mse_matrix = np.vstack([mse_matrix, mean_mse_for_included_timepoints])
        
    clust_mean = np.mean(mse_matrix, 0)    
    clust_mean = (clust_mean - min(clust_mean)) / (max(clust_mean) - min(clust_mean))
    clust_mean = clust_mean * len(clust_mean)
    cluster_distances = [_euc_distance(np.array([0,0]),\
                                       np.array([idx + 1, clust_mean[idx]])) for idx in range(0,len(clust_mean))]
    
    optimal_number_of_clusters = range(2, max_cluster_number + 1)[np.where(cluster_distances == min(cluster_distances))[0][0]]
    
    inclusion_mean = np.mean(mse_matrix, 1)
    inclusion_mean = (inclusion_mean - min(inclusion_mean)) / (max(inclusion_mean) - min(inclusion_mean))
    inclusion_mean = inclusion_mean  * -1 + 1
    inclusion_mean = inclusion_mean * len(inclusion_mean)    
    inclusion_distances = [_euc_distance(np.array([0,0]),\
                                         np.array([idx + 1, inclusion_mean[idx]])) for idx in range (0,len(inclusion_mean))]
    
    optimal_number_of_timepoints_to_include = number_of_time_points_to_include[np.where(inclusion_distances == min(inclusion_distances))[0][0]]
        
    
    cluster_labels = _stochastic_kmeans_clustering(dataset_name=dataset_name,
                                                   alias=alias,
                                                   prefix=prefix,
                                                   number_of_clusters=optimal_number_of_clusters,
                                                   timepoints_to_include=optimal_number_of_timepoints_to_include,
                                                   stochastic_kmeans_reps=stochastic_kmeans_reps)
    
    cluster_map = {'number_of_clusters' : optimal_number_of_clusters,
                   'number_of_timepoints' : optimal_number_of_timepoints_to_include,
                   'mse_matrix' : mse_matrix,
                   'cluster_labels' : cluster_labels}
    
    print("\tCreation of cluster map %s for dataset %s is done" % (cluster_map_name, dataset_name))
    
    # In principle at this point, _get_cluster_map should've raised an exception if dataset is invalid
    # no need to validate or create dir then    
    
    dataset_directory = "%s\\%s" % (_get_base_directory(), dataset_name)
    cluster_map_directory = "%s\\%s" % (dataset_directory, CLUSTER_MAP_BASE_DIRECTORY)
    cluster_map_file_name_to_write = "%s.pkl" % cluster_map_name
    final_path_to_write = "%s\\%s" % (cluster_map_directory, cluster_map_file_name_to_write)
    
    with open(final_path_to_write, 'wb') as f:
        pickle.dump(cluster_map, f)
    
    return(cluster_map)


def estimate_dimensionality(matrix=None, dataset_name=None, alias=None, prefix=None,
                            exclusion_fraction_dimensionality_estimation=0.04):
    """
    Estimates the dimensionality of a dataset. If dataset_name is not set, estimates
    dimensionality over `matrix`

    Parameters:
    - matrix (numpy.ndarray, optional): The data matrix for the dataset.
    - dataset_name (str, optional): The name of the dataset.
    - alias (str, optional): An alias for the dataset.
    - prefix (str, optional): A prefix for the dataset.
    - exclusion_fraction_dimensionality_estimation (float, optional): Fraction of datapoints to exclude.

    Returns:
    float: Estimated dimensionality of the dataset.
    """

    if matrix is None:
        matrix = get_dataset(dataset_name=dataset_name,
                             alias=alias,
                             prefix=prefix)

    # Also kinda silly
    return (_get_intrinsic_dimension(matrix, percentile=(1 - exclusion_fraction_dimensionality_estimation)))


def _get_intrinsic_dimension(mat, percentile=.96):
    
    print("Estimating dimensionality for matrix (%sx%s) exclusion_factor=%.3f" % (mat.shape[0], mat.shape[1], percentile))
    distances = pdist(np.transpose(mat), metric='euclidean')
    distance_matrix = squareform(distances)
    
    def two_nn(vec): 
        nn1 = vec[np.argsort(vec)[1]]
        nn2 = vec[np.argsort(vec)[2]]
        return(nn2/nn1)
    
    all_two_nns = np.apply_along_axis(two_nn, axis=1, arr=distance_matrix)
    sorted_all_two_nns = all_two_nns.copy()
    sorted_all_two_nns.sort()    
    
    n_points_to_use = int(np.floor(sorted_all_two_nns.shape[0] * percentile))
    effective_range = range(0,n_points_to_use)
    
    linreg_x = np.log(sorted_all_two_nns[effective_range]) - 1
    linreg_y = -np.log(1 - np.array(effective_range) / n_points_to_use)    
    linreg_x = linreg_x.reshape(-1,1)
    linreg_y = linreg_y.reshape(-1,1)
    
    lm = LinearRegression().fit(linreg_x, linreg_y)
    lm.coef_
    
    return(lm.coef_[0,0])

    