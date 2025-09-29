from manifold import internal
import numpy as np

from sklearn.manifold import *
from sklearn.decomposition import PCA

_internal_validation = {"general": {'expected_params' : ["method", "ndim"]},
                        
                        "lem":{'expected_params':{'knn':0.025},\
                               'accepted_params':{'gamma':(1/np.inf)}},
                            
                        "pca":{'expected_params':{},\
                               'accepted_params':{'whiten':False}},
                            
                        "tsne":{'expected_params':{'perplexity':50},\
                                'accepted_params':{'learning_rate':'auto','n_iter':1000}},
                            
                        "isomap":{'expected_params':{'n_neighbors':100},\
                                  'accepted_params':{'n_iter':None}},
                            
                        "mds":{'expected_params':{},\
                               'accepted_params':{'metric':True, 'eps':1e-3, 'n_iter':300, 'normalized_stress':'auto'}},
                            
                        "umap":{'expected_params':{},\
                                'accepted_params':{}}}
    
def _reduce_internal_main(matrix=None, verbose=True, **kwargs):
    
    method_specific_reduction = {"lem": _reduce_internal_lem,\
                                  "pca": _reduce_internal_pca,
                                  "tsne": _reduce_internal_tsne,
                                  "isomap": _reduce_internal_isomap,
                                  "mds": _reduce_internal_mds,
                                  "umap": _reduce_internal_umap}        
        
    # No need to validate, already passed validation at this point
    method = kwargs.pop("method")
    ndim = kwargs.pop("ndim")
    
    default_params = _internal_validation[method]["expected_params"]
    reduction_params = default_params.copy() # python is such a shitty programming language shallow copies suck like seriously wtf
    reduction_params.update(_internal_validation[method]["accepted_params"])
        
    # In case arguments that are different than default are provided
    altered_params = []
    for arg in kwargs.keys():
        altered_params.insert(0, "%s=%s" % (arg, kwargs[arg]))
        reduction_params[arg] = kwargs[arg]
        
    if verbose:
        print("Reducing matrix (%dX%d) using %s to target dimensionality %d" % (matrix.shape[0], 
                                                                                  matrix.shape[1],
                                                                                  method,
                                                                                  ndim))
        if len(altered_params) > 0:
            print("\tAdjusted parameters: %s" % ", ".join(altered_params))
                    
    reduced_matrix = method_specific_reduction[method](matrix, ndim, **reduction_params)
    
    if verbose:
        print("\tReduction done!")
        
    # Make sure dimensions are always cells over timepoints
    if reduced_matrix.shape[0] > reduced_matrix.shape[1]:
        reduced_matrix = np.transpose(reduced_matrix)
        
    return(reduced_matrix)


def _reduce_internal_validate_args(matrix, **kwargs):
    for expected in _internal_validation["general"]["expected_params"]:
        if expected not in kwargs.keys():
            internal._raise_error("Parameter `%s` must be defined when reducing dimensionality" % (expected))
            
    method = kwargs.pop("method")
    ndim = kwargs.pop("ndim")
    
    if method not in _internal_validation.keys():
        internal._raise_error("Invalid dimensionality reduction method %s, please use either (%s)" % (method, ", ".join([x for x in _internal_validation.keys()][1:])))
        
    if ndim <= 1 or ndim >= matrix.shape[0]:
        internal._raise_error("Invalid target dimensionality, ndim must be >= 1 and <= number of neurons")
        

    expected_params = _internal_validation[method]["expected_params"].keys()
    accepted_params = _internal_validation[method]["accepted_params"].keys()
    
    for method_expected in expected_params:
        if method_expected not in kwargs.keys():            
            internal._raise_error("Missing parameters when using `%s`, please enter all necessary paramters (%s)" % (method, ", ".join(expected_params)))
            
    for arg in kwargs.keys():
        if arg not in accepted_params and arg not in expected_params:
            internal._raise_error("Parameter `%s` is not an accepted parameter for method `%s`, please use (%s)" % (arg, method, ", ".join([x for x in accepted_params])))
            
    method_specific_validation = {"lem": _reduce_internal_lem_validate_args,\
                                  "pca": _reduce_internal_pca_validate_args,
                                  "tsne": _reduce_internal_tsne_validate_args,
                                  "isomap": _reduce_internal_isomap_validate_args,
                                  "mds": _reduce_internal_mds_validate_args,
                                  "umap": _reduce_internal_umap_validate_args}
        
    
    # Some methods require specific validation such as tsne
    kwargs["ndim"] = ndim
        
    method_specific_validation[method](matrix, **kwargs)
    

   
def _reduce_internal_lem_validate_args(matrix, **kwargs):
    if kwargs["knn"] < 0 or kwargs["knn"] > 1:
        internal._raise_error("Parameter knn must be in [0,1] when using lem")
    
def _reduce_internal_pca_validate_args(matrix, **kwargs):
    if 'whiten' in kwargs.keys():
        if type(kwargs["whiten"]) != bool:           
            internal._raise_error("Parameter whiten must be a boolean")
    
def _reduce_internal_tsne_validate_args(matrix, **kwargs):
    if kwargs["ndim"] >= 4:
        internal._raise_error("Parameter ndim must be < 4 when using tsne")
        
    if kwargs["perplexity"] < 0:
        internal._raise_error("Parameter perplexity must be positive")
        
    if 'learning_rate' in kwargs.keys():
        if type(kwargs["learning_rate"]) != int and\
           type(kwargs["learning_rate"]) != float and kwargs["learning_rate"] != 'auto':
            internal._raise_error("Parameter learning_rate must be numeric or set as 'auto'")
            
    if 'n_iter' in kwargs.keys():
        if kwargs["n_iter"] < 0:
            internal._raise_error("Parameter n_iter must be positive")
        
def _reduce_internal_isomap_validate_args(matrix, **kwargs):
    if kwargs["n_neighbors"] < 0:
        internal._raise_error("Parameter n_neighbors must be positive")
        
    if 'n_iter' in kwargs.keys():
        if kwargs["n_iter"] < 0 and kwargs["n_iter"] is not None:
            internal._raise_error("Parameter n_iter must be positive or set as None")

def _reduce_internal_mds_validate_args(matrix, **kwargs):
    if 'metric' in kwargs.keys():
        if type(kwargs["metric"]) != bool:           
            internal._raise_error("Parameter metric must be a boolean")
            
    if 'n_iter' in kwargs.keys():
        if kwargs["n_iter"] < 0:
            internal._raise_error("Parameter n_iter must be positive")
            
    if 'normalized_stress' in kwargs.keys():
        if type(kwargs["normalized_stress"]) != bool and kwargs["normalized_stress"] != 'auto':
            internal._raise_error("Parameter normalized_stress must be boolean or set as 'auto'")            

def _reduce_internal_umap_validate_args(matrix, **kwargs)    :
    pass
            
def _reduce_internal_get_prefix(**kwargs):
    # No need to validate, already passed validation at this point
    method = kwargs.pop("method")
    ndim = kwargs.pop("ndim")
    
    expected_params = _internal_validation[method]["expected_params"]
    accepted_params = _internal_validation[method]["accepted_params"]
        
    default_args = True
    
    # In case a single argument has been changed
    for arg in kwargs.keys():
        if arg in expected_params.keys():
            if kwargs[arg] != expected_params[arg]:
                default_args = False
                break
                
        if arg in accepted_params.keys():
            if kwargs[arg] != accepted_params[arg]:
                default_args = False                
                break
            
    if default_args:
        arg_prefix = "_defargs"
    else: 
        sorted_args = [arg for arg in kwargs.keys()]
        sorted_args.sort()
        arg_prefix = "".join(["_%s%s" % (arg,kwargs[arg]) for arg in sorted_args])
        
    reduce_final_prefix = "reduce%s_nd%s%s" % (method,ndim,arg_prefix)
    return(reduce_final_prefix)

def _reduce_internal_pca(matrix, ndim, **reduction_params):
    whiten = reduction_params["whiten"]    
    pca_embedding = PCA(n_components=ndim, whiten=whiten)
    reduced_matrix = pca_embedding .fit_transform(np.transpose(matrix))
    return(reduced_matrix)

def _reduce_internal_lem(matrix, ndim, **reduction_params):        
    n_neighbors = int(np.round(matrix.shape[1] * reduction_params["knn"]))
    gamma = reduction_params["gamma"]
    spectral_embedding = SpectralEmbedding(n_components=ndim,
                                           affinity='nearest_neighbors', 
                                           n_neighbors=n_neighbors,
                                           gamma=gamma,
                                           n_jobs=-1)
    
    reduced_matrix = spectral_embedding.fit_transform(np.transpose(matrix))
    return(reduced_matrix)

def _reduce_internal_tsne(matrix, ndim, **reduction_params):
    n_iter = reduction_params["n_iter"]
    learning_rate = reduction_params["learning_rate"]
    perplexity = reduction_params["perplexity"]
    
    tsne_embedding = TSNE(n_components=ndim, 
                          learning_rate=learning_rate,
                          n_iter=n_iter,
                          perplexity=perplexity)
    
    reduced_matrix = tsne_embedding.fit_transform(np.transpose(matrix))
    return(reduced_matrix)


def _reduce_internal_isomap(matrix, ndim, **reduction_params):
    n_neighbors = reduction_params["n_neighbors"]
    n_iter = reduction_params["n_iter"]
    
    iso_embedding = Isomap(n_components=ndim, 
                           max_iter=n_iter,
                           n_neighbors=n_neighbors)
    
    reduced_matrix = iso_embedding.fit_transform(np.transpose(matrix))
    return(reduced_matrix)
    

def _reduce_internal_umap(matrix, ndim, **reduction_params):
	pass

def _reduce_internal_mds(matrix, ndim, **reduction_params):
    n_iter = reduction_params["n_iter"]
    metric = reduction_params["metric"]
    eps = reduction_params["eps"]
    normalized_stress = reduction_params["normalized_stress"]
    
    MDS_embedding = MDS(n_components=ndim, 
                        eps=eps,
                        max_iter=n_iter,
                        metric=metric,
                        normalized_stress=normalized_stress)
    
    reduced_matrix = MDS_embedding.fit_transform(np.transpose(matrix))
    return(reduced_matrix)
    

