#!python3
from ctypes import *
import math
import random

### main lib
syndn_lib = CDLL("libdarknet.so", RTLD_GLOBAL)

gpu_index = syndn_lib.gpu_index
seed = syndn_lib.seed

### get lib structs
class METADATA(Structure):
	_fields_ = [("classes", c_int),
				("names", POINTER(c_char_p)),
				]

class TREE(Structure):
	_fields_ = [("leaf", POINTER(c_int)),
				("n", c_int),
				("parent", POINTER(c_int)),
				("child", POINTER(c_int)),
				("group", POINTER(c_int)),
				("name", POINTER(c_char_p)),
				("groups", c_int),
				("group_size", POINTER(c_int)),
				("group_offset", POINTER(c_int)),
				]

class ACTIVATION_SCHEME(Structure):
	_fields_ = [("type", c_int),
				("alpha", c_float),
				("beta", c_float),
				]

class INITIALIZER(Structure):
	_fields_ = [("type", c_int),
				("alpha", c_float),
				("mu", c_float),
				("sigma", c_float),
				("auto_sigma", c_float),
				]

class WEIGHT_TRANSFORM_SCHEME(Structure):
	_fields_ = [("type", c_int),
				("step_size", c_float),
				("num_level", c_int),
				("first_shifting_factor", c_int),
				]

class QUANTIZATION_SCHEME(Structure):
	_fields_ = [("type", c_int),
				("step_size", c_float),
				("num_level", c_int),
				("root", c_float),
				("zero_center", c_int),
				]

class FIXED_POINT_SCHEME(Structure):
	_fields_ = [("type", c_int),
				("nbit", c_int),
				("ibit", c_int),
				("sign", c_int),
				]

class CUTOUT_ARGS(Structure):
	_fields_ = [("max_w", c_int),
				("max_h", c_int),
				("prob", c_float),
				]

class UPDATE_ARGS(Structure):
	_fields_ = [("optim", c_int),
				("batch", c_int),
				("learning_rate", c_float),
				("momentum", c_float),
				("decay", c_float),
				("B1", c_float),
				("B2", c_float),
				("eps", c_float),
				("t", c_int),
				]

# pre-define LAYER and NETWORK class to include in themselves
class LAYER(Structure):
	pass

class NETWORK(Structure):
	pass

# _fields_ definition of LAYER and NETWORK
LAYER._fields_ = [("self_index", c_uint),
				("type", c_int),
				("activation", ACTIVATION_SCHEME),
				("initializer", INITIALIZER),
				("quantization", QUANTIZATION_SCHEME),
				("weight_transform", WEIGHT_TRANSFORM_SCHEME),
				("fixed_point", FIXED_POINT_SCHEME),
				("cost_type", c_int),
				("forward_gpu", CFUNCTYPE(None, LAYER, NETWORK)),
				("backward_gpu", CFUNCTYPE(None, LAYER, NETWORK)),
				("update_gpu", CFUNCTYPE(None, LAYER, UPDATE_ARGS)),
				("name", c_char * 64),
				("batch_normalize", c_int),
			    ("shortcut", c_int),
			    ("batch", c_int),
			    ("forced", c_int),
			    ("flipped", c_int),
			    ("inputs", c_int),
			    ("outputs", c_int),
			    ("nweights", c_int),
			    ("nbiases", c_int),
			    ("extra", c_int),
			    ("truths", c_int),
			    ("h", c_int),
			    ("w", c_int),
			    ("c", c_int),
			    ("out_h", c_int),
			    ("out_w", c_int),
			    ("out_c", c_int),
			    ("n", c_int),
			    ("max_boxes", c_int),
			    ("groups", c_int),
			    ("size", c_int),
			    ("side", c_int),
			    ("stride", c_int),
			    ("reverse", c_int),
			    ("flatten", c_int),
			    ("spatial", c_int),
			    ("pad", c_int),
			    ("sqrt", c_int),
			    ("flip", c_int),
			    ("prior_flip", c_int),
			    ("prior_clip", c_int),
			    ("min_size", c_float),
			    ("max_size", c_float),
			    ("variance_size", c_float),
			    ("variance_center", c_float),
			    ("dilation", c_int),
			    ("impact", c_float),
				("warmup", c_int),
				("blind", c_int),
				("logistic_derivative", c_int),
				("index", c_int),
				("steps", c_int),
				("hidden", c_int),
				("truth", c_int),
				("smooth", c_float),
				("angle", c_float),
			    ("saturation", c_float),
			    ("exposure", c_float),
			    ("shift", c_float),
			    ("ratio", c_float),
			    ("learning_rate_scale", c_float),
			    ("clip", c_float),
			    ("noloss", c_int),
			    ("softmax", c_int),
			    ("classes", c_int),
			    ("coords", c_int),
			    ("background", c_int),
			    ("rescore", c_int),
			    ("objectness", c_int),
			    ("noadjust", c_int),
			    ("reorg", c_int),
			    ("tanh", c_int),
			    ("mask", POINTER(c_int)),
			    ("mask_gpu", POINTER(c_int)),
			    ("skip_index", POINTER(c_int)),
			    ("skip_index_gpu", POINTER(c_int)),
			    ("total", c_int),
			    ("alpha", c_float),
				("beta", c_float),
				("kappa", c_float),
				("coord_scale", c_float),
				("object_scale", c_float),
				("noobject_scale", c_float),
				("class_scale", c_float),
				("bias_match", c_int),
				("ignore_thresh", c_float),
				("truth_thresh", c_float),
				("thresh", c_float),
				("onlyforward", c_int),
			    ("stopbackward", c_int),
			    ("frozen", c_int),
			    ("root_connect", c_int),
			    ("dontload", c_int),
			    ("dontsave", c_int),
			    ("dontloadscales", c_int),
			    ("numload", c_int),
			    ("temperature", c_float),
			    ("probability", c_float),
			    ("scale", c_float),
			    ("indexes", POINTER(c_int)),
			    ("input_layers", POINTER(c_int)),
			    ("input_sizes", POINTER(c_int)),
			    ("map", POINTER(c_int)),
			    ("map_gpu", POINTER(c_int)),
			    ("counts", POINTER(c_int)),
			    ("sums", POINTER(POINTER(c_float))),
			    ("rand", POINTER(c_float)),
			    ("cost", POINTER(c_float)),
			    ("state", POINTER(c_float)),
			    ("prev_state", POINTER(c_float)),
			    ("forgot_state", POINTER(c_float)),
			    ("forgot_delta", POINTER(c_float)),
			    ("state_delta", POINTER(c_float)),
			    ("combine_cpu", POINTER(c_float)),
			    ("combine_delta_cpu", POINTER(c_float)),
			    ("concat", POINTER(c_float)),
			    ("concat_delta", POINTER(c_float)),
			    ("tran_weights", POINTER(c_float)),
			    ("biases", POINTER(c_float)),
			    ("bias_updates", POINTER(c_float)),
			    ("scales", POINTER(c_float)),
			    ("scale_updates", POINTER(c_float)),
			    ("weights", POINTER(c_float)),
			    ("weight_updates", POINTER(c_float)),
			    ("delta", POINTER(c_float)),
			    ("output", POINTER(c_float)),
			    ("loss", POINTER(c_float)),
			    ("squared", POINTER(c_float)),
			    ("norms", POINTER(c_float)),
			    ("norms_delta", POINTER(c_float)),
			    ("norms_delta_gpu", POINTER(c_float)),
			    ("spatial_mean", POINTER(c_float)),
			    ("mean", POINTER(c_float)),
			    ("variance", POINTER(c_float)),
			    ("mean_delta", POINTER(c_float)),
			    ("variance_delta", POINTER(c_float)),
			    ("rolling_mean", POINTER(c_float)),
			    ("rolling_variance", POINTER(c_float)),
			    ("x", POINTER(c_float)),
			    ("x_norm", POINTER(c_float)),
			    ("m", POINTER(c_float)),
			    ("v", POINTER(c_float)),
			    ("bias_m", POINTER(c_float)),
			    ("bias_v", POINTER(c_float)),
			    ("scale_m", POINTER(c_float)),
			    ("scale_v", POINTER(c_float)),
			    ("z_cpu", POINTER(c_float)),
			    ("r_cpu", POINTER(c_float)),
			    ("h_cpu", POINTER(c_float)),
			    ("prev_state_cpu", POINTER(c_float)),
			    ("temp_cpu", POINTER(c_float)),
			    ("temp2_cpu", POINTER(c_float)),
			    ("temp3_cpu", POINTER(c_float)),
			    ("dh_cpu", POINTER(c_float)),
			    ("hh_cpu", POINTER(c_float)),
			    ("prev_cell_cpu", POINTER(c_float)),
			    ("cell_cpu", POINTER(c_float)),
			    ("f_cpu", POINTER(c_float)),
			    ("i_cpu", POINTER(c_float)),
			    ("g_cpu", POINTER(c_float)),
			    ("o_cpu", POINTER(c_float)),
			    ("c_cpu", POINTER(c_float)),
			    ("dc_cpu", POINTER(c_float)),
			    ("input_layer", POINTER(LAYER)),
			    ("self_layer", POINTER(LAYER)),
			    ("output_layer", POINTER(LAYER)),
			    ("reset_layer", POINTER(LAYER)),
			    ("update_layer", POINTER(LAYER)),
			    ("state_layer", POINTER(LAYER)),
			    ("input_gate_layer", POINTER(LAYER)),
			    ("state_gate_layer", POINTER(LAYER)),
			    ("input_save_layer", POINTER(LAYER)),
			    ("state_save_layer", POINTER(LAYER)),
			    ("input_state_layer", POINTER(LAYER)),
			    ("state_state_layer", POINTER(LAYER)),
			    ("input_z_layer", POINTER(LAYER)),
			    ("state_z_layer", POINTER(LAYER)),
			    ("input_r_layer", POINTER(LAYER)),
			    ("state_r_layer", POINTER(LAYER)),
			    ("input_h_layer", POINTER(LAYER)),
			    ("state_h_layer", POINTER(LAYER)),
			    ("wz", POINTER(LAYER)),
			    ("uz", POINTER(LAYER)),
			    ("wr", POINTER(LAYER)),
			    ("ur", POINTER(LAYER)),
			    ("wh", POINTER(LAYER)),
			    ("uh", POINTER(LAYER)),
			    ("uo", POINTER(LAYER)),
			    ("wo", POINTER(LAYER)),
			    ("uf", POINTER(LAYER)),
			    ("wf", POINTER(LAYER)),
			    ("ui", POINTER(LAYER)),
			    ("wi", POINTER(LAYER)),
			    ("ug", POINTER(LAYER)),
			    ("wg", POINTER(LAYER)),
    			("softmax_tree", POINTER(TREE)),
			    ("workspace_size", c_size_t),
			    ("indexes_gpu", POINTER(c_int)),
			    ("z_gpu", POINTER(c_float)),
			    ("r_gpu", POINTER(c_float)),
			    ("h_gpu", POINTER(c_float)),
			    ("temp_gpu", POINTER(c_float)),
			    ("temp2_gpu", POINTER(c_float)),
			    ("temp3_gpu", POINTER(c_float)),
			    ("dh_gpu", POINTER(c_float)),
			    ("hh_gpu", POINTER(c_float)),
			    ("prev_cell_gpu", POINTER(c_float)),
			    ("cell_gpu", POINTER(c_float)),
			    ("f_gpu", POINTER(c_float)),
			    ("i_gpu", POINTER(c_float)),
			    ("g_gpu", POINTER(c_float)),
			    ("o_gpu", POINTER(c_float)),
			    ("c_gpu", POINTER(c_float)),
			    ("dc_gpu;", POINTER(c_float)),
			    ("m_gpu", POINTER(c_float)),
			    ("v_gpu", POINTER(c_float)),
			    ("bias_m_gpu", POINTER(c_float)),
			    ("scale_m_gpu", POINTER(c_float)),
			    ("bias_v_gpu", POINTER(c_float)),
			    ("scale_v_gpu", POINTER(c_float)),
			    ("combine_gpu", POINTER(c_float)),
			    ("combine_delta_gpu", POINTER(c_float)),
			    ("prev_state_gpu", POINTER(c_float)),
			    ("forgot_state_gpu", POINTER(c_float)),
			    ("forgot_delta_gpu", POINTER(c_float)),
			    ("state_gpu", POINTER(c_float)),
			    ("state_delta_gpu", POINTER(c_float)),
			    ("gate_gpu", POINTER(c_float)),
			    ("gate_delta_gpu", POINTER(c_float)),
			    ("save_gpu", POINTER(c_float)),
			    ("save_delta_gpu", POINTER(c_float)),
			    ("concat_gpu", POINTER(c_float)),
			    ("concat_delta_gpu", POINTER(c_float)),
			    ("tran_weights_gpu", POINTER(c_float)),
			    ("n_coeff", c_int),
			    ("q_coeff", POINTER(c_float)),
			    ("q_coeff_gpu", POINTER(c_float)),
			    ("mean_gpu", POINTER(c_float)),
			    ("variance_gpu", POINTER(c_float)),
			    ("rolling_mean_gpu", POINTER(c_float)),
			    ("rolling_variance_gpu", POINTER(c_float)),
			    ("variance_delta_gpu", POINTER(c_float)),
			    ("mean_delta_gpu", POINTER(c_float)),
			    ("x_gpu", POINTER(c_float)),
			    ("x_norm_gpu", POINTER(c_float)),
			    ("weights_gpu", POINTER(c_float)),
			    ("weight_updates_gpu", POINTER(c_float)),
			    ("weight_change_gpu", POINTER(c_float)),
			    ("biases_gpu", POINTER(c_float)),
			    ("bias_updates_gpu", POINTER(c_float)),
			    ("bias_change_gpu", POINTER(c_float)),
			    ("scales_gpu", POINTER(c_float)),
			    ("scale_updates_gpu", POINTER(c_float)),
			    ("scale_change_gpu", POINTER(c_float)),
			    ("output_gpu", POINTER(c_float)),
			    ("loss_gpu", POINTER(c_float)),
			    ("delta_gpu", POINTER(c_float)),
			    ("rand_gpu", POINTER(c_float)),
			    ("squared_gpu", POINTER(c_float)),
			    ("norms_gpu", POINTER(c_float)),
			    ("offset_h", c_int),
			    ("offset_w", c_int),
			    ("offset_c", c_int),
			    ("srcTensorDesc", c_int64), #cudnnTensorDescriptor_t ~ 8 bytes
			    ("dstTensorDesc", c_int64), #cudnnTensorDescriptor_t ~ 8 bytes
			    ("dsrcTensorDesc", c_int64), #cudnnTensorDescriptor_t ~ 8 bytes
			    ("ddstTensorDesc", c_int64), #cudnnTensorDescriptor_t ~ 8 bytes
			    ("normTensorDesc", c_int64), #cudnnTensorDescriptor_t ~ 8 bytes
			    ("weightDesc", c_int64), #cudnnFilterDescriptor_t ~ 8 bytes
			    ("dweightDesc", c_int64), #cudnnFilterDescriptor_t ~ 8 bytes
			    ("convDesc", c_int64), #cudnnConvolutionDescriptor_t ~ 8 bytes
			    ("fw_algo", c_int32), #cudnnConvolutionFwdAlgo_t ~ 4 bytes
			    ("bd_algo", c_int32), #cudnnConvolutionBwdDataAlgo_t ~ 4 bytes
			    ("bf_algo", c_int32), #cudnnConvolutionBwdFilterAlgo_t ~ 4 bytes
			    ("mask_layer_softmax", c_int),
			    ]

NETWORK._fields_ = [("initializer", INITIALIZER),
				("pre_transformable", c_int),
    			("pre_transform", c_int),
    			("lower_bound", c_float),
    			("upper_bound", c_float),
    			("n_loss", c_int),
			    ("sub_loss", POINTER(c_float)),
			    ("sub_loss_id", POINTER(c_int)),
			    ("n", c_int),
			    ("batch", c_int),
			    ("seen", POINTER(c_size_t)),
			    ("t", POINTER(c_int)),
			    ("epoch", c_float),
			    ("subdivisions", c_int),
			    ("layers", POINTER(LAYER)),
			    ("output", POINTER(c_float)),
			    ("policy", c_int),
			    ("learning_rate", c_float),
			    ("momentum", c_float),
			    ("decay", c_float),
			    ("gamma", c_float),
			    ("scale", c_float),
			    ("power", c_float),
			    ("time_steps", c_int),
			    ("step", c_int),
			    ("max_batches", c_int),
			    ("scales", POINTER(c_float)),
			    ("steps", POINTER(c_int)),
			    ("num_steps", c_int),
			    ("burn_in", c_int),
			    ("optim", c_int),
			    ("B1", c_float),
			    ("B2", c_float),
			    ("eps", c_float),
			    ("inputs", c_int),
			    ("outputs", c_int),
			    ("truths", c_int),
			    ("notruth", c_int),
			    ("h", c_int),
			    ("w", c_int),
			    ("c", c_int),
			    ("max_crop", c_int),
			    ("min_crop", c_int),
			    ("max_ratio", c_float),
			    ("min_ratio", c_float),
			    ("center", c_int),
			    ("angle", c_float),
			    ("aspect", c_float),
			    ("exposure", c_float),
			    ("saturation", c_float),
			    ("hue", c_float),
			    ("jitter", c_float),
			    ("zoom", POINTER(c_float)),
			    ("hflip", c_float),
			    ("vflip", c_float),
			    ("solarize", c_float),
			    ("posterize", c_float),
			    ("noise", c_float),
			    ("cutout", CUTOUT_ARGS),
			    ("random", c_int),
			    ("gpu_index", c_int),
			    ("hierarchy", POINTER(TREE)),
			    ("input", POINTER(c_float)),
			    ("truth", POINTER(c_float)),
			    ("delta", POINTER(c_float)),
			    ("workspace", POINTER(c_float)),
			    ("train", c_int),
			    ("index", c_int),
			    ("cost", POINTER(c_float)),
			    ("clip", c_float),
			    ("input_gpu", POINTER(c_float)),
			    ("truth_gpu", POINTER(c_float)),
			    ("delta_gpu", POINTER(c_float)),
			    ("output_gpu", POINTER(c_float)),
				]

class AUGMENT_ARGS(Structure):
	_fields_ = [("w", c_int),
			    ("h", c_int),
			    ("scale", c_float),
			    ("rad", c_float),
			    ("dx", c_float),
			    ("dy", c_float),
			    ("aspect", c_float),
				]

class IMAGE(Structure):
	_fields_ = [("w", c_int),
			    ("h", c_int),
			    ("c", c_int),
			    ("data", POINTER(c_float)),
				]

class BOX(Structure):
	_fields_ = [("x", c_float),
				("y", c_float),
				("w", c_float),
				("h", c_float),
				]

class DETECTION(Structure):
	_fields_ = [("bbox", BOX),
				("classes", c_int),
				("prob", POINTER(c_float)),
				("mask", POINTER(c_float)),
				("objectness", c_float),
				("sort_class", c_int),
				]

class IM_BOX(Structure):
	_fields_ = [("id", c_char * 128),
				("bbox", BOX),
				]

class MATRIX(Structure):
	_fields_ = [("rows", c_int),
				("cols", c_int),
				("vals", POINTER(POINTER(c_float))),
				]

class DATA(Structure):
	_fields_ = [("w", c_int),
				("h", c_int),
				("x", MATRIX),
				("y", MATRIX),
				("shallow", c_int),
				("num_boxes", POINTER(c_int)),
				("boxes", POINTER(POINTER(BOX))),
				]

class LOAD_ARGS(Structure):
	_fields_ = [("threads", c_int),
				("thread_id", c_int),
			    ("random_array", POINTER(c_int)),
			    ("paths", POINTER(c_char_p)),
			    ("path", POINTER(c_char)),
			    ("n", c_int),
			    ("m", c_int),
			    ("labels", POINTER(c_char_p)),
			    ("h", c_int),
			    ("w", c_int),
			    ("out_w", c_int),
			    ("out_h", c_int),
			    ("nh", c_int),
			    ("nw", c_int),
			    ("num_boxes", c_int),
			    ("min", c_int),
			    ("max", c_int),
			    ("size", c_int),
			    ("classes", c_int),
			    ("background", c_int),
			    ("scale", c_int),
			    ("center", c_int),
			    ("coords", c_int),
			    ("jitter", c_float),
			    ("angle", c_float),
			    ("aspect", c_float),
			    ("saturation", c_float),
			    ("exposure", c_float),
			    ("hue", c_float),
			    ("zoom", POINTER(c_float)),
			    ("hflip", c_float),
			    ("vflip", c_float),
			    ("solarize", c_float),
			    ("posterize", c_float),
			    ("noise", c_float),
			    ("cutout", CUTOUT_ARGS),
			    ("d", POINTER(DATA)),
			    ("im", POINTER(IMAGE)),
			    ("resized", POINTER(IMAGE)),
			    ("type", c_int),
			    ("hierarchy", POINTER(TREE)),
			    ("label_dir", c_char_p),
			    ("truth", POINTER(c_float)),
				]

class BOX_LABEL(Structure):
	_fields_ = [("id", c_int),
			    ("x", c_float),
			    ("y", c_float),
			    ("w", c_float),
			    ("h", c_float),
			    ("left", c_float),
			    ("right", c_float),
			    ("top", c_float),
			    ("bottom", c_float),
				]

# pre-define NODE class to include in itself
class NODE(Structure):
	pass

# _fields_ definition of NODE
NODE._fields_ = [("val", c_void_p),
				("next", POINTER(NODE)),
				("prev", POINTER(NODE)),
				]

class LIST(Structure):
	_fields_ = [("size", c_int),
				("front", POINTER(NODE)),
				("back", POINTER(NODE)),
				]

### get lib functions

## layer.c
# void free_layer(layer);
free_layer = syndn_lib.free_layer
free_layer.argtypes = [POINTER(LAYER)]
free_layer.restype = None

## tree.c
# tree *read_tree(char *filename);
read_tree = syndn_lib.read_tree
read_tree.argtypes = [c_char_p]
read_tree.restype = POINTER(TREE)

# void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
hierarchy_predictions = syndn_lib.hierarchy_predictions
hierarchy_predictions.argtypes = [POINTER(c_float), c_int, POINTER(TREE), c_int, c_int]
hierarchy_predictions.restype = None

# void change_leaves(tree *t, char *leaf_list);
change_leaves = syndn_lib.change_leaves
change_leaves.argtypes = [POINTER(TREE), c_char_p]
change_leaves.restype = None


## data.c
# pthread_t load_data(load_args args);
load_data = syndn_lib.load_data
load_data.argtypes = [LOAD_ARGS]
load_data.restype = c_int64

# pthread_t load_data_in_thread(load_args args);
load_data_in_thread = syndn_lib.load_data_in_thread
load_data_in_thread.argtypes = [LOAD_ARGS]
load_data_in_thread.restype = c_int64

# char **get_labels(char *filename);
get_labels = syndn_lib.get_labels
get_labels.argtypes = [c_char_p]
get_labels.restype = POINTER(c_char_p)

# char **get_labels_with_n(char *filename, int* n);
get_labels_with_n = syndn_lib.get_labels_with_n
get_labels_with_n.argtypes = [c_char_p, POINTER(c_int)]
get_labels_with_n.restype = POINTER(c_char_p)

# void get_next_batch(data d, int n, int offset, float *X, float *y);
get_next_batch = syndn_lib.get_next_batch
get_next_batch.argtypes = [DATA, c_int, c_int, POINTER(c_float), POINTER(c_float)]
get_next_batch.restype = None

# void summarize_data_augmentation_options(load_args args);
summarize_data_augmentation_options = syndn_lib.summarize_data_augmentation_options
summarize_data_augmentation_options.argtypes = [LOAD_ARGS]
summarize_data_augmentation_options.restype = None

# void free_data(data d);
free_data = syndn_lib.free_data
free_data.argtypes = [DATA]
free_data.restype = None

# data copy_data(data d);
copy_data = syndn_lib.copy_data
copy_data.argtypes = [DATA]
copy_data.restype = DATA

# data concat_data(data d1, data d2);
concat_data = syndn_lib.concat_data
concat_data.argtypes = [DATA, DATA]
concat_data.restype = DATA

# data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
load_data_old = syndn_lib.load_data_old
load_data_old.argtypes = [POINTER(c_char_p), c_int, c_int, POINTER(c_char_p), c_int, c_int, c_int]
load_data_old.restype = DATA

# box_label *read_boxes(char *filename, int *n);
read_boxes = syndn_lib.read_boxes
read_boxes.argtypes = [c_char_p, POINTER(c_int)]
read_boxes.restype = POINTER(BOX_LABEL)

# list *get_paths(char *filename);
get_paths = syndn_lib.get_paths
get_paths.argtypes = [c_char_p]
get_paths.restype = POINTER(LIST)

## option_list.c
# char *option_find_str(list *l, char *key, char *def);
option_find_str = syndn_lib.option_find_str
option_find_str.argtypes = [POINTER(LIST), c_char_p, c_char_p]
option_find_str.restype = c_char_p

# char *option_find_str_quiet(list *l, char *key, char *def);
option_find_str_quiet = syndn_lib.option_find_str_quiet
option_find_str_quiet.argtypes = [POINTER(LIST), c_char_p, c_char_p]
option_find_str_quiet.restype = c_char_p

# void option_find_str_series(list *l, char *key, int* num, char*** series);
option_find_str_series = syndn_lib.option_find_str_series
option_find_str_series.argtypes = [POINTER(LIST), c_char_p, POINTER(c_int), POINTER(POINTER(c_char_p))]
option_find_str_series.restype = None

# int option_find_int(list *l, char *key, int def);
option_find_int = syndn_lib.option_find_int
option_find_int.argtypes = [POINTER(LIST), c_char_p, c_int]
option_find_int.restype = c_int

# int option_find_int_quiet(list *l, char *key, int def);
option_find_int_quiet = syndn_lib.option_find_int_quiet
option_find_int_quiet.argtypes = [POINTER(LIST), c_char_p, c_int]
option_find_int_quiet.restype = c_int

# void option_find_int_series(list *l, char *key, int* num, int** series);
option_find_int_series = syndn_lib.option_find_int_series
option_find_int_series.argtypes = [POINTER(LIST), c_char_p, POINTER(c_int), POINTER(POINTER(c_int))]
option_find_int_series.restype = None

# void option_find_float_series(list *l, char *key, int* num, float** series);
option_find_float_series = syndn_lib.option_find_float_series
option_find_float_series.argtypes = [POINTER(LIST), c_char_p, POINTER(c_int), POINTER(POINTER(c_float))]
option_find_float_series.restype = None

# list *read_module_cfg(char *filename);
read_module_cfg = syndn_lib.read_module_cfg
read_module_cfg.argtypes = [c_char_p]
read_module_cfg.restype = POINTER(LIST)

# list *read_data_cfg(char *filename);
read_data_cfg = syndn_lib.read_data_cfg
read_data_cfg.argtypes = [c_char_p]
read_data_cfg.restype = POINTER(LIST)

# metadata get_metadata(char *file);
get_metadata = syndn_lib.get_metadata
get_metadata.argtypes = [c_char_p]
get_metadata.restype = METADATA

## parser.c
# void save_convolutional_weights(layer l, FILE *fp);
save_convolutional_weights = syndn_lib.save_convolutional_weights
save_convolutional_weights.argtypes = [LAYER, POINTER(216 * c_byte)] #FILE ~ 216 bytes
save_convolutional_weights.restype = None

# void save_batchnorm_weights(layer l, FILE *fp);
save_batchnorm_weights = syndn_lib.save_batchnorm_weights
save_batchnorm_weights.argtypes = [LAYER, POINTER(216 * c_byte)] #FILE ~ 216 bytes
save_batchnorm_weights.restype = None

# void save_connected_weights(layer l, FILE *fp);
save_connected_weights = syndn_lib.save_connected_weights
save_connected_weights.argtypes = [LAYER, POINTER(216 * c_byte)] #FILE ~ 216 bytes
save_connected_weights.restype = None

# void save_weights(network *net, char *filename);
save_weights = syndn_lib.save_weights
save_weights.argtypes = [POINTER(NETWORK), c_char_p]
save_weights.restype = None

# void load_modular_weights(network* net, char* modules);
load_modular_weights = syndn_lib.load_modular_weights
load_modular_weights.argtypes = [POINTER(NETWORK), c_char_p]
load_modular_weights.restype = None

# void load_weights(network *net, char *filename);
load_weights = syndn_lib.load_weights
load_weights.argtypes = [POINTER(NETWORK), c_char_p]
load_weights.restype = None

# void save_weights_upto(network *net, char *filename, int cutoff);
save_weights_upto = syndn_lib.save_weights_upto
save_weights_upto.argtypes = [POINTER(NETWORK), c_char_p, c_int]
save_weights_upto.restype = None

# void load_weights_upto(network *net, char *filename, int start, int cutoff);
load_weights_upto = syndn_lib.load_weights_upto
load_weights_upto.argtypes = [POINTER(NETWORK), c_char_p, c_int, c_int]
load_weights_upto.restype = None

# void print_layer(layer l);
print_layer = syndn_lib.print_layer
print_layer.argtypes = [LAYER]
print_layer.restype = None

# void print_optimizer(network* net);
print_optimizer = syndn_lib.print_optimizer
print_optimizer.argtypes = [POINTER(NETWORK)]
print_optimizer.restype = None

# network *parse_network_cfg(char *filename, int batch);
parse_network_cfg = syndn_lib.parse_network_cfg
parse_network_cfg.argtypes = [c_char_p, c_int]
parse_network_cfg.restype = POINTER(NETWORK)

# list *read_cfg(char *filename);
read_cfg = syndn_lib.read_cfg
read_cfg.argtypes = [c_char_p]
read_cfg.restype = POINTER(LIST)


## utils.c
# double what_time_is_it_now();
what_time_is_it_now = syndn_lib.what_time_is_it_now
what_time_is_it_now.argtypes = None
what_time_is_it_now.restype = c_double

# unsigned char *read_file(char *filename);
read_file = syndn_lib.read_file
read_file.argtypes = [c_char_p]
read_file.restype = POINTER(c_ubyte)

# char *find_char_arg(int argc, char **argv, char *arg, char *def);
find_char_arg = syndn_lib.find_char_arg
find_char_arg.argtypes = [c_int, POINTER(c_char_p), c_char_p, c_char_p]
find_char_arg.restype = c_char_p

# char *find_char_2arg(int argc, char **argv, char *arg1, char *arg2, char *def);
find_char_2arg = syndn_lib.find_char_2arg
find_char_2arg.argtypes = [c_int, POINTER(c_char_p), c_char_p, c_char_p, c_char_p]
find_char_2arg.restype = c_char_p

# char *basecfg(char *cfgfile);
basecfg = syndn_lib.basecfg
basecfg.argtypes = [c_char_p]
basecfg.restype = c_char_p

# char *fgetl(FILE *fp);
fgetl = syndn_lib.fgetl
fgetl.argtypes = [POINTER(216 * c_byte)]
fgetl.restype = c_char_p

# void find_replace(char *str, char *orig, char *rep, char *output);
find_replace = syndn_lib.find_replace
find_replace.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p]
find_replace.restype = None

# void free_ptrs(void **ptrs, int n);
free_ptrs = syndn_lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]
free_ptrs.restype = None

# void merge(float* arr, int l, int m, int r, int* sorted_index);
merge = syndn_lib.merge
merge.argtypes = [POINTER(c_float), c_int, c_int, c_int, POINTER(c_int)]
merge.restype = None

# void merge_sort(float* arr, int l, int r, int* sorted_index);
merge_sort = syndn_lib.merge_sort
merge_sort.argtypes = [POINTER(c_float), c_int, c_int, POINTER(c_int)]
merge_sort.restype = None

# void scale_array(float *a, int n, float s);
scale_array = syndn_lib.scale_array
scale_array.argtypes = [POINTER(c_float), c_int, c_float]
scale_array.restype = None

# void strip(char *s);
strip = syndn_lib.strip
strip.argtypes = [c_char_p]
strip.restype = None

# void top_k(float *a, int n, int k, int *index);
top_k = syndn_lib.top_k
top_k.argtypes = [POINTER(c_float), c_int, c_int, POINTER(c_int)]
top_k.restype = None

# void clear_prev_line(FILE* stream);
clear_prev_line = syndn_lib.clear_prev_line
clear_prev_line.argtypes = [POINTER(216 * c_byte)]
clear_prev_line.restype = None

# void clear_n_prev_lines(FILE* stream, int n);
clear_n_prev_lines = syndn_lib.clear_n_prev_lines
clear_n_prev_lines.argtypes = [POINTER(216 * c_byte), c_int]
clear_n_prev_lines.restype = None

# void error(const char *s);
error = syndn_lib.error
error.argtypes = [c_char_p]
error.restype = None

# void normalize_array(float *a, int n);
normalize_array = syndn_lib.normalize_array
normalize_array.argtypes = [POINTER(c_float), c_int]
normalize_array.restype = None

# void call_reproducible_rand_array(int** rand_array, size_t n);
call_reproducible_rand_array = syndn_lib.call_reproducible_rand_array
call_reproducible_rand_array.argtypes = [POINTER(POINTER(c_int)), c_size_t]
call_reproducible_rand_array.restype = None

# int *read_map(char *filename);
read_map = syndn_lib.read_map
read_map.argtypes = [c_char_p]
read_map.restype = POINTER(c_int)

# int max_index(float *a, int n);
max_index = syndn_lib.max_index
max_index.argtypes = [POINTER(c_float), c_int]
max_index.restype = c_int

# int max_int_index(int *a, int n);
max_int_index = syndn_lib.max_int_index
max_int_index.argtypes = [POINTER(c_int), c_int]
max_int_index.restype = c_int

# int sample_array(float *a, int n);
sample_array = syndn_lib.sample_array
sample_array.argtypes = [POINTER(c_float), c_int]
sample_array.restype = c_int

# int *read_intlist(char *s, int *n, int d);
read_intlist = syndn_lib.read_intlist
read_intlist.argtypes = [c_char_p, POINTER(c_int), c_int]
read_intlist.restype = POINTER(c_int)

# int find_int_arg(int argc, char **argv, char *arg, int def);
find_int_arg = syndn_lib.find_int_arg
find_int_arg.argtypes = [c_int, POINTER(c_char_p), c_char_p, c_int]
find_int_arg.restype = c_int

# int find_int_2arg(int argc, char **argv, char *arg1, char *arg2, int def);
find_int_2arg = syndn_lib.find_int_2arg
find_int_2arg.argtypes = [c_int, POINTER(c_char_p), c_char_p, c_char_p, c_int]
find_int_2arg.restype = c_int

# int find_arg(int argc, char* argv[], char *arg);
find_arg = syndn_lib.find_arg
find_arg.argtypes = [c_int, POINTER(c_char_p), c_char_p]
find_arg.restype = c_int

# int find_2arg(int argc, char* argv[], char *arg1, char *arg2);
find_2arg = syndn_lib.find_2arg
find_2arg.argtypes = [c_int, POINTER(c_char_p), c_char_p, c_char_p]
find_2arg.restype = c_int

# float mse_array(float *a, int n);
mse_array = syndn_lib.mse_array
mse_array.argtypes = [POINTER(c_float), c_int]
mse_array.restype = c_float

# float variance_array(float *a, int n);
variance_array = syndn_lib.variance_array
variance_array.argtypes = [POINTER(c_float), c_int]
variance_array.restype = c_float

# float mag_array(float *a, int n);
mag_array = syndn_lib.mag_array
mag_array.argtypes = [POINTER(c_float), c_int]
mag_array.restype = c_float

# float mean_array(float *a, int n);
mean_array = syndn_lib.mean_array
mean_array.argtypes = [POINTER(c_float), c_int]
mean_array.restype = c_float

# float sum_array(float *a, int n);
sum_array = syndn_lib.sum_array
sum_array.argtypes = [POINTER(c_float), c_int]
sum_array.restype = c_float

# float sec(clock_t clocks);
sec = syndn_lib.sec
sec.argtypes = [c_int64] #clock_t ~ 8 bytes
sec.restype = c_float

# float rand_normal();
rand_normal = syndn_lib.rand_normal
rand_normal.argtypes = None
rand_normal.restype = c_float

# float rand_uniform(float min, float max);
rand_uniform = syndn_lib.rand_uniform
rand_uniform.argtypes = [c_float, c_float]
rand_uniform.restype = c_float

# float find_float_arg(int argc, char **argv, char *arg, float def);
find_float_arg = syndn_lib.find_float_arg
find_float_arg.argtypes = [c_int, POINTER(c_char_p), c_char_p, c_float]
find_float_arg.restype = c_float

# float find_float_2arg(int argc, char **argv, char *arg1, char* arg2, float def);
find_float_2arg = syndn_lib.find_float_2arg
find_float_2arg.argtypes = [c_int, POINTER(c_char_p), c_char_p, c_char_p, c_float]
find_float_2arg.restype = c_float

# size_t rand_size_t();
rand_size_t = syndn_lib.rand_size_t
rand_size_t.argtypes = None
rand_size_t.restype = c_size_t

## network.c
# void forward_network(network *net);
forward_network = syndn_lib.forward_network
forward_network.argtypes = [POINTER(NETWORK)]
forward_network.restype = None

# void backward_network(network *net);
backward_network = syndn_lib.backward_network
backward_network.argtypes = [POINTER(NETWORK)]
backward_network.restype = None

# void update_network(network *net);
update_network = syndn_lib.update_network
update_network.argtypes = [POINTER(NETWORK)]
update_network.restype = None

# void sync_nets(network **nets, int n, int interval);
sync_nets = syndn_lib.sync_nets
sync_nets.argtypes = [POINTER(POINTER(NETWORK)), c_int, c_int]
sync_nets.restype = None

# void harmless_update_network_gpu(network *net);
harmless_update_network_gpu = syndn_lib.harmless_update_network_gpu
harmless_update_network_gpu.argtypes = [POINTER(NETWORK)]
harmless_update_network_gpu.restype = None

# void pre_transform_conv_params(network *net);
pre_transform_conv_params = syndn_lib.pre_transform_conv_params
pre_transform_conv_params.argtypes = [POINTER(NETWORK)]
pre_transform_conv_params.restype = None

# void swap_weight_transform(layer *l);
swap_weight_transform = syndn_lib.swap_weight_transform
swap_weight_transform.argtypes = [POINTER(LAYER)]
swap_weight_transform.restype = None

# void free_network(network *net);
free_network = syndn_lib.free_network
free_network.argtypes = [POINTER(NETWORK)]
free_network.restype = None

# void top_predictions(network *net, int n, int *index);
top_predictions = syndn_lib.top_predictions
top_predictions.argtypes = [POINTER(NETWORK), c_int, POINTER(c_int)]
top_predictions.restype = None

# void save_training_info(FILE* file, network* net, int header, int N);
save_training_info = syndn_lib.save_training_info
save_training_info.argtypes = [POINTER(c_byte * 216), POINTER(NETWORK), c_int, c_int]
save_training_info.restype = None

# void visualize_network(network *net);
visualize_network = syndn_lib.visualize_network
visualize_network.argtypes = [POINTER(NETWORK)]
visualize_network.restype = None

# void copy_detection(detection* dst, detection* src, int n_classes, size_t nboxes);
copy_detection = syndn_lib.copy_detection
copy_detection.argtypes = [POINTER(DETECTION), POINTER(DETECTION), c_int, c_size_t]
copy_detection.restype = None

# void free_detections(detection *dets, int n);
free_detections = syndn_lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]
free_detections.restype = None

# void reset_network_state(network *net, int b);
reset_network_state = syndn_lib.reset_network_state
reset_network_state.argtypes = [POINTER(NETWORK), c_int]
reset_network_state.restype = None

# float *network_accuracies(network *net, data d, int n);
network_accuracies = syndn_lib.network_accuracies
network_accuracies.argtypes = [POINTER(c_float), DATA, c_int]
network_accuracies.restype = POINTER(c_float)

# float train_network_datum(network *net);
train_network_datum = syndn_lib.train_network_datum
train_network_datum.argtypes = [POINTER(NETWORK)]
train_network_datum.restype = c_float

# float train_networks(network **nets, int n, data d, int interval);
train_networks = syndn_lib.train_networks
train_networks.argtypes = [POINTER(POINTER(NETWORK)), c_int, DATA, c_int]
train_networks.restype = c_float

# float train_network(network *net, data d);
train_network = syndn_lib.train_network
train_network.argtypes = [POINTER(NETWORK), DATA]
train_network.restype = c_float

# float *network_predict(network *net, float *input);
network_predict = syndn_lib.network_predict
network_predict.argtypes = [POINTER(NETWORK), POINTER(c_float)]
network_predict.restype = POINTER(c_float)

# float get_current_rate(network *net);
get_current_rate = syndn_lib.get_current_rate
get_current_rate.argtypes = [POINTER(NETWORK)]
get_current_rate.restype = c_float

# char *get_layer_string(LAYER_TYPE a);
get_layer_string = syndn_lib.get_layer_string
get_layer_string.argtypes = [c_int]
get_layer_string.restype = c_char_p

# int resize_network(network *net, int w, int h);
resize_network = syndn_lib.resize_network
resize_network.argtypes = [POINTER(NETWORK), c_int, c_int]
resize_network.restype = c_int

# size_t get_current_batch(network *net);
get_current_batch = syndn_lib.get_current_batch
get_current_batch.argtypes = [POINTER(NETWORK)]
get_current_batch.restype = c_size_t

# layer get_network_output_layer(network *net);
get_network_output_layer = syndn_lib.get_network_output_layer
get_network_output_layer.argtypes = [POINTER(NETWORK)]
get_network_output_layer.restype = LAYER

# matrix network_predict_data(network *net, data test);
network_predict_data = syndn_lib.network_predict_data
network_predict_data.argtypes = [POINTER(NETWORK), DATA]
network_predict_data.restype = MATRIX

# image get_network_image_layer(network *net, int i);
get_network_image_layer = syndn_lib.get_network_image_layer
get_network_image_layer.argtypes = [POINTER(NETWORK), c_int]
get_network_image_layer.restype = IMAGE

# image get_network_image(network *net);
get_network_image = syndn_lib.get_network_image
get_network_image.argtypes = [POINTER(NETWORK)]
get_network_image.restype = IMAGE

# detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
get_network_boxes = syndn_lib.get_network_boxes
get_network_boxes.argtypes = [POINTER(NETWORK), c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

# network *load_network(char *cfg, char *weights, char* modules, int clear, int batch);
load_network = syndn_lib.load_network
load_network.argtypes = [c_char_p, c_char_p, c_char_p, c_int, c_int]
load_network.restype = POINTER(NETWORK)

# load_args get_base_args(network *net);
get_base_args = syndn_lib.get_base_args
get_base_args.argtypes = [POINTER(NETWORK)]
get_base_args.restype = LOAD_ARGS

## blas.c
# float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
dot_cpu = syndn_lib.dot_cpu
dot_cpu.argtypes = [c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]
dot_cpu.restype = c_float

# void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
axpy_cpu = syndn_lib.axpy_cpu
axpy_cpu.argtypes = [c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int]
axpy_cpu.restype = None

# void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
copy_cpu = syndn_lib.copy_cpu
copy_cpu.argtypes = [c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]
copy_cpu.restype = None

# void scal_cpu(int N, float ALPHA, float *X, int INCX);
scal_cpu = syndn_lib.scal_cpu
scal_cpu.argtypes = [c_int, c_float, POINTER(c_float), c_int]
scal_cpu.restype = None

# void fill_cpu(int N, float ALPHA, float * X, int INCX);
fill_cpu = syndn_lib.fill_cpu
fill_cpu.argtypes = [c_int, c_float, POINTER(c_float), c_int]
fill_cpu.restype = None

# void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
normalize_cpu = syndn_lib.normalize_cpu
normalize_cpu.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
normalize_cpu.restype = None

# void softmax(float *input, int n, float temp, int stride, float *output);
softmax = syndn_lib.softmax
softmax.argtypes = [POINTER(c_float), c_int, c_float, c_int, POINTER(c_float)]
softmax.restype = None


## image.c
# int best_3d_shift_r(image a, image b, int min, int max);
best_3d_shift_r = syndn_lib.best_3d_shift_r
best_3d_shift_r.argtypes = [POINTER(c_float), c_int, c_float, c_int, POINTER(c_float)]
best_3d_shift_r.restype = c_int

# int show_image(image p, const char *name, int ms);
show_image = syndn_lib.show_image
show_image.argtypes = [IMAGE, c_char_p, c_int]
show_image.restype = c_int

# void draw_label(image a, int r, int c, image label, const float *rgb);
draw_label = syndn_lib.draw_label
draw_label.argtypes = [IMAGE, c_int, c_int, IMAGE, POINTER(c_float)]
draw_label.restype = None

# void save_image(image im, const char *name);
save_image = syndn_lib.save_image
save_image.argtypes = [IMAGE, c_char_p]
save_image.restype = None

# void save_image_options(image im, const char *name, IMTYPE f, int quality);
save_image_options = syndn_lib.save_image_options
save_image_options.argtypes = [IMAGE, c_char_p, c_int, c_int]
save_image_options.restype = None

# void grayscale_image_3c(image im);
grayscale_image_3c = syndn_lib.grayscale_image_3c
grayscale_image_3c.argtypes = [IMAGE]
grayscale_image_3c.restype = None

# void normalize_image(image p);
normalize_image = syndn_lib.normalize_image
normalize_image.argtypes = [IMAGE]
normalize_image.restype = None

# void rgbgr_image(image im);
rgbgr_image = syndn_lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]
rgbgr_image.restype = None

# void censor_image(image im, int dx, int dy, int w, int h);
censor_image = syndn_lib.censor_image
censor_image.argtypes = [IMAGE, c_int, c_int, c_int, c_int]
censor_image.restype = None

# void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
draw_box_width = syndn_lib.draw_box_width
draw_box_width.argtypes = [IMAGE, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_float]
draw_box_width.restype = None

# void composite_3d(char *f1, char *f2, char *out, int delta);
composite_3d = syndn_lib.composite_3d
composite_3d.argtypes = [c_char_p, c_char_p, c_char_p, c_int]
composite_3d.restype = None

# void constrain_image(image im);
constrain_image = syndn_lib.constrain_image
constrain_image.argtypes = [IMAGE]
constrain_image.restype = None

# void find_min_max(float a, float b, float c, float d, float* min, float* max);
find_min_max = syndn_lib.find_min_max
find_min_max.argtypes = [c_float, c_float, c_float, c_float, POINTER(c_float), POINTER(c_float)]
find_min_max.restype = None

# void solarize_image(image im, float threshold);
solarize_image = syndn_lib.solarize_image
solarize_image.argtypes = [IMAGE, c_float]
solarize_image.restype = None

# void posterize_image(image im, int levels);
posterize_image = syndn_lib.posterize_image
posterize_image.argtypes = [IMAGE, c_int]
posterize_image.restype = None

# void random_distort_image_extend(image im, float solarize, float posterize, float noise);
random_distort_image_extend = syndn_lib.random_distort_image_extend
random_distort_image_extend.argtypes = [IMAGE, c_float, c_float, c_float]
random_distort_image_extend.restype = None

# void flip_image_x(image a, int h_flip, int v_flip);
flip_image_x = syndn_lib.flip_image_x
flip_image_x.argtypes = [IMAGE, c_int, c_int]
flip_image_x.restype = None

# void flip_image_horizontal(image a);
flip_image_horizontal = syndn_lib.flip_image_horizontal
flip_image_horizontal.argtypes = [IMAGE]
flip_image_horizontal.restype = None

# void flip_image_vertical(image a);
flip_image_vertical = syndn_lib.flip_image_vertical
flip_image_vertical.argtypes = [IMAGE]
flip_image_vertical.restype = None

# void ghost_image(image source, image dest, int dx, int dy);
ghost_image = syndn_lib.ghost_image
ghost_image.argtypes = [IMAGE, IMAGE, c_int, c_int]
ghost_image.restype = None

# void random_distort_image(image im, float hue, float saturation, float exposure);
random_distort_image = syndn_lib.random_distort_image
random_distort_image.argtypes = [IMAGE, c_float, c_float, c_float]
random_distort_image.restype = None

# void random_cutout_image(image im, cutout_args cutout);
random_cutout_image = syndn_lib.random_cutout_image
random_cutout_image.argtypes = [IMAGE, CUTOUT_ARGS]
random_cutout_image.restype = None

# void fill_image(image m, float s);
fill_image = syndn_lib.fill_image
fill_image.argtypes = [IMAGE, c_float]
fill_image.restype = None

# void rotate_image_cw(image im, int times);
rotate_image_cw = syndn_lib.rotate_image_cw
rotate_image_cw.argtypes = [IMAGE, c_int]
rotate_image_cw.restype = None

# void draw_detection(image im, im_box ib, float red, float green, float blue);
draw_detection = syndn_lib.draw_detection
draw_detection.argtypes = [IMAGE, IM_BOX, c_float, c_float, c_float]
draw_detection.restype = None

# void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);
draw_detections = syndn_lib.draw_detections
draw_detections.argtypes = [IMAGE, POINTER(DETECTION), c_int, c_float, POINTER(c_char_p), POINTER(POINTER(IMAGE)), c_int]
draw_detections.restype = None

# void free_image(image m);
free_image = syndn_lib.free_image
free_image.argtypes = [IMAGE]
free_image.restype = None

# image get_label(image **characters, char *string, int size);
get_label = syndn_lib.get_label
get_label.argtypes = [POINTER(POINTER(IMAGE)), c_char_p, c_int]
get_label.restype = IMAGE

# image make_random_image(int w, int h, int c);
make_random_image = syndn_lib.make_random_image
make_random_image.argtypes = [c_int, c_int, c_int]
make_random_image.restype = IMAGE

# image load_image(char *filename, int w, int h, int c);
load_image = syndn_lib.load_image
load_image.argtypes = [c_char_p, c_int, c_int, c_int]
load_image.restype = IMAGE

# image load_image_color(char *filename, int w, int h);
load_image_color = syndn_lib.load_image_color
load_image_color.argtypes = [c_char_p, c_int, c_int]
load_image_color.restype = IMAGE

# image make_image(int w, int h, int c);
make_image = syndn_lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

# image resize_image(image im, int w, int h);
resize_image = syndn_lib.resize_image
resize_image.argtypes = [IMAGE, c_int, c_int]
resize_image.restype = IMAGE

# image letterbox_image(image im, int w, int h);
letterbox_image = syndn_lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

# image crop_image(image im, int dx, int dy, int w, int h);
crop_image = syndn_lib.crop_image
crop_image.argtypes = [IMAGE, c_int, c_int, c_int, c_int]
crop_image.restype = IMAGE

# image center_crop_image(image im, int w, int h);
center_crop_image = syndn_lib.center_crop_image
center_crop_image.argtypes = [IMAGE, c_int, c_int]
center_crop_image.restype = IMAGE

# image resize_min(image im, int min);
resize_min = syndn_lib.resize_min
resize_min.argtypes = [IMAGE, c_int]
resize_min.restype = IMAGE

# image resize_max(image im, int max);
resize_max = syndn_lib.resize_max
resize_max.argtypes = [IMAGE, c_int]
resize_max.restype = IMAGE

# image threshold_image(image im, float thresh);
threshold_image = syndn_lib.threshold_image
threshold_image.argtypes = [IMAGE, c_float]
threshold_image.restype = IMAGE

# image mask_to_rgb(image mask);
mask_to_rgb = syndn_lib.mask_to_rgb
mask_to_rgb.argtypes = [IMAGE]
mask_to_rgb.restype = IMAGE

# image copy_image(image p);
copy_image = syndn_lib.copy_image
copy_image.argtypes = [IMAGE]
copy_image.restype = IMAGE

# image rotate_image(image m, float rad);
rotate_image = syndn_lib.rotate_image
rotate_image.argtypes = [IMAGE, c_float]
rotate_image.restype = IMAGE

# image rotate_image_preserved(image im, float rad);
rotate_image_preserved = syndn_lib.rotate_image_preserved
rotate_image_preserved.argtypes = [IMAGE, c_float]
rotate_image_preserved.restype = IMAGE

# image float_to_image(int w, int h, int c, float *data);
float_to_image = syndn_lib.float_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE

# image grayscale_image(image im);
grayscale_image = syndn_lib.grayscale_image
grayscale_image.argtypes = [IMAGE]
grayscale_image.restype = IMAGE

# image real_to_heat_image(float* data, int w, int h);
real_to_heat_image = syndn_lib.real_to_heat_image
real_to_heat_image.argtypes = [POINTER(c_float), c_int, c_int]
real_to_heat_image.restype = IMAGE

# image **load_alphabet();
load_alphabet = syndn_lib.load_alphabet
load_alphabet.argtypes = None
load_alphabet.restype = POINTER(POINTER(IMAGE))


## image_kernels.cu
# void flip_image_x_gpu(float* im_data, int w, int h, int c, int h_flip, int v_flip);
flip_image_x_gpu = syndn_lib.flip_image_x_gpu
flip_image_x_gpu.argtypes = [POINTER(c_float), c_int, c_int, c_int, c_int, c_int]
flip_image_x_gpu.restype = None

# void random_distort_image_gpu(float* im_data, int w, int h, int c, float hue, float saturation, float exposure);
random_distort_image_gpu = syndn_lib.random_distort_image_gpu
random_distort_image_gpu.argtypes = [POINTER(c_float), c_int, c_int, c_int, c_float, c_float, c_float]
random_distort_image_gpu.restype = None

# void random_distort_image_extend_gpu(float* im_data, int w, int h, int c, float solarize, float posterize, float noise);
random_distort_image_extend_gpu = syndn_lib.random_distort_image_extend_gpu
random_distort_image_extend_gpu.argtypes = [POINTER(c_float), c_int, c_int, c_int, c_float, c_float, c_float]
random_distort_image_extend_gpu.restype = None

# void random_cutout_image_gpu(float* im_data, int w, int h, int c, cutout_args cutout);
random_cutout_image_gpu = syndn_lib.random_cutout_image_gpu
random_cutout_image_gpu.argtypes = [POINTER(c_float), c_int, c_int, c_int, CUTOUT_ARGS]
random_cutout_image_gpu.restype = None

# void resize_image_gpu(float* input, int iw, int ih, float* output, int ow, int oh, int oc);
resize_image_gpu = syndn_lib.resize_image_gpu
resize_image_gpu.argtypes = [POINTER(c_float), c_int, c_int, POINTER(c_float), c_int, c_int, c_int]
resize_image_gpu.restype = None

## blas_kernels.cu
# void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
axpy_gpu = syndn_lib.axpy_gpu
axpy_gpu.argtypes = [c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int]
axpy_gpu.restype = None

# void fill_gpu(int N, float ALPHA, float * X, int INCX);
fill_gpu = syndn_lib.fill_gpu
fill_gpu.argtypes = [c_int, c_float, POINTER(c_float), c_int]
fill_gpu.restype = None

# void fill_int_gpu(int N, int ALPHA, int * X, int INCX);
fill_int_gpu = syndn_lib.fill_int_gpu
fill_int_gpu.argtypes = [c_int, c_float, POINTER(c_int), c_int]
fill_int_gpu.restype = None

# void scal_gpu(int N, float ALPHA, float * X, int INCX);
scal_gpu = syndn_lib.scal_gpu
scal_gpu.argtypes = [c_int, c_float, POINTER(c_float), c_int]
scal_gpu.restype = None

# void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
copy_gpu = syndn_lib.copy_gpu
copy_gpu.argtypes = [c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]
copy_gpu.restype = None

# void floorf_gpu(int N, float * X, int INCX);
floorf_gpu = syndn_lib.floorf_gpu
floorf_gpu.argtypes = [c_int, POINTER(c_float), c_int]
floorf_gpu.restype = None


## cuda.c
# int *cuda_make_int_array(int *x, size_t n);
cuda_make_int_array = syndn_lib.cuda_make_int_array
cuda_make_int_array.argtypes = [POINTER(c_int), c_size_t]
cuda_make_int_array.restype = POINTER(c_int)

# void cuda_set_device(int n);
cuda_set_device = syndn_lib.cuda_set_device
cuda_set_device.argtypes = [c_int]
cuda_set_device.restype = None

# void cuda_free(float *x_gpu);
cuda_free = syndn_lib.cuda_free
cuda_free.argtypes = [POINTER(c_float)]
cuda_free.restype = None

# void cuda_free_int(int *x_gpu);
cuda_free_int = syndn_lib.cuda_free_int
cuda_free_int.argtypes = [POINTER(c_int)]
cuda_free_int.restype = None

# void cuda_push_array(float *x_gpu, float *x, size_t n);
cuda_push_array = syndn_lib.cuda_push_array
cuda_push_array.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t]
cuda_push_array.restype = None

# void cuda_pull_array(float *x_gpu, float *x, size_t n);
cuda_pull_array = syndn_lib.cuda_pull_array
cuda_pull_array.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t]
cuda_pull_array.restype = None

# void cuda_push_int_array(int *x_gpu, int *x, size_t n);
cuda_push_int_array = syndn_lib.cuda_push_int_array
cuda_push_int_array.argtypes = [POINTER(c_int), POINTER(c_int), c_size_t]
cuda_push_int_array.restype = None

# void cuda_pull_int_array(int *x_gpu, int *x, size_t n);
cuda_pull_int_array = syndn_lib.cuda_pull_int_array
cuda_pull_int_array.argtypes = [POINTER(c_int), POINTER(c_int), c_size_t]
cuda_pull_int_array.restype = None

# float cuda_mag_array(float *x_gpu, size_t n);
cuda_mag_array = syndn_lib.cuda_mag_array
cuda_mag_array.argtypes = [POINTER(c_float), c_size_t]
cuda_mag_array.restype = c_float

# float* cuda_make_array(float *x, size_t n);
cuda_make_array = syndn_lib.cuda_make_array
cuda_make_array.argtypes = [POINTER(c_float), c_size_t]
cuda_make_array.restype = POINTER(c_float)

## matrix.c
# float matrix_topk_accuracy(matrix truth, matrix guess, int k);
matrix_topk_accuracy = syndn_lib.matrix_topk_accuracy
matrix_topk_accuracy.argtypes = [MATRIX, MATRIX, c_int]
matrix_topk_accuracy.restype = c_float

# void matrix_to_csv(matrix m);
matrix_to_csv = syndn_lib.matrix_to_csv
matrix_to_csv.argtypes = [MATRIX]
matrix_to_csv.restype = None

# void matrix_add_matrix(matrix from, matrix to);
matrix_add_matrix = syndn_lib.matrix_add_matrix
matrix_add_matrix.argtypes = [MATRIX, MATRIX]
matrix_add_matrix.restype = None

# void scale_matrix(matrix m, float scale);
scale_matrix = syndn_lib.scale_matrix
scale_matrix.argtypes = [MATRIX, c_float]
scale_matrix.restype = None

# void free_matrix(matrix m);
free_matrix = syndn_lib.free_matrix
free_matrix.argtypes = [MATRIX]
free_matrix.restype = None

# matrix make_matrix(int rows, int cols);
make_matrix = syndn_lib.make_matrix
make_matrix.argtypes = [c_int, c_int]
make_matrix.restype = MATRIX

# matrix csv_to_matrix(char *filename);
csv_to_matrix = syndn_lib.csv_to_matrix
csv_to_matrix.argtypes = [c_char_p]
csv_to_matrix.restype = MATRIX

## connected_layer.c
# void denormalize_connected_layer(layer l);
denormalize_connected_layer = syndn_lib.denormalize_connected_layer
denormalize_connected_layer.argtypes = [LAYER]
denormalize_connected_layer.restype = None

# void statistics_connected_layer(layer l);
statistics_connected_layer = syndn_lib.statistics_connected_layer
statistics_connected_layer.argtypes = [LAYER]
statistics_connected_layer.restype = None


## convolutional_layer.c
# void denormalize_convolutional_layer(layer l);
denormalize_convolutional_layer = syndn_lib.denormalize_convolutional_layer
denormalize_convolutional_layer.argtypes = [LAYER]
denormalize_convolutional_layer.restype = None

# void rescale_weights(layer l, float scale, float trans);
rescale_weights = syndn_lib.rescale_weights
rescale_weights.argtypes = [LAYER, c_float, c_float]
rescale_weights.restype = None

# void rgbgr_weights(layer l);
rgbgr_weights = syndn_lib.rgbgr_weights
rgbgr_weights.argtypes = [LAYER]
rgbgr_weights.restype = None

# image *get_weights(layer l);
get_weights = syndn_lib.get_weights
get_weights.argtypes = [LAYER]
get_weights.restype = POINTER(IMAGE)

## demo.c
# void demo(char *cfgfile, char *weightfile, char* modules, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen, int cropx, int save_frame, int add_frame_count);
demo = syndn_lib.demo
demo.argtypes = [c_char_p, c_char_p, c_char_p, c_float, c_int, c_char_p, POINTER(c_char_p), c_int, c_int, c_char_p, c_int, c_float, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
demo.restype = None

# void demo_mf(char *cfgfile, char *weightfile, char* modules, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, int cropx, int save_frame, int add_frame_count);
demo_mf = syndn_lib.demo_mf
demo_mf.argtypes = [c_char_p, c_char_p, c_char_p, c_float, c_int, c_char_p, POINTER(c_char_p), c_int, c_int, c_char_p, c_int, c_float, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
demo_mf.restype = None

# int size_network(network *net);
size_network = syndn_lib.size_network
size_network.argtypes = [POINTER(NETWORK)]
size_network.restype = c_int


## detection_layer.c
# void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);
get_detection_detections = syndn_lib.get_detection_detections
get_detection_detections.argtypes = [LAYER, c_int, c_int, c_float, POINTER(DETECTION)]
get_detection_detections.restype = None


## multibox_layer.c
# int get_multibox_detections(layer l, int w, int h, int netw, int neth, float thresh, int relative, detection *dets);
get_multibox_detections = syndn_lib.get_multibox_detections
get_multibox_detections.argtypes = [LAYER, c_int, c_int, c_int, c_int, c_float, c_int, POINTER(DETECTION)]
get_multibox_detections.restype = c_int

## yolo_layer.c
# void zero_objectness(layer l);
zero_objectness = syndn_lib.zero_objectness
zero_objectness.argtypes = [LAYER]
zero_objectness.restype = None

# int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
get_yolo_detections = syndn_lib.get_yolo_detections
get_yolo_detections.argtypes = [LAYER, c_int, c_int, c_int, c_int, c_float, POINTER(c_int), c_float, c_int, POINTER(DETECTION)]
get_yolo_detections.restype = c_int


## region_layer.c
# void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
get_region_detections = syndn_lib.get_region_detections
get_region_detections.argtypes = [LAYER, c_int, c_int, c_int, c_int, c_float, POINTER(c_int), c_float, c_int, POINTER(DETECTION)]
get_region_detections.restype = c_int


## box.c
# float box_iou(box a, box b);
box_iou = syndn_lib.box_iou
box_iou.argtypes = [BOX, BOX]
box_iou.restype = c_float

# box float_to_box(float *f, int stride);
float_to_box = syndn_lib.float_to_box
float_to_box.argtypes = [POINTER(c_float), c_int]
float_to_box.restype = BOX

# void do_nms_obj(detection *dets, int total, int classes, float thresh);
do_nms_obj = syndn_lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
do_nms_obj.restype = None

# void do_nms_sort(detection *dets, int total, int classes, float thresh, NMS_MODE mode);
do_nms_sort = syndn_lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float, c_int]
do_nms_sort.restype = None


## image_opencv.cpp
# image get_image_from_stream(void *p);
get_image_from_stream = syndn_lib.get_image_from_stream
get_image_from_stream.argtypes = [c_void_p]
get_image_from_stream.restype = IMAGE

# void *open_video_stream(const char *f, int c, int w, int h, int fps);
open_video_stream = syndn_lib.open_video_stream
open_video_stream.argtypes = [c_char_p, c_int, c_int, c_int, c_int]
open_video_stream.restype = c_void_p

# void make_window(char *name, int w, int h, int fullscreen);
make_window = syndn_lib.make_window
make_window.argtypes = [c_char_p, c_int, c_int, c_int]
make_window.restype = None


## list.c
# void **list_to_array(list *l);
list_to_array = syndn_lib.list_to_array
list_to_array.argtypes = [POINTER(LIST)]
list_to_array.restype = POINTER(c_void_p)

# void free_list(list *l);
free_list = syndn_lib.free_list
free_list.argtypes = [POINTER(LIST)]
free_list.restype = None



if __name__ == "__main__":
	cuda_set_device(1)
	net = load_network("cfg/yolov2_sim.cfg".encode(), "backup/yolov2_sim_7200".encode(), None, 0, 0)
	im = load_image_color("samples/dog.jpg".encode(), 0, 0)
	make_window("prediction".encode(), 512, 512, 0)
	show_image(im, "prediction".encode(), 0)