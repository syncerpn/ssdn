from ctypes import CDLL, RTLD_GLOBAL
from ctypes import POINTER, c_size_t, c_int, c_float, c_char_p, pointer, Structure

ssdn_lib = CDLL("libssdn.so", RTLD_GLOBAL)

gpu_index = ssdn_lib.gpu_index
seed = ssdn_lib.seed

class IMAGE(Structure):
	_fields_ = [("w", c_int),
			    ("h", c_int),
			    ("c", c_int),
			    ("data", POINTER(c_float)),
				]

# void cuda_set_device(int n);
cuda_set_device = ssdn_lib.cuda_set_device
cuda_set_device.argtypes = [c_int]
cuda_set_device.restype = None

# float* cuda_make_array(float *x, size_t n);
cuda_make_array = ssdn_lib.cuda_make_array
cuda_make_array.argtypes = [POINTER(c_float), c_size_t]
cuda_make_array.restype = POINTER(c_float)

# void cuda_free(float *x_gpu);
cuda_free = ssdn_lib.cuda_free
cuda_free.argtypes = [POINTER(c_float)]
cuda_free.restype = None

# void cuda_push_array(float *x_gpu, float *x, size_t n);
cuda_push_array = ssdn_lib.cuda_push_array
cuda_push_array.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t]
cuda_push_array.restype = None

# void cuda_pull_array(float *x_gpu, float *x, size_t n);
cuda_pull_array = ssdn_lib.cuda_pull_array
cuda_pull_array.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t]
cuda_pull_array.restype = None

# void fill_gpu(int N, float ALPHA, float * X, int INCX);
fill_gpu = ssdn_lib.fill_gpu
fill_gpu.argtypes = [c_int, c_float, POINTER(c_float), c_int]
fill_gpu.restype = None

# TODO: add all funcs below

if __name__ == "__main__":
	cuda_set_device(0)
	f = (c_float*10)(list(range(10)))
	print(f)
	c = cuda_make_array(pointer(f), 10)
	fill_gpu(3, 20, c, 1)
	cuda_pull_array(c, pointer(f), 10)
	print(f)