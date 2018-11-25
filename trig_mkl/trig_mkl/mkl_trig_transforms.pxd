from mkl_types cimport MKL_INT
from mkl_dfti cimport DFTI_DESCRIPTOR_HANDLE

cdef extern from "mkl_trig_transforms.h":
    cdef int MKL_SINE_TRANSFORM
    cdef int MKL_COSINE_TRANSFORM
    cdef void d_init_trig_transform(MKL_INT *n, MKL_INT *tt_type, MKL_INT *ipar, double *dpar, MKL_INT *stat)
    cdef void d_commit_trig_transform(double *f, DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT *ipar, double *dpar, MKL_INT *stat) nogil
    cdef void d_forward_trig_transform(double *f, DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT *ipar, double *dpar, MKL_INT *stat) nogil
    cdef void d_backward_trig_transform(double *f, DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT *ipar, double *dpar, MKL_INT *stat) nogil
    cdef void free_trig_transform(DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT *ipar, MKL_INT *stat);
