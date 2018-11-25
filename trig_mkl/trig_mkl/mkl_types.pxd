cdef extern from "mkl_types.h":
    ctypedef long long int MKL_INT64
    ctypedef unsigned long long int MKL_UINT64
    ctypedef int MKL_INT
    ctypedef unsigned int MKL_UINT
    ctypedef long int MKL_LONG
    ctypedef unsigned char MKL_UINT8
    ctypedef char MKL_INT8
    ctypedef short MKL_INT16
    ctypedef int MKL_INT32
