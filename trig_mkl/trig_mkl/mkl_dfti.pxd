cdef extern from "mkl_dfti.h":
    ctypedef struct DFTI_DESCRIPTOR:
        pass
    ctypedef DFTI_DESCRIPTOR *DFTI_DESCRIPTOR_HANDLE;