cdef extern from "mkl_service.h":
    cdef void* mkl_malloc(size_t size, int align)
    cdef void* mkl_calloc(size_t num, size_t size, int align)
    cdef void* mkl_realloc(void *ptr, size_t size)
    cdef void mkl_free(void *ptr)