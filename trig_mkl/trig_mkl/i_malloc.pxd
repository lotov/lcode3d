cdef extern from "i_malloc.h":
    cdef void *i_malloc(size_t size)
    cdef void *i_calloc(size_t nmemb, size_t size)
    cdef void *i_realloc(void *ptr, size_t size)
    cdef void i_free(void *ptr)