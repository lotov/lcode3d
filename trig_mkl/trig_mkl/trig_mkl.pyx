import numpy as np
cimport numpy as np
from mkl_types cimport MKL_INT
from mkl_service cimport mkl_malloc, mkl_calloc, mkl_free
from mkl_dfti cimport DFTI_DESCRIPTOR_HANDLE
cimport mkl_trig_transforms as mkltt
from mkl cimport mkl_free_buffers
cimport cython
from cython.parallel import prange, parallel, threadid
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef class TrigTransform:
    def __cinit__(self, MKL_INT n, str tt_type = 'dst', int num_threads = 1):
        cdef np.ndarray[double, ndim=1, mode="c"] commit_arr
        cdef int i,
        cdef MKL_INT ir
        if n <= 0:
            raise ValueError("n must be greater than 0")
        if num_threads < 1:
            raise ValueError("num_threads must be greater than 1")
        self.num_threads = num_threads
        self.ipar = NULL
        self.dpar = NULL
        self.handles = NULL

        self.ipar = <MKL_INT **>mkl_calloc(self.num_threads, sizeof(MKL_INT *), 64)
        self.handles = <DFTI_DESCRIPTOR_HANDLE *>mkl_calloc(self.num_threads, sizeof(DFTI_DESCRIPTOR_HANDLE), 64)
        if self.ipar == NULL:
            raise MemoryError
        if self.handles == NULL:
            raise MemoryError
        for i in range(self.num_threads):
            self.ipar[i] = <MKL_INT *>mkl_calloc(128, sizeof(MKL_INT), 64)
            if self.ipar[i] == NULL:
                raise MemoryError

        if tt_type == 'dst':
            self.n = n + 1
            self.tt_type = mkltt.MKL_SINE_TRANSFORM
            self.dpar = <double *>mkl_malloc((self.n // 2 + 2) * sizeof(double), 64)
            self._full_array = np.zeros((self.n - 1, self.n + 1), dtype=np.double, order="c")
            self.array = self._full_array[:, 1:self.n]
        elif tt_type == 'dct':
            self.n = n - 1
            self.tt_type = mkltt.MKL_COSINE_TRANSFORM
            self.dpar = <double *>mkl_malloc((self.n + 2) * sizeof(double), 64)
            self._full_array = np.zeros((self.n + 1, self.n + 1), dtype=np.double, order="c")
            self.array = self._full_array
        else:
            raise ValueError("tt_type must be either 'dct' or 'dst'")

        if self.dpar == NULL:
            raise MemoryError

        for i in range(self.num_threads):
            mkltt.d_init_trig_transform(&self.n, &self.tt_type, self.ipar[i], self.dpar, &ir)
            if ir != 0:
                raise RuntimeError

        commit_arr = np.zeros(self.n + 1, dtype=np.double, order="c")
        for i in range(self.num_threads):
            mkltt.d_commit_trig_transform(&commit_arr[0], &self.handles[i], self.ipar[i], self.dpar, &ir)
            self.ipar[i][7] = 0
            if ir != 0:
                raise RuntimeError

    cdef MKL_INT _forward(self, double *data, int thread) nogil:
        cdef MKL_INT ir = 0
        mkltt.d_forward_trig_transform(data, &self.handles[thread], self.ipar[thread], self.dpar, &ir)
        return ir

    cdef MKL_INT _backward(self, double *data, int thread) nogil:
        cdef MKL_INT ir = 0
        mkltt.d_backward_trig_transform(data, &self.handles[thread], self.ipar[thread], self.dpar, &ir)
        return ir

    cpdef void dst_2d(TrigTransform self):
        assert self.tt_type == mkltt.MKL_SINE_TRANSFORM
        cdef int i, k
        # self._full_array is assumed to be set via self.array assignment
        # except to two columns
        with nogil, parallel(num_threads=self.num_threads):
            k = threadid()
            for i in prange(self.n - 1):
                self._full_array[i, 0] = self._full_array[i, self.n] = 0
                #if self._commit(&out2[i, 0], k) != 0 or self._forward(&out2[i, 0], k) != 0:
                #if self._forward(&out2[i, 0], k) != 0:
                #    break
                self._forward(&self._full_array[i, 0], k)  # error check below
        for i in range(self.num_threads):
            if self.ipar[i][6] != 0:
                raise RuntimeError
        # *= self.n  # MOVED TO OUTER CODE

    cpdef void dct_2d(TrigTransform self):
        assert self.tt_type == mkltt.MKL_COSINE_TRANSFORM
        cdef int i, k
        # self._full_array is assumed to be set via self.array assignment
        with nogil, parallel(num_threads=self.num_threads):
            k = threadid()
            for i in prange(self.n + 1):
                #if self._commit(&out2[i, 0], k) != 0 or self._forward(&out2[i, 0], k) != 0:
                #if self._forward(&self._full_array[i, 0], k) != 0:
                #    break
                self._forward(&self._full_array[i, 0], k)  # error check below
        for i in range(self.num_threads):
            if self.ipar[i][6] != 0:
                raise RuntimeError
        # *= self.n  # MOVED TO OUTER CODE

    cpdef void idct_2d(TrigTransform self):
        assert self.tt_type == mkltt.MKL_COSINE_TRANSFORM
        cdef int i, k
        # self._full_array is assumed to be set via self.array assignment
        with nogil, parallel(num_threads=self.num_threads):
            k = threadid()
            for i in prange(self.n + 1):
                #if self._commit(&out2[i, 0], k) != 0 or self._backward(&out2[i, 0], k) != 0:
                #if self._backward(&out2[i, 0], k) != 0:
                #    break
                self._backward(&self._full_array[i, 0], k)  # error check below
        for i in range(self.num_threads):
            if self.ipar[i][6] != 0:
                raise RuntimeError
        # *= 2  # MOVED TO OUTER CODE

    def __dealloc__(self):
        cdef MKL_INT ir
        cdef int i
        if self.handles != NULL:
            for i in range(self.num_threads):
                mkltt.free_trig_transform(&self.handles[i], self.ipar[i], &ir)
            mkl_free(self.handles)
        if self.ipar != NULL:
            for i in range(self.num_threads):
                mkl_free(self.ipar[i])
            mkl_free(self.ipar)
        if self.dpar != NULL:
            mkl_free(self.dpar)
        mkl_free_buffers()
