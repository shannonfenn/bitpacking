# cython: language_level=3
import numpy as np
cimport numpy as np


ctypedef np.uint64_t packed_type_t
# binary function type
ctypedef packed_type_t (*f_type)(packed_type_t, packed_type_t)
cdef f_type* function_list


cdef size_t PACKED_SIZE
cdef packed_type_t PACKED_ALL_SET
cdef packed_type_t PACKED_HIGH_BIT_SET


cpdef setbits(packed_type_t[:] vec, set positions)

cpdef pack_chunk(packed_type_t[:] mat, packed_type_t[:, :] packed, size_t col)
cpdef packmat(np.uint8_t[:, :] mat, bint transpose=*)

cpdef unpackmat(packed_type_t[:, :] packed_mat, size_t N, bint transpose=*)
cpdef unpackvec(packed_type_t[:] packed_vec, size_t N)

cpdef packed_type_t generate_end_mask(N)

cpdef partition_columns(packed_type_t[:, :] matrix, size_t N, indices)
cpdef sample_columns(packed_type_t[:, :] matrix, size_t N, indices, bint invert=*)

cpdef transpose(packed_type_t[:, :] matrix, size_t N)
