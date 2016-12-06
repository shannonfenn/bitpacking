# cython: language_level=3
# distutils: libraries = gmp

# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True

from bitpacking.packing cimport PACKED_HIGH_BIT_SET, PACKED_SIZE

cpdef size_t popcount_matrix(packed_type_t[:, :] mat):
    return mpn_popcount(&mat[0, 0], mat.shape[0]*mat.shape[1])


cpdef size_t popcount_vector(packed_type_t[:] vec):
    return mpn_popcount(&vec[0], vec.shape[0])


cpdef size_t popcount_chunk(packed_type_t chunk):
    return mpn_popcount(&chunk, 1)


cpdef void popcount_matrix_rows(packed_type_t[:, :] mat, size_t[:] row_counts):
    cdef size_t i, cols
    cols = mat.shape[1]
    for i in range(row_counts.shape[0]):
        row_counts[i] = mpn_popcount(&mat[i, 0], cols)


cpdef size_t scan_vector(packed_type_t[:] vec):
    cdef bint was_zero
    cdef size_t pos, row_len = PACKED_SIZE * vec.shape[0]

    # to ensure there is at least a 1 at the end for mpn_scan1 we need to add it, but also
    # remember if the last bit was set so we can unset it, and also return the correct count
    was_zero = ((vec[-1] & PACKED_HIGH_BIT_SET) == 0)
    vec[-1] |= PACKED_HIGH_BIT_SET
    pos = mpn_scan1(&vec[0], 0)
    if was_zero:
        vec[-1] &= ~PACKED_HIGH_BIT_SET
        if pos == row_len - 1:
            return row_len
    return pos


cpdef size_t floodcount_vector(packed_type_t[:] vec, size_t end_mask_len=0):
    cdef bint was_zero
    cdef size_t pos, row_len = PACKED_SIZE * vec.shape[0]

    # to ensure there is at least a 1 at the end for mpn_scan1 we need to add it, but also
    # remember if the last bit was set so we can unset it, and also return the correct count
    was_zero = ((vec[-1] & PACKED_HIGH_BIT_SET) == 0)
    vec[-1] |= PACKED_HIGH_BIT_SET
    pos = mpn_scan1(&vec[0], 0)
    if was_zero:
        vec[-1] &= ~PACKED_HIGH_BIT_SET
        if pos == row_len - 1:
            return 0
    pos = row_len - pos
    if pos <= end_mask_len:
        return 0
    return pos - end_mask_len


cpdef size_t floodcount_chunk(packed_type_t chunk, size_t end_mask_len=0):
    cdef bint was_zero
    cdef size_t pos

    # to ensure there is at least a 1 at the end for mpn_scan1 we need to add it, but also
    # remember if the last bit was set so we can unset it, and also return the correct count
    was_zero = ((chunk & PACKED_HIGH_BIT_SET) == 0)
    chunk |= PACKED_HIGH_BIT_SET
    pos = mpn_scan1(&chunk, 0)
    if was_zero:
        chunk &= ~PACKED_HIGH_BIT_SET
        if pos == PACKED_SIZE - 1:
            return 0
    pos = PACKED_SIZE - pos
    if pos <= end_mask_len:
        return 0
    return pos - end_mask_len
