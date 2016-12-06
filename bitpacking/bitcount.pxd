# cython: language_level=3
from bitpacking.packing cimport packed_type_t

cdef extern from "gmp.h":
    # unsigned long int mpn_popcount (mp_limb_t *s1p, mp_size_t n)
    unsigned long int mpn_popcount (packed_type_t *s1p, size_t n)
    unsigned long int mpn_scan1 (packed_type_t *s1p, size_t bit)

    #void mpn_nand_n(mp_ptr rp, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n)
    #void mpn_xor_n(mp_ptr rp, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n)

cpdef size_t popcount_matrix(packed_type_t[:, :] mat)
cpdef size_t popcount_vector(packed_type_t[:] vec)
cpdef size_t popcount_chunk(packed_type_t chunk)
cpdef void popcount_matrix_rows(packed_type_t[:, :] mat, size_t[:] row_counts)

cpdef size_t scan_vector(packed_type_t[:] vec)

cpdef size_t floodcount_vector(packed_type_t[:] vec, size_t end_mask_len=*)
cpdef size_t floodcount_chunk(packed_type_t chunk, size_t end_mask_len=*)