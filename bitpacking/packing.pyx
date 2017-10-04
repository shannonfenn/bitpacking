# cython: language_level=3, profile=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
from math import ceil
from libc.limits cimport ULLONG_MAX


PACKED_SIZE = 64
PACKED_SIZE_PY = PACKED_SIZE
PACKED_ALL_SET = ULLONG_MAX
PACKED_HIGH_BIT_SET = 0x8000000000000000
packed_type = np.uint64


cpdef setbits(packed_type_t[:] vec, set positions):
    cdef:
        size_t pos, bit, chunk
    for pos in positions:
        chunk = pos // PACKED_SIZE
        bit = pos % PACKED_SIZE
        vec[chunk] |= (<packed_type_t>1) << bit


cpdef pack_chunk(packed_type_t[:] mat, packed_type_t[:, :] packed, size_t col):
    ''' Assumes mat.shape[0] == PACKED_SIZE.'''
    cdef:
        size_t row, bit
        packed_type_t chunk, mask
    # build packed matrix
    mask = 1
    for row in range(packed.shape[0]):
        chunk = 0
        for bit in range(PACKED_SIZE):
            chunk |= (((mat[bit] & mask) >> row) << bit)
        mask <<= 1
        packed[row, col] = chunk


cpdef packmat(np.uint8_t[:, :] mat, bint transpose=True):
    cdef:
        size_t Nr, Nc, num_chunks, i, ch, bit
        packed_type_t chunk
        packed_type_t[:, :] packed
        np.uint8_t[:, :] padded

    Nr, Nc = mat.shape[0], mat.shape[1]
    if transpose:
        num_chunks = int(ceil(Nr / <double>PACKED_SIZE))
        # pad rows with zeros to next multiple of PACKED_SIZE
        padded = np.zeros((num_chunks * PACKED_SIZE, Nc), dtype=np.uint8)
        padded[:Nr, :] = mat
        # build packed matrix
        packed = np.empty((Nc, num_chunks), dtype=packed_type)
        for i in range(Nc):
            for ch in range(num_chunks):
                chunk = 0
                for bit in range(PACKED_SIZE):
                    chunk |= (<packed_type_t>(padded[ch*PACKED_SIZE+bit, i]) << bit)
                packed[i, ch] = chunk
    else:
        num_chunks = int(ceil(Nc / <double>PACKED_SIZE))
        # pad rows with zeros to next multiple of PACKED_SIZE
        padded = np.zeros((Nr, num_chunks * PACKED_SIZE), dtype=np.uint8)
        padded[:, :Nc] = mat
        # build packed matrix
        packed = np.empty((Nr, num_chunks), dtype=packed_type)
        for i in range(Nr):
            for ch in range(num_chunks):
                chunk = 0
                for bit in range(PACKED_SIZE):
                    chunk |= (<packed_type_t>(padded[i, ch*PACKED_SIZE+bit]) << bit)
                packed[i, ch] = chunk
    return np.asarray(packed)


cpdef unpackmat(packed_type_t[:, :] packed_mat, size_t N, bint transpose=True):
    cdef:
        size_t Nr, Nc, num_chunks, i, ch, bit, r, c
        packed_type_t mask, chunk
        np.uint8_t[:, :] unpacked

    num_chunks = packed_mat.shape[1]
    if transpose:
        Nc = packed_mat.shape[0]
        Nr = N
        unpacked = np.zeros((num_chunks*PACKED_SIZE, Nc), dtype=np.uint8)
        for i in range(Nc):
            r = 0
            for ch in range(num_chunks):
                mask = 1
                chunk = packed_mat[i, ch]
                for bit in range(PACKED_SIZE):
                    unpacked[r, i] += ((chunk & mask) >> bit)
                    mask <<= 1
                    r += 1
        return np.asarray(unpacked[:Nr, :])
    else:
        Nr = packed_mat.shape[0]
        Nc = N
        unpacked = np.zeros((Nr, num_chunks*PACKED_SIZE), dtype=np.uint8)
        for i in range(Nr):
            c = 0
            for ch in range(num_chunks):
                mask = 1
                chunk = packed_mat[i, ch]
                for bit in range(PACKED_SIZE):
                    unpacked[i, c] += ((chunk & mask) >> bit)
                    mask <<= 1
                    c += 1
        return np.asarray(unpacked[:, :Nc])


cpdef unpackvec(packed_type_t[:] packed_vec, size_t N):
    cdef:
        size_t num_chunks, c, bit, example
        packed_type_t mask, chunk
        np.uint8_t[:] unpacked

    num_chunks = packed_vec.shape[0]
    unpacked = np.zeros(num_chunks*PACKED_SIZE, dtype=np.uint8)
    for c in range(num_chunks):
        mask = 1
        example = c * PACKED_SIZE
        chunk = packed_vec[c]
        for bit in range(PACKED_SIZE):
            unpacked[example] += ((chunk & mask) >> bit)
            mask <<= 1
            example += 1
    return np.asarray(unpacked[:N])


cpdef packed_type_t generate_end_mask(N):
    cdef:
        packed_type_t end_mask, shift
        size_t bits, b, bits_to_remain

    end_mask = PACKED_ALL_SET
    bits_to_remain = N % PACKED_SIZE

    if bits_to_remain != 0:
        # shift = 1
        # for b in range(PACKED_SIZE-bits_to_remain):
        #     end_mask -= shift
        #     shift <<= 1
        shift = PACKED_HIGH_BIT_SET
        for b in range(PACKED_SIZE-bits_to_remain):
            end_mask &= ~shift
            shift >>= 1
    return end_mask


cpdef partition_columns(packed_type_t[:, :] matrix, size_t N, indices):
    cdef:
        packed_type_t mask, bit_value
        packed_type_t[:, :] M, M1, M2
        size_t Nr, Nw, Nw1, Nw2, N1, r, w, b, b1, b2, w1, w2

    N1 = indices.shape[0]
    Nr = matrix.shape[0]
    Nw = matrix.shape[1]

    # find number of words for each partition
    Nw1 = int(np.ceil(N1 / <double>PACKED_SIZE))
    Nw2 = int(np.ceil((N - N1) / <double>PACKED_SIZE))

    M1 = np.zeros((Nr, Nw1), dtype=packed_type)
    M2 = np.zeros((Nr, Nw2), dtype=packed_type)

    for r in range(Nr):
        # word and bit positions for training and test samples
        b1 = b2 = 0
        w1 = w2 = 0
        for w in range(Nw):
            mask = 1
            for b in range(PACKED_SIZE):
                # get the bit from the original matrix
                bit_value = (matrix[r, w] & mask) >> b
                # if this bit of this word is in the sample
                if b + w * PACKED_SIZE in indices:
                    # insert into training sample at the next position
                    M1[r, w1] += bit_value << b1
                    # increment training sample word and bit indices
                    b1 = (b1 + 1) % PACKED_SIZE
                    w1 += b1 == 0
                else:
                    # insert into test sample at the next position
                    M2[r, w2] += bit_value << b2
                    # increment test sample word and bit indices
                    b2 = (b2 + 1) % PACKED_SIZE
                    w2 += b2 == 0
                mask <<= 1
    return M1, M2


cpdef sample_columns(packed_type_t[:, :] matrix, size_t N, indices, bint invert=False):
    cdef packed_type_t mask, bit_value
    cdef packed_type_t[:, :] sample
    cdef size_t Nr, Nw, Nws, Nsamp, r, w, b, sw, sb, cols, index

    Nr = matrix.shape[0]
    Nsamp = indices.shape[0]

    if invert:
        # sample matrix
        Nw = matrix.shape[1]
        Nws = int(ceil((N - Nsamp) / <double>PACKED_SIZE))
        sample = np.zeros((Nr, Nws), dtype=packed_type)

        # word and bit positions for sample
        sb = sw = 0
        for w in range(Nw):
            # if this bit of this word is in the sample
            mask = 1
            for b in range(PACKED_SIZE):
                if w * PACKED_SIZE + b not in indices:
                    # get the bit
                    for r in range(Nr):
                        bit_value = (matrix[r, w] & mask) >> b
                        # insert into the sample at the next position
                        sample[r, sw] += bit_value << sb
                    # increment sample word and bit indices
                    sb = (sb + 1) % PACKED_SIZE
                    sw += sb == 0
                mask <<= 1
    else:
        Nws = int(ceil(Nsamp / <double>PACKED_SIZE))
        sample = np.zeros((Nr, Nws), dtype=packed_type)

        # word and bit positions for sample
        sw = sb = 0
        # for each index in sample
        for index in indices:
            # word and bit indices into original matrix
            w = index // PACKED_SIZE
            b = index % PACKED_SIZE
            mask = <packed_type_t>1 << b
            for r in range(Nr):
                # get the bit
                bit_value = (matrix[r, w] & mask) >> b
                # insert into into the sample at the next position
                sample[r, sw] += (bit_value << sb)
            # increment word and bit indices
            sb = (sb + 1) % PACKED_SIZE
            sw += sb == 0
    return sample


cpdef transpose(packed_type_t[:, :] matrix, size_t N):
    cdef:
        size_t Nr1, Nc1, Nr2, Nc2
        size_t r1, c1, r2, c2, bit1, bit2
        packed_type_t mask
        packed_type_t[:, :] transposed

    Nr1, Nc1 = matrix.shape[0], matrix.shape[1]
    Nr2 = N
    Nc2 = int(ceil(Nr1 / <double>PACKED_SIZE))

    transposed = np.zeros((Nr2, Nc2), dtype=packed_type)

    # build packed matrix
    for r2 in range(Nr2):
        c1 = r2 // PACKED_SIZE
        bit1 = r2 % PACKED_SIZE
        mask = (<packed_type_t>1) << bit1
        # looping only over existing rows
        for r1 in range(Nr1):
            c2 = r1 // PACKED_SIZE
            bit2 = r1 % PACKED_SIZE

            transposed[r2, c2] |= ((matrix[r1, c1] & mask) >> bit1 << bit2)

    return np.asarray(transposed)
