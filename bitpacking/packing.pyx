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


cpdef pack_chunk(packed_type_t[:] mat, packed_type_t[:, :] packed, size_t N, size_t column):
    ''' This method assumed mat.shape[0] == PACKED_SIZE.'''
    cdef:
        size_t i, bit
        packed_type_t chunk, mask
    # build packed matrix
    mask = 1
    for i in range(N):
        chunk = 0
        for bit in range(PACKED_SIZE):
            chunk |= (((mat[bit] & mask) >> i) << bit)
        mask <<= 1
        packed[i, column] = chunk


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


cpdef partition_packed(packed_type_t[:, :] matrix, size_t N, size_t[:] indices):
    cdef:
        packed_type_t mask, bit
        packed_type_t[:, :] M, M_trg, M_test
        size_t Nr, Nw, Nw1, Nw2, N1, r, w
        size_t b, b1, b2, w1, w2

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
                bit = (matrix[r, w] & mask) >> b
                # if this bit of this word is in the sample
                if b + w * PACKED_SIZE in indices:
                    # insert into training sample at the next position
                    M1[r, w1] += bit << b1
                    # increment training sample word and bit indices
                    b1 = (b1 + 1) % PACKED_SIZE
                    w1 += b1 == 0
                else:
                    # insert into test sample at the next position
                    M2[r, w2] += bit << b2
                    # increment test sample word and bit indices
                    b2 = (b2 + 1) % PACKED_SIZE
                    w2 += b2 == 0
                mask <<= 1
    return M1, M2


cpdef sample_packed(packed_type_t[:, :] matrix, size_t N, size_t[:] indices, invert=False):
    cdef packed_type_t mask
    cdef packed_type_t[:, :] sample
    cdef size_t Nr, Nw, Ns, r, w, b, sw, sb, cols, index

    Nr = matrix.shape[0]
    Ns = indices.shape[0]

    if invert:
        # sample matrix
        Nw = int(ceil((N - Ns) / <double>PACKED_SIZE))
        sample = np.zeros((Nr, Nw), dtype=packed_type)

        # word and bit positions for sample
        sb = sw = 0
        for w in range(Nw):
            # if this bit of this word is in the sample
            mask = 1
            for b in range(PACKED_SIZE):
                if w * PACKED_SIZE + b not in indices:
                    # get the bit
                    for r in range(Nr):
                        bit = (matrix[r, w] & mask) >> b
                        # insert into the sample at the next position
                        sample[r, sw] += bit << sb
                    # increment sample word and bit indices
                    sb = (sb + 1) % PACKED_SIZE
                    sw += sb == 0
                mask <<= 1
    else:
        Nw = int(ceil(Ns / <double>PACKED_SIZE))
        sample = np.zeros((Nr, Nw), dtype=packed_type)
        
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
                bit = (matrix[r, w] & mask) >> b
                # insert into into the sample at the next position
                sample[r, sw] += (bit << sb)
            # increment word and bit indices
            sb = (sb + 1) % PACKED_SIZE
            sw += sb == 0
    return sample


function_list = [__f0, __f1, __f2, __f3, __f4, __f5, __f6, __f7,
                 __f8, __f9, __f10, __f11, __f12, __f13, __f14, __f15,]

cdef packed_type_t __f0(packed_type_t x, packed_type_t y):  return 0
cdef packed_type_t __f1(packed_type_t x, packed_type_t y):  return ~(x|y)   # NOR
cdef packed_type_t __f2(packed_type_t x, packed_type_t y):  return ~x&y
cdef packed_type_t __f3(packed_type_t x, packed_type_t y):  return ~x
cdef packed_type_t __f4(packed_type_t x, packed_type_t y):  return x&~y
cdef packed_type_t __f5(packed_type_t x, packed_type_t y):  return ~y
cdef packed_type_t __f6(packed_type_t x, packed_type_t y):  return x^y      # XOR
cdef packed_type_t __f7(packed_type_t x, packed_type_t y):  return ~(x&y)   # NAND
cdef packed_type_t __f8(packed_type_t x, packed_type_t y):  return x&y      # AND
cdef packed_type_t __f9(packed_type_t x, packed_type_t y):  return ~(x^y)   # XNOR
cdef packed_type_t __f10(packed_type_t x, packed_type_t y): return y
cdef packed_type_t __f11(packed_type_t x, packed_type_t y): return ~x|y
cdef packed_type_t __f12(packed_type_t x, packed_type_t y): return x
cdef packed_type_t __f13(packed_type_t x, packed_type_t y): return x|~y
cdef packed_type_t __f14(packed_type_t x, packed_type_t y): return x|y      # OR
cdef packed_type_t __f15(packed_type_t x, packed_type_t y): return 1