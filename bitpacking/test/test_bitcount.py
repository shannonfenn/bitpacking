import bitpacking.bitcount as bc
import numpy as np
from pytest import fixture


@fixture(params=[
    (0, 0, 0),
    (4370323815288199633, 64, 33), (1129843337900893199, 64, 30),
    (649369928250959927, 64, 23), (588858933541175576, 61, 17),
    (1971513181041067719, 64, 32), (3555176812501657937, 64, 33),
    (4558881714062980293, 64, 36), (7796458688392721668, 62, 33),
    (5717748707411915565, 64, 33), (9086724868667118591, 64, 35)] + [
    (2**i, 64 - i, 1) for i in range(64)])
def chunk_instance(request):
    return request.param


@fixture(params=list(range(5)))
def chunk_mask_instance(request, chunk_instance):
    mask_len = np.random.randint(1, 64)
    return [chunk_instance[0], mask_len, max(0, chunk_instance[1] - mask_len)]


@fixture(params=[
    ([0, 0], 0, 0), ([0, 0, 0], 0, 0), ([0, 0, 0, 0, 0], 0, 0),
    ([4370323815288199633, 1129843337900893199], 128, 63),
    ([588858933541175576, 649369928250959927], 125, 40),
    ([7796458688392721668, 1971513181041067719, 3555176812501657937], 190, 98),
    ([4558881714062980293, 9086724868667118591, 5717748707411915565], 192, 104)] + [
    ([2**i, 2**j], 128 - i, 2) for i in range(0, 64, 16) for j in range(0, 64, 16)] + [
    ([2**i, 2**j, 2**j], 192 - i, 3) for i in range(0, 64, 16) for j in range(0, 64, 16)])
def vector_instance(request):
    return request.param


@fixture(params=list(range(5)))
def vector_mask_instance(request, vector_instance):
    mask_len = np.random.randint(1, 64*len(vector_instance[0]))
    return [vector_instance[0], mask_len, max(0, vector_instance[1] - mask_len)]


def test_floodcount_chunk(chunk_instance):
    expected = chunk_instance[1]
    actual = bc.floodcount_chunk(chunk_instance[0])
    assert expected == actual


def test_floodcount_vector(vector_instance):
    expected = vector_instance[1]
    actual = bc.floodcount_vector(np.array(vector_instance[0], dtype=np.uint64))
    assert expected == actual


def test_floodcount_chunk_with_mask(chunk_mask_instance):
    expected = chunk_mask_instance[2]
    actual = bc.floodcount_chunk(chunk_mask_instance[0], chunk_mask_instance[1])
    assert expected == actual


def test_floodcount_vector_with_mask(vector_mask_instance):
    expected = vector_mask_instance[2]
    actual = bc.floodcount_vector(
        np.array(vector_mask_instance[0], dtype=np.uint64),
        vector_mask_instance[1])
    assert expected == actual


def test_popcount_chunk(chunk_instance):
    expected = chunk_instance[2]
    actual = bc.popcount_chunk(chunk_instance[0])
    assert expected == actual


def test_popcount_vector(vector_instance):
    expected = vector_instance[2]
    actual = bc.popcount_vector(np.array(vector_instance[0], dtype=np.uint64))
    assert expected == actual
