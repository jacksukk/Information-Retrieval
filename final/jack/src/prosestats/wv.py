"""
Load word vectors.
"""

from collections import OrderedDict
import numpy as np

def load_word_vector_mapping(fstream):
    """
    Load word vector mapping using @fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    """
    ret = OrderedDict()
    for line in fstream:
        word, *vector = line.split()
        ret[word] = np.array(list(map(float, vector)))

    return ret

def test_load_word_vector_mapping():
    vectors = """UUUNKKK 0.172414 -0.091063 0.255125 -0.837163 0.434872 -0.499848 -0.042904 -0.059642 -0.635087 -0.458795 -0.105671 0.506513 -0.105105 -0.405678 0.493365 0.408807 0.401635 -0.817805 0.626340 0.580636 -0.246996 -0.008515 -0.671140 0.301865 -0.439651 0.247694 -0.291402 0.873009 0.216212 0.145576 -0.211101 -0.352360 0.227651 -0.118416 0.371816 0.261296 0.017548 0.596692 -0.485722 -0.369530 -0.048807 0.017960 -0.040483 0.111193 0.398039 0.162765 0.408946 0.005343 -0.107523 -0.079821
the -0.454847 1.002773 -1.406829 -0.016482 0.459856 -0.224457 0.093396 -0.826833 -0.530674 1.211044 -0.165133 0.174454 -1.130952 -0.612020 -0.024578 -0.168508 0.320113 0.774229 -0.360418 1.483124 -0.230922 0.301055 -0.119924 0.601642 0.694616 -0.304431 -0.414284 0.667385 0.171208 -0.334842 -0.459286 -0.534202 0.533660 -0.379468 -0.378721 -0.240499 -0.446272 0.686113 0.662359 -0.865312 0.861331 -0.627698 -0.569544 -1.228366 -0.152052 1.589123 0.081337 0.182695 -0.593022 0.438300
, -0.408797 -0.109333 -0.099279 -0.857098 -0.150319 -0.456398 -0.781524 -0.059621 0.302548 0.202162 -0.319892 -0.502241 -0.014925 0.020889 1.506245 0.247530 0.385598 -0.170776 0.325960 0.267304 0.157673 0.125540 -0.971452 -0.485595 0.487857 0.284369 -0.062811 -1.334082 0.744133 0.572701 1.009871 -0.457229 0.938059 0.654805 -0.430244 -0.697683 -0.220146 0.346002 -0.388637 -0.149513 0.011248 0.818728 0.042615 -0.594237 -0.646138 0.568898 0.700328 0.290316 0.293722 0.828779
. -0.583585 0.413481 -0.708189 0.168942 0.238435 0.789011 -0.566401 0.177570 -0.244441 0.328214 -0.319583 -0.468558 0.520323 0.072727 1.792047 -0.781348 -0.636644 0.070102 -0.247090 0.110990 0.182112 1.609935 -1.081378 0.922773 -0.605783 0.793724 0.476911 -1.279422 0.904010 -0.519837 1.235220 -0.149456 0.138923 0.686835 -0.733707 -0.335434 -1.865440 -0.476014 -0.140478 -0.148011 0.555169 1.356662 0.850737 -0.484898 0.341224 -0.056477 0.024663 1.141509 0.742001 0.478773
of -0.811262 -1.017245 0.311680 -0.437684 0.338728 1.034527 -0.415528 -0.646984 -0.121626 0.589435 -0.977225 0.099942 -1.296171 0.022671 0.946574 0.204963 0.297055 -0.394868 0.028115 -0.021189 -0.448692 0.421286 0.156809 -0.332004 0.177866 0.074233 0.299713 0.148349 1.104055 -0.172720 0.292706 0.727035 0.847151 0.024006 -0.826570 -1.038778 -0.568059 -0.460914 -1.290872 -0.294531 0.663751 -0.646503 0.499024 -0.804777 -0.402926 -0.292201 0.348031 0.215414 0.043492 0.165281
and -0.156019 0.405009 -0.370058 -1.417499 0.120639 -0.191854 -0.251213 -0.883898 -0.025010 0.150738 1.038723 0.038419 0.036411 -0.289871 0.588898 0.618994 0.087019 -0.275657 -0.105293 -0.536067 -0.181410 0.058034 0.552306 -0.389803 -0.384800 -0.470717 0.800593 -0.166609 0.702104 0.876092 0.353401 -0.314156 0.618290 0.804017 -0.925911 -1.002050 -0.231087 0.590011 -0.636952 -0.474758 0.169423 1.293482 0.609088 -0.956202 -0.013831 0.399147 0.436669 0.116759 -0.501962 1.308268
in -0.008573 -0.731185 -1.108792 -0.358545 0.507277 -0.050167 0.751870 0.217678 -0.646852 -0.947062 -1.187739 0.490993 -1.500471 0.463113 1.370237 0.218072 0.213489 -0.362163 -0.758691 -0.670870 0.218470 1.641174 0.293220 0.254524 0.085781 0.464454 0.196361 -0.693989 -0.384305 -0.171888 0.045602 1.476064 0.478454 0.726961 -0.642484 -0.266562 -0.846778 0.125562 -0.787331 -0.438503 0.954193 -0.859042 -0.180915 -0.944969 -0.447460 0.036127 0.654763 0.439739 -0.038052 0.991638""".split("\n")

    wvs = load_word_vector_mapping(vectors)
    assert "UUUNKKK" in wvs
    assert np.allclose(wvs["UUUNKKK"], np.array([0.172414, -0.091063, 0.255125, -0.837163, 0.434872, -0.499848, -0.042904, -0.059642, -0.635087, -0.458795, -0.105671, 0.506513, -0.105105, -0.405678, 0.493365, 0.408807, 0.401635, -0.817805, 0.626340, 0.580636, -0.246996, -0.008515, -0.671140, 0.301865, -0.439651, 0.247694, -0.291402, 0.873009, 0.216212, 0.145576, -0.211101, -0.352360, 0.227651, -0.118416, 0.371816, 0.261296, 0.017548, 0.596692, -0.485722, -0.369530, -0.048807, 0.017960, -0.040483, 0.111193, 0.398039, 0.162765, 0.408946, 0.005343, -0.107523, -0.079821]))
    assert "the" in wvs
    assert "of" in wvs
    assert "and" in wvs
