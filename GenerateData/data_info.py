import numpy as np

# Information of YUV files to generate the database for CU partition of HEVC (CPH)

YUV_NAME_LIST_FULL = [
    'RaceHorses_832x480_30'
]

YUV_WIDTH_LIST_FULL = np.array([832])
YUV_HEIGHT_LIST_FULL = np.array([480])

assert (len(YUV_NAME_LIST_FULL) == len(YUV_WIDTH_LIST_FULL))
assert (len(YUV_NAME_LIST_FULL) == len(YUV_HEIGHT_LIST_FULL))
