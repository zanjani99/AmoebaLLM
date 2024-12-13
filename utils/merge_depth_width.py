import sys
import os
import numpy as np
import json

s1_path = 'output/calib_dp/final_strategy.npy'
s2_path = 'output/width_calib/width_mask.npy'
s3_path = 'output/width_calib/width_bias.npy'

s1 = np.load(s1_path, allow_pickle=True).item()
s2 = np.load(s2_path, allow_pickle=True).item()
s3 = np.load(s3_path, allow_pickle=True).item()

s1['width_mask'] = s2
s1['bias'] = s3

np.save(f'dp_selection_strategy.npy', s1)
