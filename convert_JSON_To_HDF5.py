import h5py
import numpy as np
import json

# Convert JSON to HDF5
with open('train.json', 'r') as f:
    data = json.load(f)

with h5py.File('train.h5', 'w') as hf:
    for i, entry in enumerate(data):
        group = hf.create_group(f'entry_{i}')
        group.create_dataset('band_1', data=np.array(entry['band_1']).reshape(75, 75))
        group.create_dataset('band_2', data=np.array(entry['band_2']).reshape(75, 75))
        group.attrs['id'] = entry['id']
        group.attrs['inc_angle'] = entry.get('inc_angle', 'na')
        group.attrs['is_iceberg'] = entry.get('is_iceberg', -1)
