import numpy as np
from glob import glob
import os, sys
import h5py
import time
import pandas as pd
import json

results_data_dir = sys.argv[1]
placement_file = 'data/'+sys.argv[2]
output_data_dir = 'outputs'
trials=sys.argv[3]
train_test=sys.argv[4]
gid_stim_file=sys.argv[5]
shuff=sys.argv[6]
print('starting output processing',flush=True)
t_start=time.time()

def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count




# CREATE NETWORK FILE
with open('%s/network.csv'%results_data_dir,'w') as f_out:
    with h5py.File(placement_file,'r') as positions:
        # Sort populations based on GID order
        population_keys = [pop for pop,v in positions.items() if len(v)>0]
        population_init_gid = np.array([v[0,0]    for pop,v in positions.items() if len(v)>0])
        population_sort = np.argsort(population_init_gid)
        for v_node_type_id,(pop_id) in enumerate(population_sort):
            pop = population_keys[pop_id]
            pyr_positions = positions[pop]
            # Write CSV network file
            np.savetxt(f_out, np.array(pyr_positions,dtype=int), delimiter=',', fmt=['%i', '%i', '%i', '%i', ])
print('created network file in :',time.time()-t_start,flush=True)

# CREATE ACTIVITY FILE
pop = 'CA3_comm_Network'
print('shuff',shuff)
if int(shuff)>0:

    activity_file = f'shuffle_folder/activity_{shuff}_'
else:
    activity_file = f'activity_'
with open('%s/%s%s%s.csv'%(results_data_dir,activity_file,train_test,trials),'w') as act:
    for filename in os.listdir(output_data_dir):
        if filename.endswith(".dat") and ('_'+train_test+str(trials)) in filename and results_data_dir+'_' in filename:
            fn = os.path.join(output_data_dir, filename)
            if buf_count_newlines_gen(fn) > 3:
                spks = pd.read_csv(fn,skiprows=2,delimiter='\t').to_numpy()
                print(fn, spks.shape)
                if spks.any():
                    spks = spks.reshape((-1,2))
                    if spks.shape[0] > 1:
                        spks[:,0] = spks[:,0] - 1
                    else:
                        spks[0] = spks[0] - 1
                    np.savetxt(act, spks, delimiter=',', fmt=['%i','%.2f'])

print(f'created activity file {activity_file} in :',time.time()-t_start,flush=True)
print('Saved')
