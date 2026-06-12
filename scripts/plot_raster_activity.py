import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import sys

def plot_raster_test(sonata_folder,placement_file,trials,train_test,gid_stim_file):
    rec_p=[0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.625, 0.75, 0.875, 0.875]

    
    activity_path = f"{sonata_folder}/activity_{train_test}{trials}.csv"
    act_df = pd.read_csv(activity_path, header=None, names=['id', 'spike_time'])

    gid_stim_file = f"data/{gid_stim_file}"

    # Network size
    size_net = 0
    with h5py.File(placement_file, 'r') as f:
        for i in f.keys():
            size_net += len(f[i])

    # Stimulated GIDs
    gid_stim = sorted(pd.read_csv(gid_stim_file)['gid'].tolist())
    print('gid stim of shape: ', len(gid_stim), flush=True)
    rec=[int(i*len(gid_stim)) for i in rec_p]
    times = np.arange(1000,5000*len(rec),5000)

    # Map from gid spike_times array
    act_gid = {
        gid: group['spike_time'].values
        for gid, group in act_df.groupby('id')
    }

    # -----------------------------
    # PC GIDs
    # -----------------------------
    pc_gid = []
    with h5py.File(placement_file, 'r') as f:
        for k in ['SP_PC', 'SO_PC']:
            g0 = int(f[k][0, 0])
            g1 = int(f[k][-1, 0]) + 1
            pc_gid.extend(range(g0, g1))

    pc_gid = sorted(pc_gid)
    pc_gid_nostim = [g for g in pc_gid if g not in gid_stim]

    # -----------------------------
    # Final ordered GID list
    # -----------------------------
    all_gids = list(range(size_net))
    remaining_gids = [g for g in all_gids if g not in gid_stim and g not in pc_gid_nostim]

    final_gid_list = gid_stim + pc_gid_nostim + remaining_gids

    if train_test=='test':
        retr_gid_file = f"{sonata_folder}/id_{trials}.npy"
        retr_gid = np.load(retr_gid_file, allow_pickle=True).item()
        print('retr_gid: ',retr_gid,flush=True)
    else:
        retr_gid ={}
        for r in rec:
            retr_gid[str(r)] = gid_stim
    print('retr_gid',retr_gid,flush=True)
 
    # -----------------------------
    # Plotting
    # -----------------------------
    fig, ax = plt.subplots(figsize=(20, 16))
    msize=5
    for row_i, gid in enumerate(final_gid_list):
        if gid not in act_gid:
            continue  # no spikes: skip

        spikes = act_gid[gid]

        # For stimulated GIDs, spikes must be split into sections
        if gid in gid_stim:
            for sec in range(len(rec) - 1):
                t0, t1 = times[sec], times[sec + 1]
                mask = (spikes >= t0) & (spikes < t1)
                spk = spikes[mask]
                if len(spk) == 0:
                    continue

                # classification for this section
                stim_now = set(retr_gid[str(rec[sec])])
                if gid in stim_now:
                    color = 'r'
                else:
                    color = 'b'

                ax.plot(spk, [row_i] * len(spk), '|', color=color,markersize=msize)

        # Non-stimulated PCs
        elif gid in pc_gid_nostim:
            ax.plot(spikes, [row_i] * len(spikes), '|k',markersize=msize)

        # All remaining neurons
        else:
            ax.plot(spikes, [row_i] * len(spikes), '|g',markersize=msize)

    ax.set_xlabel("Spike Time")
    ax.set_ylim([-2, len(gid_stim)+10])
    plt.savefig('%s/raster_%s_%s.png'%(sonata_folder,train_test,trials))
    
    ax.set_xlim(30900, 31100)
    plt.savefig(f'{sonata_folder}/raster_{train_test}_{trials}_cropped_31k.png')   
    
    ax.set_xlim(45900, 46100)
    plt.savefig(f'{sonata_folder}/raster_{train_test}_{trials}_cropped_46k.png')
    return


sonata_folder = sys.argv[1]   
placement_file = 'data/'+sys.argv[2]
trials = int(sys.argv[3])          
train_test = sys.argv[4]      #train/test
gid_stim_file = sys.argv[5] 
shuff = int(sys.argv[6])

plot_raster_test(sonata_folder,placement_file,trials,train_test,gid_stim_file)
