import nest
from pathlib import Path
import os, sys
import json
import h5py
import time
import numpy as np
import pandas as pd
import pprint
import h5py as h5
import time
from collections import defaultdict
sonata_folder = sys.argv[1]
trials = int(sys.argv[2])
train_test = sys.argv[3]
gid_stim_file = sys.argv[4]
shuff = int(sys.argv[5])
###############################################################################

nest.set_verbosity("M_ALL")
nest.ResetKernel()
dt=.1
nest.SetKernelStatus({""
                        "local_num_threads": 16,
                        "data_path": 'outputs/',
                        "print_time" : False,
                        "overwrite_files":True,
                            "rng_seed": 12345,
                            "resolution": dt
                        })

print(nest.GetKernelStatus()["num_processes"])
print(nest.GetKernelStatus()["local_num_threads"])
###############################################################################
window_duration = 200
with open(sonata_folder+'/config.circuit.json', 'r') as file:
    config_f = json.load(file)
    if train_test=='train':
        config_f['run']['tstop']=trials*window_duration
    else:
        config_f['run']['tstop']=110000.0
with open(sonata_folder+'/config.circuit.json', 'w') as f:
    json.dump(config_f , f, indent=4)

base_path = Path(__file__).resolve().parent
sonata_path = base_path / f"../{sonata_folder}"

net_config = sonata_path / "config.circuit.json"

module_name='target/ht_tso_stdp_module'
nest.Install(module_name)

syn_folder = os.path.join(sonata_folder, "synapse_params_cus")
for filename in os.listdir(syn_folder):
    pre = filename.split("_to_")[0]

    if pre in ["SP_PC", "SO_PC"]:
        filepath = os.path.join(syn_folder, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
        if train_test=='train':
            data["use_stdp"] = 1.0
            data["Wmax"] = .1
            data["lambda"]=5e-04
        elif train_test=='test':
            data["use_stdp"] = 0.0

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

###############################################################################
print('Start config...')
t = time.time()

sonata_net = nest.SonataNetwork(net_config)
print('Done in ',int(time.time()-t), 's.')

print('Start build...')
t = time.time()
node_collections = sonata_net.BuildNetwork(hdf5_hyperslab_size=2**20)
print('Done in ',int(time.time()-t), 's.')

###############################################################################
# We can now verify whether the built network has the expected number of
# nodes and connections.

print(f'Network built from SONATA specifications in directory "{sonata_path.name}"')
print(f"  Number of nodes      : {nest.network_size}")
print(f"  Number of connections: {nest.num_connections}")

###############################################################################

## RECORDING ACTIVITY
s_rec = nest.Create("spike_recorder")
s_rec.record_to = 'ascii'
s_rec.label = sonata_folder+'_'+train_test+str(trials)
pop_name_full = "CA3_comm_Network"
nest.Connect(node_collections[pop_name_full], s_rec)

# # record from INPUT
i_rec = nest.Create("spike_recorder")
i_rec.record_to = 'ascii'
i_rec.label = sonata_folder+"_stim"+train_test+str(trials)
pop_name = "INPUT_Network"
record_node_ids = [0]
nest.Connect(node_collections[pop_name], i_rec)

## CREATE AUTOASSOCIATIVE STIM
bunch_nodes=pd.read_csv('data/'+gid_stim_file)
bunch_nodes=np.array(bunch_nodes).flatten()
bunch_nodes.sort()
print('loaded gid stims',flush=True)
print('Create associative stim ... ',flush=True)

if train_test == 'train':
    associative_stim = nest.Create('spike_generator', 1 )

    spike_times = []
    for i in range(trials):
        spike_times.extend([x +i*window_duration for x in [2.0,7.0,12.0,17.0]])

    nest.SetStatus(associative_stim, {"spike_times": np.array(spike_times)} )
    nest.Connect(associative_stim, node_collections[pop_name_full][bunch_nodes], syn_spec={'synapse_model':'static_synapse','receptor_type':1,'weight':3.})
elif train_test == 'test':
    n_gid_stim = len(bunch_nodes)
    rec_p=[0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.625, 0.75, 0.875, 0.875]
    rec=[int(i*n_gid_stim) for i in rec_p]
    times = np.arange(1000,5000*len(rec),5000)
    retr_nodes_all = {}
    if shuff:
        
        np.random.seed(shuff)
        for i, r in enumerate(rec):
            locals()["retrieval_stim"+str(i)] = nest.Create('spike_generator', 1 )
            ret_spk = np.array([2.0,7.0,12.0,17.0]) + times[i]
            nest.SetStatus(locals()["retrieval_stim"+str(i)], {"spike_times": ret_spk} )
            #retr_nodes = bunch_nodes[:r]
            retr_nodes = np.random.choice(bunch_nodes, r, replace=False)
            retr_nodes.sort()
            retr_nodes_all[str(r)] = retr_nodes
            nest.Connect(locals()["retrieval_stim"+str(i)], node_collections[pop_name_full][retr_nodes], syn_spec={'synapse_model':'static_synapse','receptor_type':1,'weight':3.})
        np.save('%s/shuffle_folder/id_sh_%s_%s.npy'%(sonata_folder,shuff,trials),retr_nodes_all)
    else:
        for i, r in enumerate(rec):
            locals()["retrieval_stim"+str(i)] = nest.Create('spike_generator', 1 )
            ret_spk = np.array([2.0,7.0,12.0,17.0]) + times[i]
            nest.SetStatus(locals()["retrieval_stim"+str(i)], {"spike_times": ret_spk} )
            retr_nodes = bunch_nodes[:r]
            retr_nodes_all[str(r)] = retr_nodes
            nest.Connect(locals()["retrieval_stim"+str(i)], node_collections[pop_name_full][retr_nodes], syn_spec={'synapse_model':'static_synapse','receptor_type':1,'weight':3.})
        np.save('%s/id_%s.npy'%(sonata_folder,trials),retr_nodes_all)


## CREATE BACKGROUND NOISE

nodes_csv_filename=sonata_folder+'/CA3_comm_Network_node_types.csv'
df=pd.read_csv(nodes_csv_filename, sep='\s+' )
pyr_node_ids=[]
for i,pop_name in enumerate(df['pop_name']):
    if pop_name.count('PC'):
        print(i, pop_name, df['node_type_id'][i])
        pyr_node_ids.append(df['node_type_id'][i])
print('pyr node ids of length ',len(pyr_node_ids),flush=True)

with h5.File(sonata_folder+'/CA3_comm_Network_nodes.h5') as f:
    group_ids=np.array(f['nodes/CA3_comm_Network/node_type_id'])
    mask=np.isin(group_ids,pyr_node_ids)
    pyr_gids=f['nodes/CA3_comm_Network/node_id'][mask].flatten()
    print('pyr gids of shape:', pyr_gids.shape,flush=True)

#adding a poisson generator


poisson_gen = nest.Create("poisson_generator") #background pyr noise
rates = 0.5
nest.SetStatus(poisson_gen, {"rate": rates})

nest.Connect(poisson_gen, node_collections[pop_name_full][pyr_gids],syn_spec={'synapse_model':'static_synapse','receptor_type':1,'weight':3.0})

Rate_i= {"SO_OLM": 0.5,
        "SO_BS": 14.0,
        "SO_TRI": 5.7,
        "SP_AA": 19.2,
        "SP_PVBC": 8.2,
        "SP_CCKBC": 0.5,
        "SP_BS": 14.0,
        "SP_IVY": 1.4,
        "SL_SL": 5.7,
        "SL_MFA": 1.0,
        "SR_R": 1.0,
        "SR_PP": 0.5,
        "SLM_NGF": 1.4
    }

nodes_csv_filename=sonata_folder+'/CA3_comm_Network_node_types.csv'
df=pd.read_csv(nodes_csv_filename, sep='\s+')
print(df.columns.tolist(),flush=True)
for i,pop_name in enumerate(df['pop_name']):
    if pop_name.count('PC')==0:
        node_type_id=df['node_type_id'][i]
        with h5.File(sonata_folder+'/CA3_comm_Network_nodes.h5') as f:
            group_ids=np.array(f['nodes/CA3_comm_Network/node_type_id'])
            mask=np.isin(group_ids,node_type_id)
            gids_inh=f['nodes/CA3_comm_Network/node_id'][mask].flatten()        
            poisson_gen = nest.Create("poisson_generator") #background pyr noise
            rates =Rate_i[pop_name]
            nest.SetStatus(poisson_gen, {"rate": rates})

            nest.Connect(poisson_gen, node_collections[pop_name_full][gids_inh],syn_spec={'synapse_model':'static_synapse','receptor_type':1,'weight':3.0})

if train_test == 'test':
    print('starting setting weights',flush=True)
    t_start=time.time()
    filename = sonata_folder+f'/connections_pc_{trials}_'+sonata_folder+'.h5'
    print(f' weight filename: {filename}',flush=True)
    print(f'pyr_gids {len(pyr_gids)}',flush=True)

    t_premap=time.time()
    targets_by_pre = defaultdict(list)
    weights_by_pre = defaultdict(list)

    wanted = set(pyr_gids.astype(int) + 1)

    chunk_size = 5_000_000

    with h5.File(filename, 'r') as f:
        pre_d  = f['pre']
        post_d = f['post']
        ww_d   = f['weight']

        N = len(pre_d)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            pre_chunk  = pre_d[start:end]
            post_chunk = post_d[start:end]
            ww_chunk   = ww_d[start:end]
            #ww_chunk = np.full(end - start, 0.001, dtype=ww_d.dtype)
            for p in np.unique(pre_chunk):
                if p in wanted:
                    m = (pre_chunk == p)
                    targets_by_pre[p].append(post_chunk[m])
                    weights_by_pre[p].append(ww_chunk[m])
    for p in targets_by_pre:
        targets_by_pre[p] = np.concatenate(targets_by_pre[p])
        weights_by_pre[p] = np.concatenate(weights_by_pre[p])

    print('premap in :', time.time()-t_premap)
    for idp in (pyr_gids.astype(int) + 1):
        conn = nest.GetConnections(node_collections[pop_name_full][idp-1])
        res = conn.get(custom=True)
        source=res['source']
        targ=res['target']
        saved_targ = targets_by_pre[idp]
        saved_w = weights_by_pre[idp]

        assert len(saved_targ) == len(targ), (
            f'idp {idp}: len(saved_targ)={len(saved_targ)}, '
            f'len(targ)={len(targ)}'
        )

        if not np.all(np.sort(saved_targ) == np.sort(targ)):
            missing = set(targ) - set(saved_targ)
            extra   = set(saved_targ) - set(targ)

            raise AssertionError(
                f'idp {idp}: '
                f'missing targets {missing}, '
                f'extra targets {extra}'
            )
        order_targ = np.argsort(targ)
        order_saved = np.argsort(saved_targ)
        inv_order_targ = np.empty_like(order_targ)
        inv_order_targ[order_targ] = np.arange(len(order_targ))
        conn.set({'weight':saved_w[order_saved][inv_order_targ]},custom=True)
    print('Done in: ',time.time()-t_start,flush=True)

## SIMULATE
print('starting_simulate, Simulation time: ', config_f['run']['tstop'], ' ms',flush=True)
t = time.time()
sonata_net.Simulate()

print('simulation Done',flush=True)
import os, gc
gc.collect()



## SAVE WEIGHTS
if train_test == 'train':
    print('querying pyr nodes',flush=True)
    t = time.time()
    pyr_nodes = node_collections[pop_name_full][pyr_gids]

    t = time.time()
    batch_size=100
    sources_all=[]
    targets_all=[]
    weights_all=[]

    n = len(pyr_nodes)
    total_conns = 0

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        conns = nest.GetConnections(source=pyr_nodes[i:j],custom=True)
        total_conns += len(conns['weight'])

    print("Total connections:", total_conns, flush=True)

    t=time.time()

    #filename = f"{sonata_folder}/connections_pc_{trials}_{sonata_folder}_test.h5"
    filename = f"{sonata_folder}/connections_pc_{trials}_{sonata_folder}.h5"
    with h5py.File(filename, "w") as f:
        pre_ds = f.create_dataset(
            "pre",
            shape=(total_conns,),
            dtype=np.int32,
            compression=None,
            chunks=(1_000_000,),  # IMPORTANT
        )
        post_ds = f.create_dataset(
            "post",
            shape=(total_conns,),
            dtype=np.int32,
            compression=None,
            chunks=(1_000_000,),
        )
        w_ds = f.create_dataset(
            "weight",
            shape=(total_conns,),
            dtype=np.float32,
            compression=None,
            chunks=(1_000_000,),
        )

        offset = 0
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            conn=nest.GetConnections(source=pyr_nodes[i:j])
            res=conn.get(custom=True)
            #res= nest.GetConnections(source=pyr_nodes[i:j],custom=True)
            # res = conns.get(custom=True)

            m = len(res["source"])
            pre_ds[offset:offset+m] = res["source"]
            post_ds[offset:offset+m] = res["target"]
            w_ds[offset:offset+m] = res["weight"]

            offset += m



    print('Done in ',int(time.time()-t), 's.',flush=True)
print('Sonata_simulate_all Done',flush=True)

