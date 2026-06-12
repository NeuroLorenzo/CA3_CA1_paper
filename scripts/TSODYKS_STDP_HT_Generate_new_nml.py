
import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import numpy as np
import os
import re

from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils
from pynestml.frontend.pynestml_frontend import generate_target
from synapse_params import *
   
# n_neurons_in=1000
n_neurons_in=1000
# sim_time=10000
sim_time=1000
dt=.01

## $$ ONCE GENERATED, SET as FALSE TO AVOID MODELS REBUILDING
generate= True
# generate= False



if generate:
	module_name, neuron_model_name,synapse_model_name = \
		NESTCodeGeneratorUtils.generate_code_for(
												 'custom_models/ht_neuron_custom_gspike_fix_new_nml.nestml',
												 'custom_models/stdp_tsodyks_sym_new_nml_synapse.nestml',
												 post_ports=["post_spikes"],
												 logging_level="DEBUG",
												 codegen_opts={"delay_variable": {"stdp_tsodyks_Tartarini_synapse": "d"},
															  "weight_variable": {"stdp_tsodyks_Tartarini_synapse": "w"}},


												 module_name='ht_tso_stdp_module')

	print('-------------------------------------------------------------')
	print(module_name, neuron_model_name,synapse_model_name)




else:
	# module_name='target/ht_tso_module' 	
	module_name='target/ht_tso_stdp_module' 
	neuron_model_name="hill_tononi_neuron_nestml__with_stdp_tsodyks_Tartarini_synapse_nestml"
	synapse_model_name="stdp_tsodyks_Tartarini_synapse_nestml__with_hill_tononi_neuron_nestml"
      
    

# module_name='target/test2mlmodule'/
print('installed',nest.Install(module_name))


'''

nest.SetKernelStatus({"resolution": dt})  # Set dt to 0.1 ms
# # Define parameters for the ht_neuron
neuron_params_cus = {
	"V_m": -60.5,
    "tau_m": 5.0,
    "t_ref": 2.0,
    "tau_spike": 1.5,
    "theta": -50.0,
    "tau_rise_AMPA": 0.2,
    "tau_rise_GABA_A": 1.0,
    "tau_rise_GABA_B": 30.0,
    "tau_rise_NMDA": 9.0,
    "tau_decay_AMPA": 3.0,
    "tau_decay_NMDA": 148.5,
    "tau_decay_GABA_A": 10.0,
    "tau_decay_GABA_B": 250.0,
    "g_peak_AMPA": 0.6,
    "g_peak_NMDA": 0.7,
    "g_peak_GABA_A": 1.0,
    "g_peak_h": 0.0,
    "g_peak_NaP": 0.0,
    "E_rev_GABA_A": -90.0,
    "g_peak_T": 0.5,
    "g_peak_KNa": 3.0,
	"is_refractory": True
}

neuron_params_nest ={
    "V_m": -60.5,
    "tau_m": 5.0,
    "t_ref": 2.0,
    "tau_spike": 1.5,
    "theta": -50.0,
    "tau_rise_AMPA": 0.2,
    "tau_rise_GABA_A": 1.0,
    "tau_rise_GABA_B": 30.0,
    "tau_rise_NMDA": 9.0,
    "tau_decay_AMPA": 3.0,
    "tau_decay_NMDA": 148.5,
    "tau_decay_GABA_A": 10.0,
    "tau_decay_GABA_B": 250.0,
    "g_peak_AMPA": 0.6,
    "g_peak_NMDA": 0.7,
    "g_peak_GABA_A": 1.0,
    "g_peak_h": 0.0,
    "g_peak_NaP": 0.0,
    "E_rev_GABA_A": -90.0,
    "g_peak_T": 0.5,
    "g_peak_KNa": 3.0
}




neuron_mod_nest = "ht_neuron"
neuron_mod_custom = neuron_model_name
neuron_type_nest = 'nest_ht'
neuron_type_custom = 'custom_ht'

# Initialisation of the neuron type
nest.CopyModel(neuron_mod_nest, neuron_type_nest)
nest.SetDefaults(neuron_type_nest, neuron_params_nest)

nest.CopyModel(neuron_mod_custom, neuron_type_custom)
nest.SetDefaults(neuron_type_custom, neuron_params_cus)

post_cus = nest.Create(neuron_type_custom)
post = 	   nest.Create(neuron_type_nest)

# post_cus = nest.Create(neuron_model_name)
# post = 	   nest.Create('ht_neuron')

# dt_spike=30
n_spikes=1
# spike_times = np.append(10,np.arange(dt_spike, dt_spike*9, dt_spike)+100).astype(float)
parrots=[]
for i in range(n_neurons_in):
    spike_times=np.round(1+np.random.rand(n_spikes)*(sim_time-1))
    print(spike_times)
    spike_generator = nest.Create('spike_generator', params={'spike_times':spike_times})
    pn1 = nest.Create("parrot_neuron")
    nest.Connect(spike_generator, pn1)
    parrots.append(pn1)
	

weight=50
r_t=1
delay=0.5

syn_param_custom=get_cus_tso_params(synapse_model_name,r_t,weight,delay)
syn_param_nest=get_nest_tso_params(r_t,weight,delay)
print('custom')

print(post_cus.get().keys())
print('nest native')
print(post.get().keys())

for parrot in parrots:
    # Connect spike generator to the neuron
    nest.Connect(parrot, post,"all_to_all",syn_spec=syn_param_nest)
    nest.Connect(parrot, post_cus,"all_to_all",syn_spec=syn_param_custom)

# Create a spike recorder to record spikes
spike_recorder_parrot = nest.Create('spike_recorder')
spike_recorder_nest = nest.Create('spike_recorder')
spike_recorder_cus = nest.Create('spike_recorder')

# Connect the neuron to the spike recorder
nest.Connect(post, spike_recorder_nest)
nest.Connect(post_cus, spike_recorder_cus)
for parrot in parrots:
    nest.Connect(parrot, spike_recorder_parrot)

var_to_read =['V_m']

var_to_read=['V_m','I_h', 'I_KNa', 'I_NaP', 'I_T','g_AMPA', 'g_NMDA', 'g_GABA_A', 'g_GABA_B' ]
var_to_read_cus=['V_m','I_h', 'I_KNa', 'I_NaP', 'I_T','g_AMPA__DOLLAR__X__AMPA', 'g_NMDA__DOLLAR__X__NMDA', 'g_GABAA__DOLLAR__X__GABA_A', 'g_GABAB__DOLLAR__X__GABA_B'  ]

dt_mult=dt
m_cus = nest.Create("multimeter", params={"interval": dt_mult, "record_from":var_to_read_cus , "label": "my_multimeter"})
m = nest.Create("multimeter", params={"interval": dt_mult, "record_from":var_to_read , "label": "my_multimeter"})


synapses = nest.GetConnections(pn1, post)

nest.Connect(m, post)
nest.Connect(m_cus, post_cus)



synapses = nest.GetConnections(pn1, post)
synapses_cus = nest.GetConnections(pn1, post_cus)
print('nest syn ',synapses)
print('cus syn ',synapses_cus)
print('nest neu ',post)
print('cus neu ',post_cus)


# nest.Simulate(500)
# nest.Simulate(1)
us_cus_step = []
ys_cus_step = []
xs_cus_step = []
us = []
ys=[]
xs = []
conn_cus1=nest.GetConnections(pn1, post_cus)
conn_1=nest.GetConnections(pn1, post)
# print(conn_cus1.get().keys())
# print(conn_1.get().keys())
nest.set_verbosity('M_ERROR' )
nest.Prepare()
for i in range(sim_time):
	nest.Run(1)
	# synapses = nest.GetConnections(pn1, post)
	# synapses_cus = nest.GetConnections(pn1, post_cus)

	# u=nest.GetStatus(synapses, 'u')[0]
	# y=nest.GetStatus(synapses, 'y')[0]
	# x=nest.GetStatus(synapses, 'x')[0]
	# us.append(u)
	# xs.append(x)
	# ys.append(y)
	


	# us_cus_step.append(nest.GetStatus(synapses_cus, 'u_step')[0])
	# ys_cus_step.append(nest.GetStatus(synapses_cus, 'y_step')[0])
	# xs_cus_step.append(nest.GetStatus(synapses_cus, 'x_step')[0])
	
		



spike_events_in = nest.GetStatus(spike_recorder_parrot, 'events')[0]
print("Recorded parrot spike times:", spike_events_in['times'])

spike_events = nest.GetStatus(spike_recorder_nest, 'events')[0]
print("Recorded nest spike times:", spike_events['times'])

spike_events = nest.GetStatus(spike_recorder_cus, 'events')[0]
print("Recorded custom spike times:", spike_events['times'])


folder= '/Tso_STDP_syn/outputs/images_tsodyks'

events_cus = nest.GetStatus(m_cus)[0]["events"]
events = nest.GetStatus(m)[0]["events"]


plt.figure(figsize=(200,150))
plt.plot(events_cus['V_m'],label='custom VM')
plt.plot(events['V_m'],'--',label='nest VM')
plt.legend()
plt.savefig(f'{folder}/VMs_cus_{n_neurons_in}.png')
plt.close()

plt.plot(events_cus['V_m'],label='custom VM')
plt.plot(events['V_m'],'--',label='nest VM')
plt.xlim([40000,50000])
plt.legend()
plt.savefig(f'{folder}/VMs_zoom_{n_neurons_in}.png')
plt.close()


plt.plot(events_cus['V_m']-events['V_m'])
plt.savefig(f'{folder}/V_diff_{n_neurons_in}.png')
plt.close()




spike_events = nest.GetStatus(spike_recorder_nest, 'events')[0]
times_nest=spike_events['times']
ids_nest=spike_events['senders']
print("Recorded nest spike times:", times_nest)

spike_events = nest.GetStatus(spike_recorder_cus, 'events')[0]
times_cus=spike_events['times']
ids_cus=spike_events['senders']
print("Recorded custom spike times:", spike_events['times'])



fig, ax1 = plt.subplots(figsize=(30, 15))

# Scatter plot (asse primario)
ax1.scatter(times_nest, ids_nest-np.min(ids_nest)+np.min(ids_cus), s=5, marker='|', label="NEST", color='red')
ax1.scatter(times_cus, ids_cus, s=1, marker='|', label="Custom", color='blue')
ax1.set_xlabel("Time (ms)", fontsize=16)
ax1.set_ylabel("Neuron ID", fontsize=16)
ax1.set_title("Spiking Activity from NEST Simulation", fontsize=18)


plt.savefig('scatter_nest.png')

plt.close()


plt.plot(times_nest,times_nest-times_cus)
plt.title('spike time differences')
plt.savefig('spike_time_diffs.png')
plt.close()



# I_e external always 0

# for i,var in enumerate(var_to_read):
#     for j,spike in enumerate(10*spike_events_in['times']): #show spikes as vertical lines in the final plot
#         spike=int(spike)
#         plt.axvline(x=spike,color='g')
#     I_cus = events_cus[var_to_read_cus[i]]
#     I_syn = events[var_to_read[i]]
#     plt.plot(I_cus,label='custom')
#     plt.plot(I_syn,'--',label='nest')
#     plt.legend()
#     plt.savefig(f"{folder}/{var}_TSO.png")
#     plt.close()


# for i,spike in enumerate(2*spike_events_in['times']):
#     spike=int(spike)
#     plt.axvline(x=spike)
# plt.plot(us_cus_step,label='custom')
# plt.plot(us,'--',label='nest')
# plt.title('u')
# plt.legend()
# plt.savefig(f"{folder}/us.png")
# plt.close()



# for i,spike in enumerate(2*spike_events_in['times']):
#     spike=int(spike)
#     plt.axvline(x=spike)
# plt.plot(ys_cus_step,label='custom')
# plt.plot(ys,'--',label='nest')
# plt.title('y')

# plt.legend()
# plt.savefig(f"{folder}/ys.png")
# plt.close()


# for i,spike in enumerate(2*spike_events_in['times']):
#     spike=int(spike)
#     plt.axvline(x=spike)
# plt.plot(xs_cus_step,label='custom')
# plt.plot(xs,'--',label='nest')
# plt.title('x')
# plt.legend()
# plt.savefig(f"{folder}/xs.png")
# plt.close()


'''