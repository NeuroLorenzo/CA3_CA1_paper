def get_cus_stat_params(synapse_model_name,r_t,weight,delay):
    syn_param_custom={
        "synapse_model": synapse_model_name,
        "receptor_type": r_t,
        "weight": weight,
        "delay": delay
    }
    return syn_param_custom

def get_nest_stat_params(r_t,weight,delay):
    syn_param_custom={
        "synapse_model": "static_synapse",
        "receptor_type": r_t,
        "weight": weight,
        "delay": delay

    }
    return syn_param_custom

def get_cus_tso_params(synapse_model_name,r_t,weight,delay):
    syn_param_custom={
        "synapse_model": synapse_model_name,
        "receptor_type": r_t,
        "weight": weight,
        "delay": delay,
        "tau_fac": 17.,
        "tau_psc": 5.,
        "tau_rec": 660.0,
        "U": 0.5,
        "dt": 0.1,
        "u" :0.,
        "x" :1.,
        "y" :0.,

    }
    return syn_param_custom

def get_nest_tso_params(r_t,weight,delay):
    syn_param_nest={
        "synapse_model": "tsodyks_synapse",

        "receptor_type": r_t,
        "weight": weight,
        "delay": delay,


        "tau_fac":17.,
        "tau_psc": 5.0,
        "tau_rec": 660.0,
        "U": 0.5,
        "u": 0.0,
        "x": 1.0,
        "y": 0.0
    }
    return syn_param_nest

