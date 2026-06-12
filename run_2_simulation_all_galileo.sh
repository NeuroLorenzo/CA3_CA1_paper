#!/bin/bash

#SBATCH --partition=g100_usr_smem
#SBATCH --job-name=train_son_nestml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

#SBATCH --mem=50G #360G
#SBATCH --time=2:00:00 #24:00:00
#SBATCH --output=run_sim_output.txt
#SBATCH --error=run_sim_error.txt
#SBATCH --account=EIRI_E_UNIMR2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

sonata_folder='network_4000_400' #"network_4000_400" #"network_sl7_400" 
placement_file="placements_CA3_sl7_remap.h5" # "placement_4000_ei_CA3_sl7_remap.h5" #"placements_CA3_sl7_remap.h5"  #
trials=90
train_test='train' #test' #'train'
gid_stim_file='gid_stim_400_4000.csv' #gid_stim_400_4000.csv #gid_stim_400_sl7.csv
shuff=0
save_w=0
###########################################

# Run Singularity container and Python scripts
singularity run --bind ../FAIR_folder:/base_folder install/NEST_nestml_SM6.sif bash -c "

. /opt/nest_env/bin/activate
cd /base_folder
python scripts/Sonata_pipeline.py $sonata_folder $trials $train_test $gid_stim_file $shuff 
python scripts/save_output_simulation.py $sonata_folder $placement_file $trials $train_test $gid_stim_file $shuff
"



