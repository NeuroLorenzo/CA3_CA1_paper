#!/bin/bash
#SBATCH --partition=g100_usr_smem
#SBATCH --job-name=recall_index
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --output=ri_output.txt
#SBATCH --error=ri_error.txt
#SBATCH --account=EIRI_E_UNIMR2
 

###############
# sonata_folder = sys.argv[1]   #'sonata_network_upscale_train'
# placement_file = sys.argv[2]
# trials = int(sys.argv[3])          #10
# train_test = sys.argv[4]      #train/test
# gid_stim_file = sys.argv[5] #'gids_400_sl7_pc_update.csv'
# shuff = int(sys.argv[6])
################

sonata_folder='network_4000_400/' #"network_4000_400" #"network_sl7_400" 00" #
train_activity_file='activity_train120.csv'
gid_stim_file='gid_stim_400_4000.csv' #'gid_stim_400_sl7.csv'
shuffle_folder='shuffle_folder'
trials_exp="30,40,50,60,70,80,90,100"

singularity run --bind ../FAIR_folder:/base_folder install/NEST_nestml_SM6.sif bash -c "
. /opt/nest_env/bin/activate
cd /base_folder
python scripts/recall_index_trials.py $sonata_folder $train_activity_file $gid_stim_file $shuffle_folder $trials_exp  
"
#python scripts/recall_index.py $sonata_folder $train_activity_file $gid_stim_file $shuffle_folder $trials_exp
