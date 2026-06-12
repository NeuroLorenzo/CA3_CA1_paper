#!/bin/bash
#SBATCH --partition=g100_usr_smem
#SBATCH --job-name=plot_ca3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --output=plot_output.txt
#SBATCH --error=plot_error.txt
#SBATCH --account=EIRI_E_UNIMR2
 

###############
# sonata_folder = sys.argv[1]   #'sonata_network_upscale_train'
# placement_file = sys.argv[2]
# trials = int(sys.argv[3])          #10
# train_test = sys.argv[4]      #train/test
# gid_stim_file = sys.argv[5] #'gids_400_sl7_pc_update.csv'
# shuff = int(sys.argv[6])
################

sonata_folder='network_4000_400' #"network_4000_400" #"network_sl7_400" 00" #
placement_file="placement_4000_ei_CA3_sl7_remap.h5" # "placement_4000_ei_CA3_sl7_remap.h5" #"placements_CA3_sl7_remap.h5"  #
trials=80
train_test='test' #'test' #'train'
gid_stim_file='gid_stim_400_4000.csv' #'gid_stim_400_sl7.csv'
shuff=0

singularity run --bind ../FAIR_folder:/base_folder install/NEST_nestml_SM6.sif bash -c "
. /opt/nest_env/bin/activate
cd /base_folder
python scripts/plot_raster_activity.py $sonata_folder $placement_file $trials $train_test $gid_stim_file $shuff 
"
