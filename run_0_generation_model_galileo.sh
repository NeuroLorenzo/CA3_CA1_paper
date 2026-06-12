#!/bin/bash

#SBATCH --partition=g100_usr_smem
#SBATCH --job-name=train_son_nestml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

#SBATCH --mem=360G
#SBATCH --time=15:00:00
#SBATCH --output=NESTML_compile.txt
#SBATCH --error=NESTML_compile.txt
#SBATCH --account=EIRI_E_UNIMR2


singularity run --bind  ../FAIR_folder:/base_folder install/NEST_nestml_SM6.sif bash -c "
. /opt/nest_env/bin/activate
cd /base_folder
python scripts/TSODYKS_STDP_HT_Generate_new_nml.py
"

