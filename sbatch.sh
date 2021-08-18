#!/bin/sh

############################################################
##### Run python script as slurm job:                  #####
##### Create temporary sbatch job script               #####
##### Fill it with slurm options                       #####
##### Then fill it with the script to run and its args #####
##### Then submit it to slurm cluster                  #####
############################################################

# =====================
# How to call this script
# =====================
# call this script as:
#    ./sbatch.sh python train_apc.py --experiment_name test --batch_size 32
#    ./sbatch.sh python train_tacotron.py --hp_file hparams.py
#    ./sbatch.sh python gen_tacotron.py --hp_file hparams.py
# note you can specify a gpu, this will ensure that the job is only submitted to nodes with that GPU
#    ./sbatch.sh 2080 python train_tacotron.py --hp_file hparams.py


# =====================
# define directories used for copying preprocessed audio data to the node's scratch disk 
# =====================
#repo_home="/home/${USER}/representation-mixing"
#scratch_folder="/disk/scratch/${USER}"

# =====================
# Global vars
# =====================

LOG_DIR="./slurm_logs"

# =====================
# check if specific gpu was supplied
# =====================

if [ $1 = "2080" ]; then
    gpu_type="gtx2080ti:"
    cmd_to_run_on_cluster=${@:2} #skip first argument as this is the gpu specified
else
    gpu_type=""
    cmd_to_run_on_cluster=${@} #use all supplied arguments i.e. "python train_tacotron.py --hp_file hparams.py"
fi

# =====================
# Create directory for logging files if it doesn't exist
# =====================

if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR;
fi

# =====================
# setup sbatch params
# =====================
nodes=1
gpu_num=1
gpus=${gpu_type}${gpu_num} #note if gpu_type is empty string then it is just gpu_num, which is fine
# cpus=$(( 2*gpus ))
cpus=1 #might need to use only 1 cpu if node does not have many cores
tasks=1
#part=ILCC_GPU
part=ILCC_GPU,CDT_GPU
# part=ILCC_GPU,CDT_GPU,M_AND_I_GPU
time=10-00:00:00
#mem=8G
mem=16G
mail_user=s1785140@sms.ed.ac.uk
# mail_type=BEGIN,END,FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT # same as ALL
mail_type=END,FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT
# Exclude particular nodes (used when nodes scratch disk is full)
# exclude="--exclude=nuesslein"

# =====================
# create the sbatch file
# =====================
#shebang
echo '#!/bin/bash' > temp_slurm_job.sh

#job params
echo "#SBATCH --nodes=${nodes}" >> temp_slurm_job.sh
echo "#SBATCH --gres=gpu:${gpus}" >> temp_slurm_job.sh
echo "#SBATCH --cpus-per-task=${cpus}" >> temp_slurm_job.sh
echo "#SBATCH --mem=${mem}" >> temp_slurm_job.sh
echo "#SBATCH --ntasks=${tasks}" >> temp_slurm_job.sh
echo "#SBATCH --time=${time}" >> temp_slurm_job.sh
echo "#SBATCH --partition=${part}" >> temp_slurm_job.sh
echo "#SBATCH --mail-user=${mail_user}" >> temp_slurm_job.sh
echo "#SBATCH --mail-type=${mail_type}" >> temp_slurm_job.sh
echo "#SBATCH --output=${LOG_DIR}/%j" >> temp_slurm_job.sh #Note! remember to make this directory if it doesn't exist

#rsyncing of data to scratch disk
# echo "rsync -avu ${repo_home}/data $scratch_folder/" >> temp_slurm_job.sh #move preprocessed data from repo dir to the scratch disk
# echo 'if [ "$?" -eq "0" ]' >> temp_slurm_job.sh #'$?' holds result of last command, '0' is success
# echo 'then' >> temp_slurm_job.sh
# echo "  echo \"Rsync succeeded.\"; data_path_flag=\"--data_path ${scratch_folder}/data\"" >> temp_slurm_job.sh #load data from scratch disk
# echo 'else' >> temp_slurm_job.sh
# echo '  echo "Error while running rsync."; data_path_flag=""' >> temp_slurm_job.sh #load data over network
# echo 'fi' >> temp_slurm_job.sh

##pre experiment logging
#start_date=`date '+%d/%m/%Y %H:%M:%S'`
#echo "echo \"Job started: $start_date\"" >> temp_slurm_job.sh
#echo "start=`date +%s`" >> temp_slurm_job.sh

#actual command to be run on cluster
#echo "srun ${cmd_to_run_on_cluster} \${data_path_flag}" >> temp_slurm_job.sh
echo "srun ${cmd_to_run_on_cluster}" >> temp_slurm_job.sh

##post experiment logging
#echo "echo \"Job started: $start_date\"" >> temp_slurm_job.sh
#echo "echo \"Job finished: $(date '+%d/%m/%Y %H:%M:%S')\"" >> temp_slurm_job.sh
#echo "end=`date +%s`" >> temp_slurm_job.sh
#echo "runtime=$((end-start))" >> temp_slurm_job.sh
#echo "echo \"Job took: $runtime seconds\"" >> temp_slurm_job.sh

# =====================
# submit this temporary sbatch script to the cluster
# =====================
# cat temp_slurm_job.sh #debug
sbatch temp_slurm_job.sh
# rm temp_slurm_job.sh #dont need to do this if u want to inspect/modify the job script that was created
