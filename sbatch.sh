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

LOG_DIR="./logs_slurm"

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
mem=8G
mail_user=s1785140@sms.ed.ac.uk
# mail_type=BEGIN,END,FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT # same as ALL
mail_type=END,FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT
# Exclude particular nodes (used when nodes scratch disk is full)
# exclude="--exclude=nuesslein"

# =====================
# create the sbatch file
# =====================
#shebang
echo '#!/bin/bash' > job.sh

#job params
echo "#SBATCH --nodes=${nodes}" >> job.sh
echo "#SBATCH --gres=gpu:${gpus}" >> job.sh
echo "#SBATCH --cpus-per-task=${cpus}" >> job.sh
echo "#SBATCH --mem=${mem}" >> job.sh
echo "#SBATCH --ntasks=${tasks}" >> job.sh
echo "#SBATCH --time=${time}" >> job.sh
echo "#SBATCH --partition=${part}" >> job.sh
echo "#SBATCH --mail-user=${mail_user}" >> job.sh
echo "#SBATCH --mail-type=${mail_type}" >> job.sh
echo "#SBATCH --output=${LOG_DIR}/%j" >> job.sh #Note! remember to make this directory if it doesn't exist

#rsyncing of data to scratch disk
# echo "rsync -avu ${repo_home}/data $scratch_folder/" >> job.sh #move preprocessed data from repo dir to the scratch disk
# echo 'if [ "$?" -eq "0" ]' >> job.sh #'$?' holds result of last command, '0' is success
# echo 'then' >> job.sh
# echo "  echo \"Rsync succeeded.\"; data_path_flag=\"--data_path ${scratch_folder}/data\"" >> job.sh #load data from scratch disk
# echo 'else' >> job.sh
# echo '  echo "Error while running rsync."; data_path_flag=""' >> job.sh #load data over network
# echo 'fi' >> job.sh

##pre experiment logging
#start_date=`date '+%d/%m/%Y %H:%M:%S'`
#echo "echo \"Job started: $start_date\"" >> job.sh
#echo "start=`date +%s`" >> job.sh

#actual command to be run on cluster
#echo "srun ${cmd_to_run_on_cluster} \${data_path_flag}" >> job.sh
echo "srun ${cmd_to_run_on_cluster}" >> job.sh

##post experiment logging
#echo "echo \"Job started: $start_date\"" >> job.sh
#echo "echo \"Job finished: $(date '+%d/%m/%Y %H:%M:%S')\"" >> job.sh
#echo "end=`date +%s`" >> job.sh
#echo "runtime=$((end-start))" >> job.sh
#echo "echo \"Job took: $runtime seconds\"" >> job.sh

# =====================
# submit this temporary sbatch script to the cluster
# =====================
# cat job.sh #debug
sbatch job.sh
# rm job.sh #dont need to do this if u want to inspect/modify the job script that was created
