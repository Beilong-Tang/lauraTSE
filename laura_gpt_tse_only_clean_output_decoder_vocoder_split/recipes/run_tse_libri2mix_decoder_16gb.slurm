#!/bin/bash
#SBATCH -J run_tse_libri2mix_decoder_16gb
#SBATCH -N 4
#SBATCH -o log/run_tse_libri2mix_decoder_16gb.out
#SBATCH -e log/run_tse_libri2mix_decoder_16gb.err
#SBATCH -p kshdnormal
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4

export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PRICE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bltang_new

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-23.10

###########
# Setting #
###########

name="libri2mix"
config_path=exp/$name/config_log_mel_aux_5s_decoder.yaml
resume=""

###############
# DONT CHANGE #
###############
save_dir=$(dirname "$config_path")/$(basename "$config_path" .yaml)
save_dir=${save_dir/#exp\//}
echo $save_dir
ckpt_path=ckpt/${save_dir}_16gb
log_path=log/${save_dir}_16gb
mkdir -p $ckpt_path
mkdir -p $log_path

###############
## Run Slurm ##
###############
# change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
# change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=16

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python -u src/train.py --config $config_path --log $log_path --ckpt_path $ckpt_path --resume $resume \
    --fine_tune ""
###############
## Run  DDP  ##
###############
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# python -u src/train.py --config $config_path --log $log_path --ckpt_path $ckpt_path --resume $resume --fine_tune ""
