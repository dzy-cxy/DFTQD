#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=torch_sample

module purge

singularity exec --nv \
	    --overlay $SCRATCH/singularity-images/cp38-torch1.13-cu117.ext3:ro \
	    /vast/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
		/bin/bash -c "source /vast/zd2362/python.sh; python /vast/zd2362/Project/Q-Diffusion/DFTQD/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 4 --quant_mode qdiff --quant_act --act_bit 8 --a_sym --split --resume -l /vast/zd2362/Project/Q-Diffusion/DFTQD/output --cali_ckpt /vast/zd2362/Project/Q-Diffusion/DFTQD/ckpt/cifar_w4a8_ckpt.pth --cali_st 20 --cali_n 256 --exp /vast/zd2362/Project/Q-Diffusion/DFTQD --doc cifar_1 --fine_tune --ni"