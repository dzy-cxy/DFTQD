#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=torch_sample

module purge

singularity exec --nv \
	    --overlay $SCRATCH/singularity-images/cp38-torch1.13-cu117.ext3:ro \
	    /vast/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /vast/zd2362/python.sh; python /vast/zd2362/Project/Q-Diffusion/q-diffusion/sample_diffusion_ldm.py -r models/ldm/lsun_beds256/model.ckpt -n 50000 --batch_size 10 -c 200 -e 1.0  --seed 40 --ptq  --weight_bit 2 --quant_mode qdiff --cali_st 20 --cali_batch_size 8 --cali_n 256 --quant_act --act_bit 8 --a_sym --a_min_max --running_stat --cali_data_path /vast/zd2362/Project/Q-Diffusion/q-diffusion/Dataset/Bedroom/bedroom_sample2040_allst.pt -l /vast/zd2362/Project/Q-Diffusion/q-diffusion/saved_model"