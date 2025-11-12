#! /bin/bash

singularity exec --nv ~/Containers/tensorflow_2.4.1-gpu.sif python B_stage1_code_updated_w_gr.py --gpu 0 > B_stage1_code_updated_w_gr.out &
singularity exec --nv ~/Containers/tensorflow_2.4.1-gpu.sif python C_stage1_interval_bin.py --gpu 0 > C_stage1_interval_w_gr.out &
wait

singularity exec --nv ~/Containers/tensorflow_2.4.1-gpu.sif python D_export_updated_w_gr.py --gpu 0 -B 6 
python E_gan_updated2_w_gr.py --gpu 3 --serial 4 -B 20 > E_gan_updated2_w_gr.out
python F_gen_w_gr.py --gpu 1 --serial test1 -B 6 -C 12 -E 1 -k 1 -n 8000 > F_gen_w_gr.out
python G_distinguish_episode_w_gr.py --gpu 1 --folder > G_distinguish_episode_w_gr.out

