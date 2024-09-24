#export http_proxy=http://192.168.48.17:18000
#export https_proxy=http://192.168.48.17:18000
export HF_HOME="/mnt/pfs/share/pretrained_model/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
pip install open_clip_torch

#python get_dino_feature.py test.txt test.txt.bin
# python get_dino_feature.py 434w_sketchfab.txt 434w_sketchfab.txt.dinov2.bin  
#torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 get_clip_378_feature.py 434w_sketchfab.txt 434w_sketchfab.txt.clip.378.bin  
#torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 get_clip_378_feature.py new_normal_4view_success.txt new_normal_4view_success.txt.bin
#torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 get_clip_378_feature.py v0.2/tmp_sketchafb_no_big_small.txt v0.2/tmp_sketchafb_no_big_small.txt.bin
#torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 get_clip_378_feature.py real_input.txt real_input.txt.bin
#torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 get_clip_378_feature.py thingiverse_4view_normal_success.txt thingiverse_4view_normal_success.txt.bin
#torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 get_clip_378_feature.py thingiverse_4view_normal_success.txt thingiverse_4view_normal_success.txt.rgb.bin
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 get_clip_378_feature.py 190w_other_source.txt 190w_other_source.txt.rgb.bin
