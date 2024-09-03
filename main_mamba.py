import os

if __name__ == '__main__':
    os.system("accelerate launch --num-processes=1 --gpu_ids 1 --main_process_port 29502 train_mamba.py --model TrafficDiffuser-B --max-num-agents 46 --hist-length 8 --seq-length 5 --use-ckpt-wrapper")
    os.system("python sample_mamba.py --ckpt /data/ahmed.ghorbel/workdir/autod/TrafficDiffuser/results/008-TrafficDiffuser-B --use-ckpt-wrapper")