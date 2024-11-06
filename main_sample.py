import os

if __name__ == '__main__':
    
    os.system("python sample.py --cuda-device 1 \
        --ckpt /data/ahmed.ghorbel/workdir/autod/traffic-diffuser/results/000-TrafficDiffuser-S/checkpoints/0030000.pt \
            --model-module model_td --model TrafficDiffuser-S --max-num-agents 20 --hist-length 8 --seq-length 5 \
                --num-sampling-steps 100 --test-dir /data/ahmed.ghorbel/workdir/autod/traffic-diffuser/backup/data/nuscenes_trainval_clean_test")

    
    