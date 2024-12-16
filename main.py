import os

if __name__ == '__main__':
    
    # Train on multiple gpus
    os.system("accelerate launch -m scripts.train_wmp --config configs/config_train2.yaml")
    
    os.system("accelerate launch --main_process_port 29502 -m scripts.train --config configs/config_train1.yaml")

    # Train on 1 gpu
    os.system("accelerate launch --num-processes=1 --gpu_ids 1 -m scripts.train --config configs/config_train.yaml")

    # Sample
    os.system("python -m scripts.sample --config configs/config_sample.yaml")
    