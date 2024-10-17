import os
import multiprocessing



def run_sample(ckpt, gpu_id, module="model_td", model="TrafficDiffuser-S", nagents=46, histl=8, seql=5, n_steps=250, 
               use_map_embed=True, use_gmlp=False):
    # Base command
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python sample.py --ckpt {ckpt} \
        --model-module {module} --model {model} --max-num-agents {nagents} --hist-length {histl} --seq-length {seql} \
        --num-sampling-steps {n_steps}"
    
    # Conditionally add flags based on boolean values
    if use_map_embed:
        command += " --use-map-embed"
    if use_gmlp:
        command += " --use-gmlp"
    
    # Run the command
    os.system(command)
    
    
if __name__ == '__main__':
    
    os.system("accelerate launch train.py --model-module model_td --model TrafficDiffuser-S \
        --max-num-agents 46 --hist-length 8 --seq-length 5 --use-map-embed")
    os.system("accelerate launch train.py --model-module model_td_ --model TrafficDiffuser-S \
        --max-num-agents 46 --hist-length 8 --seq-length 5 --use-map-embed")
    
    process_1 = multiprocessing.Process(
        target=run_sample,
        args=("/data/ahmed.ghorbel/workdir/autod/traffic-diffuser/results/000-TrafficDiffuser-S/checkpoints/0084000.pt", "0", "model_td")
    )
    process_2 = multiprocessing.Process(
        target=run_sample,
        args=("/data/ahmed.ghorbel/workdir/autod/traffic-diffuser/results/001-TrafficDiffuser-S/checkpoints/0084000.pt", "1", "model_td_")
    )

    # Start both processes simultaneously
    # The main script will finish immediately after starting both processes.
    process_1.start()
    process_2.start()
    
    # if you plan to monitor or perform additional tasks after sampling is complete,
    # you should use join() to ensure the main script waits for both processes to finish
    # Wait for both processes to finish
    #process_1.join()
    #process_2.join()
    
    