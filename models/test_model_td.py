import torch
import time
from calflops import calculate_flops
from diffusion import create_diffusion
from models.backbones.model_td import TrafficDiffuser_models



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
name = "TrafficDiffuser-H"
num_agents = 10
seq_length = 8
hist_length = 3
dim_size = 2
use_map_embed = True
num_sampling_steps = 1000

# Create random torch tensors
batch_size = 1 # to ensure flops and inference time are calculated for a single scenario
dummy_x = torch.randn(batch_size, num_agents, seq_length, dim_size, device=device)
dummy_t = torch.randn(batch_size, device=device)
dummy_h = torch.randn(batch_size, num_agents, hist_length, dim_size, device=device)
if use_map_embed:
    num_segments = 16
    dummy_mp = torch.zeros(batch_size, num_agents, num_segments, 128, 2).to(device)
else: 
    dummy_mp = None
            
# Model init
model = TrafficDiffuser_models[name](
    num_agents=num_agents,
    seq_length=seq_length,
    hist_length=hist_length,
    dim_size=dim_size,
    use_map_embed=use_map_embed,
    use_ckpt_wrapper=False,
).to(device)
model.eval() 

# Print model parameters, and summary
print(f"{name} Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"{name} Model summary:\n{model}")

# Print model flops
flops, macs, params = calculate_flops(
    model=model,
    args=[],  # No positional arguments
    kwargs={'x': dummy_x, 't': dummy_t, 'h': dummy_h, 'mp': dummy_mp},
    print_results=True,
    print_detailed=True,
    output_as_string=True,
    output_precision=4,
)

# Diffusion init
diffusion = create_diffusion(timestep_respacing=str(num_sampling_steps))

# Print model sampling time
print(f"---------------- {name} Sampling time calculation ----------------")
model_kwargs = dict(h=dummy_h, mp=dummy_mp)
num_trials = 10
avg_sampling_time = 0
for _ in range(num_trials):
    torch.cuda.synchronize()
    tic = time.time()
    samples = diffusion.p_sample_loop(
            model.forward, dummy_x.shape, dummy_x, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
    )
    torch.cuda.synchronize()
    toc = time.time()
    avg_sampling_time += (toc - tic)
avg_sampling_time /= num_trials
print(f"{name} Sampling time: {avg_sampling_time:.2f} s")
print(f"-----------------------------------------------------------------------------")