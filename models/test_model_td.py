import torch
import time
from fvcore.nn import FlopCountAnalysis
from models.backbones.model_td import TrafficDiffuser_models
from fvcore.nn import FlopCountAnalysis
from diffusion import create_diffusion


# Test setup
model_name = 'TrafficDiffuser-S'
batch_size = 4
max_num_agents = 20
hist_length = 8
seq_length = 5
dim_size = 2
use_map_embed = False
map_channels = 4
map_size = 256
map_ft = 32
map_length = 128
interm_size = 64
use_ckpt_wrapper = False

num_sampling_steps = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create random torch tensors
dummy_x = torch.randn(batch_size, max_num_agents, seq_length, dim_size, device=device)
dummy_t = torch.randn(batch_size, device=device)
dummy_h = torch.randn(batch_size, max_num_agents, hist_length, dim_size, device=device)
data = torch.cat([dummy_x, dummy_h], dim=2)
B, N, L, _ = data.shape
print(data.shape)
dummy_mask = (data.sum(dim=-1) != 0).unsqueeze(2).repeat(1, 1, L, 1).view(B * N, L, L)
num_heads = 8 # for model S
dummy_mask = dummy_mask.unsqueeze(1).repeat(1, num_heads, 1, 1).view(B * N * num_heads, L, L)
print(dummy_mask.shape)
print(dummy_mask)

if use_map_embed:
    dummy_m = torch.randn(batch_size, map_channels, map_size, map_size, device=device)
else:
    dummy_m = None
            
# Initialize the models
model = TrafficDiffuser_models[model_name](
    max_num_agents=max_num_agents,
    seq_length=seq_length,
    hist_length=hist_length,
    dim_size=dim_size,
    map_ft=map_ft,
    map_length=map_length,
    interm_size=interm_size,
    use_map_embed=use_map_embed,
    use_ckpt_wrapper=use_ckpt_wrapper,
).to(device)
#model.eval() 

# Print model parameters, and summary
print(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"{model_name} Model summary:\n{model}")

# Print model flops
flops = FlopCountAnalysis(model, (dummy_x, dummy_t, dummy_h, dummy_mask))
gflops = flops.total() / 1e9
print(f"{model_name} GFLOPs: {gflops:.4f}")

# Print model sampling time
model_kwargs = dict(h=dummy_h, mask=dummy_mask)
num_trials = 10
avg_sampling_time = 0

diffusion = create_diffusion(timestep_respacing=str(num_sampling_steps))

for _ in range(num_trials):
    torch.cuda.synchronize()
    tic = time.time()
    samples = diffusion.p_sample_loop(
            model.forward, dummy_x.shape, dummy_x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    torch.cuda.synchronize()
    toc = time.time()
    avg_sampling_time += (toc - tic)
avg_sampling_time /= num_trials
print(f"{model_name} Sampling time: {avg_sampling_time:.2f} s")