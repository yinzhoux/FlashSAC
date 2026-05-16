```bash
python tests/main.py 
--run_group Debug 
--env ant 
--max_path_length 200 
--seed 0 
--traj_batch_size 8 
--n_parallel 1 
--normalizer_type preset
--eval_plot_axis -50 50 -50 50 
--trans_optimization_epochs 50 
--n_epochs_per_log 100 
--n_epochs_per_eval 1000 
--n_epochs_per_save 10000 
--sac_max_buffer_size 1000000 
--algo metra 
--discrete 0 
--dim_option 2
```
sac_tau = 5e-3
