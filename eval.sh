cd /home/helix/projects/METRA-Projects/FlashSAC
MUJOCO_GL=egl uv run python plot_metra_eval.py \
  --checkpoint-path /home/helix/projects/METRA-Projects/FlashSAC/models/compare/test-postion-info/Ant-v4/seed0-0525-165746/epoch460 \
  --config-path configs \
  --config-name metra_aligned_base \
  --output-dir ./eval_artifacts \
  --num-random-trajectories 48 \
  --num-video-repeats 2 \
  --video-fps 15 \
  --video-skip-frames 1