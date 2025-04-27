ssh -p 50673 root@connect.bjc1.seetacloud.com
ybwIqZbQoO19


# merge
python lerobot/scripts/merge.py --max_dim 32 --sources /home/yons/media/.cache/huggingface/lerobot/shelbin/act10_32 /home/yons/media/.cache/huggingface/lerobot/shelbin/box_01 --output /home/yons/media/.cache/huggingface/lerobot/shelbin/all

# box_002/box_005/box_006box_007

# record task
# act_007
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the block and place it on the plate." \
  --control.repo_id=${HF_USER}/bluebox \
  --control.warmup_time_s=3 \
  --control.episode_time_s=30 \
  --control.reset_time_s=3 \
  --control.num_episodes=20 \
  --control.push_to_hub=false \
  --control.resume=true

# 接着这个数据集继续采
  --control.resume=true



# reply
python lerobot/scripts/control_robot.py \
--robot.type=so100 \
--control.type=replay \
--control.fps=30 \
--control.repo_id=${HF_USER}/so100_006 \
--control.episode=0



# visualize
# ~/.cache/huggingface/lerobot/shelbin/
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/all2

  

# local train 
python lerobot/scripts/train.py \
--dataset.repo_id=${HF_USER}/bluebox \
--policy.type=act \
--output_dir=./models/bluebox \
--job_name=bluebox \
--policy.device=cuda \
--wandb.enable=true \
--dataset.image_transforms.enable=true



# eval
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the block and place it on the plate." \
  --control.repo_id=${HF_USER}/eval_bluebox \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=5 \
  --control.num_episodes=50 \
  --control.push_to_hub=false \
  --control.policy.path=./models/bluebox/checkpoints/100000/pretrained_model



# pycharm debug
  --robot.type=so100 
  --control.type=record 
  --control.fps=30 
  --control.single_task="Pick up the block and place it on the plate." 
  --control.repo_id=${HF_USER}/eval_box_002_07
  --control.warmup_time_s=5 
  --control.episode_time_s=30 
  --control.reset_time_s=10 
  --control.num_episodes=10 
  --control.push_to_hub=false 
  --control.policy.path=/media/yons/843b68f6-dfaf-466a-871e-769728918988/models/020000/pretrained_model





# 控制记录代码
/home/yons/app/lerobot/lerobot/common/robot_devices/robots/configs.py
control_robot.py#teleoperate
manipulator.py#teleop_step
  




    























mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/media/yons/843b68f6-dfaf-466a-871e-769728918988/conda_envs/lerobot
make -j$(nproc)
make install
cd ..


export PKG_CONFIG_PATH="/media/yons/843b68f6-dfaf-466a-871e-769728918988/conda_envs/lerobot/lib/pkgconfig:$PKG_CONFIG_PATH"

./configure --prefix=/media/yons/843b68f6-dfaf-466a-871e-769728918988/conda_envs/lerobot \
            --enable-shared \
            --disable-static \
            --enable-libsvtav1 \
            --enable-gpl \
            --enable-nonfree \
            --enable-libaom \
            --enable-libvpx \
            --enable-libopus \
            --enable-libvorbis \
            --enable-libass \
            --enable-libfreetype \
            --enable-libmp3lame \
            --enable-zlib \
            --enable-sdl2 \
            --extra-cflags="-I/media/yons/843b68f6-dfaf-466a-871e-769728918988/conda_envs/lerobotinclude" \
            --extra-ldflags="-L/media/yons/843b68f6-dfaf-466a-871e-769728918988/conda_envs/lerobot/lib"


python lerobot/scripts/control_robot.py \
  --robot.type=shelbin \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the block and place it on the plate." \
  --control.repo_id=${HF_USER}/test \
  --control.warmup_time_s=5 \
  --control.episode_time_s=60 \
  --control.reset_time_s=5 \
  --control.num_episodes=20 \
  --control.push_to_hub=false 





