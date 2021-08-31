from pint_horizon.aidi import traincli

cfg = dict(
    gpus="0,1,2,3",
    num_machines=1,
    submit_root="./",
    job_name="NYUv2-seg-lr2e-5-4gpus",
    job_pwd=5160,
    project_id="TD2021002",
    docker_image=\
    "docker.hobot.cc/imagesys/hdlt:fsd_multitask-cu10-20210621-v0.3",  # noqa
    job_list=[
        "pip install pytorch-lightning",
        "mkdir -p /home/users/dixiao.wei/.cache/torch/hub/checkpoints",
        "cp /cluster_home/custom_data/vgg16-397923af.pth /home/users/dixiao.wei/.cache/torch/hub/checkpoints/",  # noqa
        "python3 train.py",
    ],
    task_label="mnist",
    priority=5,
)
traincli(cfg)
