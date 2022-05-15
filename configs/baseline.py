seed_everything = 1822

data = dict(
    class_type="nevermore.data.NYUv2DataModule",
    init_args=dict(
        data_root="data/NYUv2",
        input_size=(320, 320),
        output_size=(320, 320),
        batch_size=1,
        num_workers=4,
    ),
    remote_data_root="/cluster_home/custom_data/NYU",
    remote_save_dir="/job_data",
)

model = dict(
    class_type="nevermore.model.SingleSegnetNYUv2Model",
    init_args=dict(
        learning_rate=2e-5,
        task="segmentation",
    )
)

trainer = dict(
    max_epochs=8890,
    gpus="0,1",
    check_val_every_n_epoch=1,
    accelerator="ddp",
    num_sanity_val_steps=0,
    progress_bar_refresh_rate=0,
    log_every_n_steps=5,
)
