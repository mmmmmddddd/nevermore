import os

import pytorch_lightning as pl
from easydict import EasyDict as edict
from pl_extension.loggers import LoggingLogger
from pl_extension.utilities.file_io import load_file
from pytorch_lightning.callbacks import ModelCheckpoint

import nevermore # noqa: F401


def main():
    import argparse
    parser = argparse.ArgumentParser(description='train cli')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    cfg = edict(load_file(args.config))
    pl.seed_everything(cfg.seed_everything)

    # ------------
    # traincli
    # ------------
    if cfg.get('traincli', None) and not os.path.exists('/running_package'):
        from pint_horizon.aidi import traincli
        traincli(cfg.traincli)
        return

    # ------------
    # args
    # ------------
    if os.path.exists('/running_package'):
        # run in remote, not local
        save_dir = cfg.data.remote_save_dir
        cfg.data.init_args.data_root = cfg.data.remote_data_root
    else:
        save_dir = os.path.join("tmp_outputs")

    # ------------
    # data
    # ------------
    datamodule_class = eval(cfg.data.pop('class_type'))
    dm = datamodule_class(**cfg.data.init_args)

    # ------------
    # model
    # ------------
    module_class = eval(cfg.model.pop('class_type'))
    model = module_class(**cfg.model.init_args)

    # ------------
    # callback
    # ------------
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename='sample-NYUv2-' + cfg.model.init_args.task +
        '-{epoch:02d}-{val_loss:.2f}'
    )

    # ------------
    # logger
    # ------------
    logging_logger = LoggingLogger(logdir=save_dir, prefix='nevermore')

    # ------------
    # training
    # ------------
    pl_config = dict(cfg.trainer)
    pl_config['callbacks'] = [checkpoint_callback]
    pl_config['default_root_dir'] = save_dir
    pl_config['logger'] = [logging_logger]
    trainer = pl.Trainer(**pl_config)
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    # trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
