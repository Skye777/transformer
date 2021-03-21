"""
@author: Skye Cui
@file: train_distributed.py
@time: 2021/3/19 11:20
@description: 
"""
import os
os.environ["LOGURU_INFO_COLOR"] = "<green>"
import time
import re
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from loguru import logger
from progress.spinner import MoonSpinner
from input import *
from model import UTransformer
from loss import Loss

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

distribution = tf.distribute.MirroredStrategy()

GLOBAL_BATCH_SIZE = hp.batch_size * distribution.num_replicas_in_sync

with distribution.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)
    model = UTransformer(hp)
    model_loss = Loss(model)

    checkpoint_file = hp.ckpt
    if checkpoint_file == '':
        checkpoint_file = 'ckp_0.h5'
    else:
        model.load_weights(f'{hp.single_gpu_model_dir}/{checkpoint_file}')

with distribution.scope():
    def single_step(x_batch, ys_batch, model, flag='train'):
        with tf.GradientTape() as tape:
            y_predict = model(x_batch, ys_batch, training=True)
            loss_ssim, loss_l2, loss_l1, loss = model_loss((y_predict, ys_batch[1]))
        if flag == 'test':
            return loss_ssim, loss_l2, loss_l1, loss
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_ssim, loss_l2, loss_l1, loss

with distribution.scope():
    @tf.function
    def distributed_step(x_batch, ys_batch, model, flag='train'):
        loss_ssim, loss_l2, loss_l1, loss = distribution.run(single_step, args=(x_batch, ys_batch, model, flag))
        loss_ssim = distribution.reduce(tf.distribute.ReduceOp.MEAN, loss_ssim, axis=None)
        loss_l2 = distribution.reduce(tf.distribute.ReduceOp.MEAN, loss_l2, axis=None)
        loss_l1 = distribution.reduce(tf.distribute.ReduceOp.MEAN, loss_l1, axis=None)
        loss = distribution.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        return loss_ssim, loss_l2, loss_l1, loss

    logger.add(f"{hp.logdir}/{hp.in_seqlen}_{hp.out_seqlen}_{hp.lead_time}_train.log", enqueue=True)
    train_dataset, test_dataset = train_input_fn()
    train_dist_dataset = distribution.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = distribution.experimental_distribute_dataset(test_dataset)

    for epoch in range(hp.num_epochs):
        total_train = 0
        for step, (x_batch_train, ys_batch_train) in enumerate(train_dist_dataset):
            total_train += 1
        for step, (x_batch_train, ys_batch_train) in enumerate(train_dist_dataset):
            if step < total_train - 1:
                start = time.clock()
                loss_ssim, loss_l2, loss_l1, loss = distributed_step(x_batch_train, ys_batch_train, model)
                elapsed = (time.clock() - start)
                template = ("step {} loss is {:1.5f}, "
                            "loss ssim is {:1.5f}, "
                            "loss l2 is {:1.5f}, "
                            "loss l1 is {:1.5f}."
                            "({:1.2f}s/step)")
                logger.info(template.format(step, loss.numpy(), loss_ssim.numpy(), loss_l2.numpy(), loss_l1.numpy(), elapsed))
        if epoch % hp.num_epoch_record == 0:
            total_test = 0
            for step, (x_batch_test, ys_batch_test) in enumerate(test_dist_dataset):
                total_test += 1
            loss_test = 0
            loss_ssim_test = 0
            loss_l2_test = 0
            loss_l1_test = 0
            count = 0
            spinner = MoonSpinner('Testing ')
            for step, (x_batch_test, ys_batch_test) in enumerate(test_dist_dataset):
                if step < total_test - 1:
                    loss_ssim, loss_l2, loss_l1, loss = distributed_step(x_batch_test, ys_batch_test, model, flag='test')
                    loss_ssim_test += loss_ssim.numpy()
                    loss_l2_test += loss_l2.numpy()
                    loss_l1_test += loss_l1.numpy()
                    loss_test += loss.numpy()
                    count += 1
                spinner.next()
            spinner.finish()
            logger.info("TEST COMPLETE!")
            template = ("TEST DATASET STATISTICS: "
                        "loss is {:1.5f}, "
                        "loss ssim is {:1.5f}, "
                        "loss l2 is {:1.5f}, "
                        "loss l1 is {:1.5f}.")
            logger.info(template.format(loss_test/count, loss_ssim_test/count, loss_l2_test/count, loss_l1_test/count))

            total_epoch = int(re.findall("\d+", checkpoint_file)[0])
            checkpoint_file = checkpoint_file.replace(f'_{total_epoch}.h5', f'_{total_epoch + 1}.h5')
            model.save_weights(f'{hp.single_gpu_model_dir}/{checkpoint_file}')
            logger.info("Saved checkpoint_file {}".format(checkpoint_file))
