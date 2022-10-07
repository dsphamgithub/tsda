# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The main training script."""
import os
import platform
import math
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import horovod.tensorflow.keras as hvd

import dataloader
import hparams_config
import utils
from tf2 import tfmot
from tf2 import train_lib
from tf2 import util_keras

# from wandb.keras import WandbCallback
# from wandb import init as wandb_init

FLAGS = flags.FLAGS

# Horovod: initialize horovod.
hvd.init()

# Horovod: pin a GPU to current process.
physical_devices = tf.config.list_physical_devices('GPU')
local_rank = hvd.local_rank()
current_gpu = physical_devices[local_rank]

if local_rank >= len(physical_devices):
  raise Exception("Not enough gpus.")

try:
  tf.config.set_visible_devices(current_gpu, 'GPU')
  tf.config.experimental.set_memory_growth(current_gpu, True)
except:
  raise Exception('Invalid device or cannot modify virtual devices once initialized.')

logical_devices = tf.config.list_logical_devices('GPU')
if len(logical_devices) > 1:
  raise Exception('Only 1 GPU should be visible per MPI process but more than one is')
print([device.name for device in logical_devices])

def define_flags():
  """Define the flags."""
  # Model specific paramenters
  flags.DEFINE_string('model_dir', None, 'Location of model_dir')
  flags.DEFINE_string('pretrained_ckpt', None,
                      'Start training from this EfficientDet checkpoint.')
  flags.DEFINE_string(
      'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
      ' containing attributes to use as hyperparameters.')
  flags.DEFINE_integer('batch_size', 64, 'training batch size')
  flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                       'evaluation.')
  flags.DEFINE_integer('steps_per_execution', 1,
                       'Number of steps per training execution.')
  flags.DEFINE_string('train_file_pattern', None,
      'Glob for train data files (e.g., COCO train - minival set)')
  flags.DEFINE_string('val_file_pattern', None,
                      'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
  flags.DEFINE_string('val_json_file', None,
      'COCO validation JSON containing golden bounding boxes. If None, use the'
      'ground truth from the dataloader. Ignored if testdev_dir is not None.')
  flags.DEFINE_integer('num_examples_per_epoch', 120000,
                       'Number of examples in one epoch')
  flags.DEFINE_integer('num_epochs', None, 'Number of epochs for training')
  flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name.')
  flags.DEFINE_integer(
      'tf_random_seed', 111111,
      'Fixed random seed for deterministic execution across runs for debugging.'
  )
  flags.DEFINE_bool('profile', False, 'Enable profile mode')


def setup_model(model, config):
  """Build and compile model."""
  model.build((None, *config.image_size, 3))
  model.compile(
      steps_per_execution=config.steps_per_execution,  
      # Horovod: wrap optimizer in horovod's distributed optimizer for allreduce
      optimizer=hvd.DistributedOptimizer(train_lib.get_optimizer(config.as_dict())),
      loss={
          train_lib.BoxLoss.__name__:
              train_lib.BoxLoss(
                  config.delta, reduction=tf.keras.losses.Reduction.NONE),
          train_lib.BoxIouLoss.__name__:
              train_lib.BoxIouLoss(
                  config.iou_loss_type,
                  config.min_level,
                  config.max_level,
                  config.num_scales,
                  config.aspect_ratios,
                  config.anchor_scale,
                  config.image_size,
                  reduction=tf.keras.losses.Reduction.NONE),
          train_lib.FocalLoss.__name__:
              train_lib.FocalLoss(
                  config.alpha,
                  config.gamma,
                  label_smoothing=config.label_smoothing,
                  reduction=tf.keras.losses.Reduction.NONE),
          tf.keras.losses.SparseCategoricalCrossentropy.__name__:
              tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
      })
  return model


def get_dataset(is_training, config):
    file_pattern = (
        FLAGS.train_file_pattern
        if is_training else FLAGS.val_file_pattern)
    if not file_pattern:
      raise ValueError('No matching files.')

    return dataloader.InputReader(
        file_pattern,
        is_training=is_training,
        use_fake_data=False,
        max_instances_per_image=config.max_instances_per_image)(config.as_dict())


def init_experimental(config):
  """Serialize train config to model directory."""
  tf.io.gfile.makedirs(config.model_dir)
  config_file = os.path.join(config.model_dir, 'config.yaml')
  if not tf.io.gfile.exists(config_file):
    tf.io.gfile.GFile(config_file, 'w').write(str(config))

def main(_):
  # Parse and override hparams
  config = hparams_config.get_detection_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  if FLAGS.num_epochs:  
    config.num_epochs = FLAGS.num_epochs

  # Parse image size in case it is in string format.
  config.image_size = utils.parse_image_size(config.image_size)

  steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.batch_size // hvd.size()
  params = dict(
      profile=False,
      model_name=FLAGS.model_name,
      steps_per_execution=FLAGS.steps_per_execution,
      model_dir=FLAGS.model_dir,
      steps_per_epoch=steps_per_epoch,
      strategy='gpus',
      batch_size=FLAGS.batch_size,
      tf_random_seed=FLAGS.tf_random_seed,
      debug=False,
      val_json_file=FLAGS.val_json_file,
      eval_samples=FLAGS.eval_samples)
  config.override(params, True)
  
  # set mixed precision policy by keras api.
  precision = utils.get_precision(config.strategy, config.mixed_precision)
  policy = tf.keras.mixed_precision.Policy(precision)
  tf.keras.mixed_precision.set_global_policy(policy)
  
  if config.model_optimizations:
    tfmot.set_config(config.model_optimizations.as_dict())
    
  model = train_lib.EfficientDetNetTrain(config=config)
  model = setup_model(model, config)
  init_experimental(config)
  
  if FLAGS.pretrained_ckpt:
    ckpt_path = tf.train.latest_checkpoint(FLAGS.pretrained_ckpt)
    util_keras.restore_ckpt(
        model,
        ckpt_path,
        config.moving_average_decay,
        exclude_layers=['class_net'])
    
  val_dataset = get_dataset(False, config)
  params = config.as_dict()
  train_lib.update_learning_rate_schedule_parameters(params)
  callbacks = []
  
  # Horovod: broadcast initial variable states from rank 0 to all other processes.
  # This is necessary to ensure consistent initialization of all workers when
  # training is started with random weights or restored from a checkpoint.
  callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
  
  # Horovod: average metrics among workers at the end of every epoch.
  #
  # Note: This callback must be in the list before the ReduceLROnPlateau,
  # TensorBoard or other metrics-based callbacks.
  callbacks.append(hvd.callbacks.MetricAverageCallback())

  # Horovod: using `lr = x * hvd.size()` from the very beginning leads to worse final
  # accuracy. Scale the learning rate `lr = x` ---> `lr = x * hvd.size()` during
  # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
  callbacks.append(hvd.callbacks.LearningRateWarmupCallback(params['adjusted_learning_rate'], warmup_epochs=3, verbose=1))
  
  # Add logging callbacks
  callbacks.extend(train_lib.get_callbacks(config.as_dict(), val_dataset))
  
  class StepwiseDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr: float, warmup_epochs: int, 
                  first_lr_drop_step: int, second_lr_drop_step: int):
      self.initial_lr = initial_lr
      self.warmup_epochs = warmup_epochs
      self.first_lr_drop_step = first_lr_drop_step
      self.second_lr_drop_step = second_lr_drop_step

    def on_epoch_begin(self, epoch, logs=None):
      lr_schedule = [[1.0, self.warmup_epochs], [0.1, self.first_lr_drop_step],
                    [0.01, self.second_lr_drop_step]]
      for mult, start_global_step in lr_schedule:
        learning_rate = tf.where(epoch < start_global_step, learning_rate,
                                self.initial_lr * mult)  
      if epoch > self.warmup_epochs:
        tf.keras.backend.set_value(self.model.optimizer.lr, learning_rate)   
    
  class CosineDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr: float, warmup_epochs: float, 
                total_epochs: int):
      self.initial_lr = initial_lr
      self.warmup_epochs = warmup_epochs
      self.decay_steps = tf.cast(total_epochs - warmup_epochs, tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
      cosine_lr = 0.5 * self.initial_lr * (
          1 + tf.cos(math.pi * tf.cast(epoch, tf.float32) / self.decay_steps))
      if epoch > self.warmup_epochs:
        tf.keras.backend.set_value(self.model.optimizer.lr, cosine_lr)

  class PolynomialDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr: float, warmup_epochs: int, 
                 power: float, total_epochs: int):
      self.initial_lr = initial_lr
      self.warmup_epochs = warmup_epochs
      self.power = power
      self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
      polynomial_lr = self.initial_lr * tf.pow(
          1 - (tf.cast(epoch, dtype=tf.float32) / self.total_epochs), self.power)
      if epoch > self.warmup_epochs:
        tf.keras.backend.set_value(self.model.optimizer.lr, polynomial_lr)
  
  # Add learning rate decay callback:
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'stepwise':
    lr_decay_callback = StepwiseDecayCallback(params['adjusted_learning_rate'],
      params['lr_warmup_epoch'],
      params['first_lr_drop_step'],
      params['second_lr_drop_step'])
  elif lr_decay_method == 'cosine':
    lr_decay_callback = CosineDecayCallback(params['adjusted_learning_rate'],
      params['lr_warmup_epoch'],
      params['num_epochs'])
  elif lr_decay_method == 'polynomial':
    lr_decay_callback = PolynomialDecayCallback(params['adjusted_learning_rate'],
      params['lr_warmup_epoch'],
      params['poly_lr_power'],
      params['num_epochs'])
  callbacks.append(lr_decay_callback)

  # # Initialize the W&B run
  # run = wandb_init(project='tsigns-efficientdet', config=config.as_dict(),
  #                  job_type='train', entity='kris_rados')

  # # Define WandbCallback for experiment tracking
  # wandb_callback = WandbCallback(monitor='val_loss',
  #                                log_weights=True,
  #                                log_evaluation=True,
  #                                validation_steps=5)
  
  # callbacks.append(wandb_callback)

  # Horovod: adjust steps per epoch based on number of GPUs
  model.fit(
      get_dataset(True, config),
      epochs=config.num_epochs,
      steps_per_epoch=steps_per_epoch,
      callbacks=callbacks,
      validation_data=val_dataset,
      validation_steps=(FLAGS.eval_samples // FLAGS.batch_size // hvd.size()))

  # wandb.finish()

if __name__ == '__main__':
  define_flags()
  logging.set_verbosity(logging.ERROR)
  app.run(main)
