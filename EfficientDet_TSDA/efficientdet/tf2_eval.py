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
"""Eval libraries."""
import os
import sys
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import dataloader
import hparams_config
import utils
from tf2 import anchors
from tf2 import efficientdet_keras
from tf2 import util_keras

# Cloud TPU Cluster Resolvers
flags.DEFINE_string('tpu', None, 'The Cloud TPU name.')
flags.DEFINE_string('gcp_project', None, 'Project name.')
flags.DEFINE_string('tpu_zone', None, 'GCE zone name.')

flags.DEFINE_integer('eval_samples', 6624, 'Number of eval samples.')
flags.DEFINE_string('val_file_pattern', None,
                    'Glob for eval tfrecords, e.g. coco/val-*.tfrecord.')
flags.DEFINE_string('model_name', 'efficientdet-d2', 'Model name to use.')
flags.DEFINE_string('model_dir', None, 
                    'Location of the checkpoint to run.')
flags.DEFINE_integer('batch_size', 4, 'GLobal batch size.')
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file')
flags.DEFINE_string('eval_dir', None,
                    'Location of output directory for evaluation')
flags.DEFINE_string('strategy', 'gpu', 'Eval strategy')
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def main(_):
  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  config.strategy = FLAGS.strategy
  config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
  config.drop_remainder = False  # eval all examples w/o drop.
  config.image_size = utils.parse_image_size(config['image_size'])

  if config.strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif config.strategy == 'gpus':
    ds_strategy = tf.distribute.MirroredStrategy()
    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
      ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  with ds_strategy.scope():
    # Network
    model = efficientdet_keras.EfficientDetModel(config=config)
    util_keras.restore_ckpt(model,
                            tf.train.latest_checkpoint(FLAGS.model_dir),
                            config.moving_average_decay,
                            skip_mismatch=False)
    
    @tf.function
    def model_fn(images, labels):
      outputs = model(
        images, image_scales=labels['image_scales'], training=False, 
        pre_mode='infer', post_mode='global')
  
      # Convert to COCO bbox
      boxes = tf.expand_dims(outputs[0], axis=-1)
      xmin = boxes[:, :, 1]
      ymin = boxes[:, :, 0]
      w = boxes[:, :, 3] - boxes[:, :, 1]
      h = boxes[:, :, 2] - boxes[:, :, 0]
      boxes = tf.concat([xmin, ymin, w, h], axis=-1) 

      scores = outputs[1]
      classes = outputs[2]
      damages = outputs[3]
      detections = tf.concat(
        [boxes, tf.expand_dims(scores, -1), damages, tf.expand_dims(classes, -1)], axis=-1)
      return detections

    # dataset
    batch_size = FLAGS.batch_size   # global batch size.
    ds = dataloader.InputReader(
        FLAGS.val_file_pattern,
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)(
            config, batch_size=batch_size)
    if FLAGS.eval_samples:
      ds = ds.take((FLAGS.eval_samples + batch_size - 1) // batch_size)
    ds = ds_strategy.experimental_distribute_dataset(ds)

    # evaluate all images.
    all_dets = None
    eval_samples = FLAGS.eval_samples or 5000
    pbar = tf.keras.utils.Progbar((eval_samples + batch_size - 1) // batch_size)
    for i, (images, labels) in enumerate(ds):
      detections = ds_strategy.run(model_fn, [images, labels])
      ids = tf.tile(
      tf.expand_dims(labels["source_ids"], -1), [1, tf.shape(detections)[1]])
      detections = tf.concat((tf.expand_dims(ids, -1), detections), axis=-1)
      detections = tf.reshape(detections, (-1, tf.shape(detections)[-1]))
      
      if all_dets is None:
        all_dets = detections.numpy()
      else:
        all_dets = np.concatenate((all_dets, detections.numpy()), axis=0)
      pbar.update(i)
      
    if not FLAGS.eval_dir:
      raise ValueError('eval_dir is not specified.')
    else:
      np.save(os.path.join(FLAGS.eval_dir, 'detections_testdev_results.npy'), all_dets)


if __name__ == '__main__':
  flags.mark_flag_as_required('val_file_pattern')
  flags.mark_flag_as_required('model_dir')
  logging.set_verbosity(logging.ERROR)
  # tf.debugging.enable_check_numerics()
  app.run(main)
