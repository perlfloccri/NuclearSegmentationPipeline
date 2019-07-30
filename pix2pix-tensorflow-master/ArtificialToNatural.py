from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

def convertToNatural(image=0,output_dir=None,checkpoint=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if checkpoint is None:
        raise Exception("checkpoint required for test mode")
        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        scale_size = CROP_SIZE
        flip = False
    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)
    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        e=1
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
        max_steps = 2**32
        if max_epochs is not None:
            max_steps = examples.steps_per_epoch * max_epochs
        if max_steps is not None:
            max_steps = max_steps

        # testing
        # at most, process the test data once
        start = time.time()
        max_steps = min(examples.steps_per_epoch, max_steps)
        for step in range(max_steps):
            results = sess.run(display_fetches)
            filesets = save_images(results)
            for i, f in enumerate(filesets):
                print("evaluated image", f["name"])
            index_path = append_index(filesets)
        print("wrote index at", index_path)
        print("rate", (time.time() - start) / max_steps)
