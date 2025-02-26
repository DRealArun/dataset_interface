# Using `tensorflow/models`

This document describes how the tools and model definitions in the
[`tensorflow/models` repository](http://github.com/tensorflow/models) are used for training detection models for the
b-it-bots@Home team.

* [Generate protobuf label map file](#generate-protobuf-label-map-file)
* [Generate `TFRecord`'s from images and annotations](#generate-tfrecords-from-images-and-annotations)
* [Training process](#training-process)
* [Generate frozen graph from checkpoint](#generate-frozen-graph-from-checkpoint)

## Generate protobuf label map file

For b-it-bots@Home, a YAML file is used for annotating class names in the following format:

```YAML
1: cup
2: bowl
...
<class_id>: <class_name>
```

Note that `<class_id>` is the numeric output of the detection algorithm, and `<class_name>` is the descriptive string
which annotates the class. `<class_id>` starts at 1 to take into account the background (ID 0) class built into many
Single-Shot Multibox Detector (SSD) implementations. For example,
[`go_2019_classes.yml`](../configs/go_2019_classes.yml) contains the classes used in the
[RoboCup@Home 2019 German Open](https://www.robocupgermanopen.de/en/major/athome).

This YAML file needs to be converted to the `pbtxt` format expected by `tensorflow/models` scripts as label map file.
The format of `pbtxt` file is as follow:

```pbtxt
item {
  id: 1
  name: 'cup'
}

item {
  id: 2
  name: 'bowl'
}

...
item {
  id: <class_id>
  name: <class_name>
}
```

The script [`convert_yaml_to_pbtxt.py`](../scripts/convert_yaml_to_pbtxt.py) is provided to handle this conversion:

```sh
$ scripts/convert_yaml_to_pbtxt.py -h
usage: convert_yaml_to_pbtxt.py [-h] yaml_file pbtxt_file

Tool to convert YAML class annotations to pbtxt format

positional arguments:
  yaml_file   input YAML file containing class annotations
  pbtxt_file  output pbtxt file to write to

optional arguments:
  -h, --help  show this help message and exit
```

## Generate `TFRecord`'s from images and annotations

`TFRecord` is the data format expected by `tensorflow/models` scripts. It store images and annotations together as
binary files and can be broken into smaller chunks (i.e. shards) for easier management (see
[this tutorial](https://www.tensorflow.org/tutorials/load_data/tf_records) for more details).

The script [`create_robocup_tf_record.py`](../scripts/create_robocup_tf_record.py) handles the TFRecord generation.
The `annotation_file` argument expects a YAML file containing bounding box annotations for each image. This file should
have the same format that is generated using the [`generate_detection_data.py`](../scripts/generate_detection_data.py)
script. An example of the file is as follow:

```yaml
- image_name: /path/to/image.jpg
  objects:
  - class_id: 14
    xmax: 674
    xmin: 389
    ymax: 674
    ymin: 452
  - class_id: 6
    xmax: 434
    xmin: 280
    ymax: 512
    ymin: 266
...
```

Script arguments:

```sh
$ python3 scripts/create_robocup_tf_record.py -h
usage: create_robocup_tf_record.py [-h] [--image_dir IMAGE_DIR]
                                   [--num_shards NUM_SHARDS]
                                   annotation_file class_file output_file

Tool to generate TFRecord (optionally sharded) from images and bounding box
annotations.

positional arguments:
  annotation_file       YAML file containing image location and bounding box
                        annotations
  class_file            YAML file containing the mapping from class ID to
                        class name
  output_file           path where TFRecord's should be written to, e.g.
                        './robocup_train.record'

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR, -d IMAGE_DIR
                        if specified, will prepend to image paths in
                        annotation file
  --num_shards NUM_SHARDS, -n NUM_SHARDS
                        number of fragments to split the TFRecord into
```

## Training process

We use the `object_detection/model_main.py` script from `tensorflow/models` repo for training. The following sample
execution follows the `research/object_detection/g3doc/running_locally.md` markdown in the repo.

```sh
# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=data/ssd_resnet/pipeline.config
MODEL_DIR=/path/to/store/trained_model
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
```

Notes:

* In the above execution, `model_main.py` takes the following important arguments:
  * `--pipeline_config_path`: path to the `pipeline.config` file which configures the training. Sample config files
    can be found in the `tensorflow/models` repo under the `research/object_detection/samples/configs/` folder.
    Entries that may need to be modified in this config file include:
    * `pbtxt` label map, i.e. the one generated using the
      [class annotation conversion script](../scripts/convert_yaml_to_pbtxt.py)
    * Location of the TFRecord shards generated by the
      [TFRecord generation script](../scripts/create_robocup_tf_record.py)
  * `--model_dir`: path to the directory where the models checkpoints will be stored
  * `--num_train_steps`: number of training steps for the training process
  * `--sample_1_of_n_eval_examples`: TODO(minhnh)
* In order to use the `model_main.py` training script, follow installation instructions described in
  `research/object_detection/g3doc/installation.md` of the `tensorflow/models` repo. Most importantly, under the
  `research` folder of the repo:
  * compile Protobuf file: `protoc object_detection/protos/*.proto --python_out=.`
  * add libraries to `PYTHONPATH`: `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`

## Generate frozen graph from checkpoint

We use the `research/object_detection/export_inference_graph.py` script in the `tensorflow/models` repo to export
the frozen graph for inference. Similar to the `model_main.py` script, this also needs installation and exporting
of libraries as described above. A sample execution of this script as taken from the
`research/object_detection/g3doc/exporting_models.md` markdown in `tensorflow/models`:

```sh
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH={path to pipeline config file}
TRAINED_CKPT_PREFIX={path to model.ckpt}
EXPORT_DIR={path to folder that will be used for export}
python3 object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```

In the above execution, `export_inference_graph.py` takes the following important arguments:

* `--input_type`: model input type, should be `image_tensor` for object detection in images
* `--pipeline_config_path`: path to the `pipeline.config` file used for training and creating the checkpoints
* `--trained_checkpoint_prefix`: path to the checkpoints (e.g. `model.ckpt`) to be exported
* `--output_directory`: where the frozen graph will be stored
