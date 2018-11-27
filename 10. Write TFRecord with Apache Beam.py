from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import tempfile


import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


raw_data_feature_spec = dict(
    [('f1', tf.FixedLenFeature([], tf.float32)),
     ('y', tf.FixedLenFeature([], tf.float32))]
)

raw_data_metadata = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec(raw_data_feature_spec))


def main():
  with beam.Pipeline() as p:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      converter = tft.coders.CsvCoder(['f1','y'], raw_data_metadata.schema)
      coder = tft.coders.ExampleProtoCoder(raw_data_metadata.schema)
      raw_data = (
          p
          | beam.io.ReadFromText('./train.csv')
          | beam.Map(lambda line: line.replace(', ', ','))
          | beam.Map(converter.decode)
          | beam.io.WriteToTFRecord('./train_tx', coder)
      )

if __name__ == '__main__':
  main()
