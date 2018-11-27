import tensorflow as tf
import numpy as np

x_feature = tf.feature_column.numeric_column('f1')

def train_input_fn1():
    def parser(record):
        features = {'f1': tf.FixedLenFeature((), tf.float32),
                    'y': tf.FixedLenFeature((), tf.float32)}
        parsed = tf.parse_single_example(record, features)
        my_features = {'f1': parsed['f1']}
        return my_features, parsed['y']
    dataset = tf.data.TFRecordDataset(['train.tfrecords']).map(parser)
    dataset = dataset.shuffle(1000).repeat().batch(2)
    #return dataset.make_one_shot_iterator().get_next()
    return dataset

raw_data_feature_spec = dict(
    [('f1', tf.FixedLenFeature([], tf.float32)),
     ('y', tf.FixedLenFeature([], tf.float32))]
)

def train_input_fn():
    dataset = tf.contrib.data.make_batched_features_dataset(
            file_pattern=['train.tfrecords'],
            batch_size=2,
            features=raw_data_feature_spec,
            reader=tf.data.TFRecordDataset,
            shuffle=True)
    return dataset

samples = np.array([8., 9.])
def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(({"f1": samples}))
    dataset = dataset.repeat(1).batch(len(samples)) # batch cannot be none
    return dataset

regressor = tf.estimator.LinearRegressor(
    feature_columns=[x_feature],
    model_dir='./output'
)

regressor.train(input_fn=train_input_fn, max_steps=2500)
predictions = list(regressor.predict(input_fn=predict_input_fn))
for input, p in zip(samples, predictions):
    v  = p["predictions"][0]
    print("{} -> {:.4f}".format(input, v))