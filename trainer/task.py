#!/usr/bin/env python
import tensorflow as tf
import numpy as np

def serving_input_fn():
    inputs = {}
    inputs['x_input'] = tf.placeholder(shape=[None,28,28,1], dtype=tf.float32)
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def main():
    x_feature = tf.feature_column.numeric_column('f1')
    
    train_features = np.array([1., 2., 3., 4.])
    train_labels = np.array([1.5, 3.5, 5.5, 7.5])
    
    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(({"f1": train_features}, train_labels))
        dataset = dataset.shuffle(1000).repeat().batch(2)
        #return dataset.make_one_shot_iterator().get_next()
        return dataset

    test_features = np.array([5., 6., 7.])
    test_labels = np.array([9.5, 11.5, 13.5])

    def test_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(({"f1": test_features}, test_labels))
        dataset = dataset.repeat(1).batch(len(test_features)) # batch cannot be none
        return dataset
    
    run_config = tf.estimator.RunConfig(
        model_dir='output'
    )
    regressor = tf.estimator.LinearRegressor(
        feature_columns=[x_feature],
        config = run_config
    )
    
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=2500)
    eval_spec = tf.estimator.EvalSpec(test_input_fn, steps=1, exporters=[exporter])
    tf.estimator.train_and_evaluate(regressor, train_spec, eval_spec)

if __name__ == '__main__':
    tf.app.run()