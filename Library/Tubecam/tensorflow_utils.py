"""
This Module contains utilities for tensorflow.
"""

import tensorflow as tf
import numpy as np

#Tensorflow version1.1
from tensorflow.tensorboard.backend.event_processing import event_accumulator

def load_session_with_graph(frozen_graph_path):
    """
    Loads a graph from the given .pb file and returns a session.

    Keyword arguments:
    frozen_graph_path -- The path to the frozen graph *.pb.
    """

    tf.reset_default_graph()
    with tf.gfile.FastGFile(frozen_graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    return tf.Session()

def predict(session, images):
    """
    Returns predictions of the given images.
    Only works with graphs having a softmax layer named 'final_result:0'.
    Performance could be better,
    if there was a input layer in inception v3 accepting batches of images.

    Keyword arguments:
    Images -- images numpy-array of shape (?,?,?,3).
    """
    predictions = []
    for image in images:
        softmax_tensor = session.graph.get_tensor_by_name('final_result:0')
        prediction = session.run(softmax_tensor, {'DecodeJpeg:0': image})
        prediction = np.squeeze(prediction)
        predictions.append(prediction)

    return predictions

def extract_features(session, images):
    """
    Returns a 2048-Dimensional feature-vector from the "pool_3" Layer of the given image.
    Only works with graphs having a feature layer named 'pool_3:0'.

    Keyword arguments:
    Image -- image numpy-array of shape (?,?,?,3)
    """
    representation_tensor = session.graph.get_tensor_by_name('pool_3:0')
    representations = np.zeros((len(images), 2048), dtype='float32')
    for i in range(len(images)):
        reps = session.run(representation_tensor, {'DecodeJpeg:0': images[i]})
        representations[i] = np.squeeze(reps)

    return representations

def load_event_scalars(path, *scalar_names):
    """
    Loads scalar values from eventfiles.

    Keyword arguments:
    path -- path to the event file
    *event_names -- scalar names
    """
    accumulator = event_accumulator.EventAccumulator(path)
    accumulator.Reload()

    scalar_values = []

    for scalar_name in scalar_names:
        scalar_values.append(accumulator.Scalars(scalar_name))

    return scalar_values

def get_confusion_matrix(predictions, truth_labels, pred_labels, corresponding_data=None):
    """
    Produces a confusion matrix from the given parameter.

    Keyword arguments:
    predictions -- Predictions array produced by a softmax layer.
    truth_labels -- Ground truth labels corresponding to the given predictions.
    pred_labels -- Prediction labels, which contain the labels
                    in which order the prediction array calculated
                    the probabilites.
    corresponding_data -- Optional data if you want to fill the data_matrix.
    """

    confusion_matrix = np.zeros((len(pred_labels), len(pred_labels)), dtype=np.int8)
    if corresponding_data and len(corresponding_data) == len(predictions):
        data_matrix = [[[] for x in range(len(pred_labels))] for y in range(len(pred_labels))]

    for index, truth_label in enumerate(truth_labels):
        prediction = predictions[index]
        column = prediction.argmax()
        row = pred_labels.index(truth_label)
        confusion_matrix[row][column] += 1
        if corresponding_data and len(corresponding_data) == len(predictions):
            data_matrix[row][column].append(corresponding_data[index])

    return confusion_matrix, data_matrix
