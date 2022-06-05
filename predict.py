import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import onnxruntime as ort
import time

def prepare_test_data():
    # Load mnist
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return test_images

def predict_default_model(input):
    # Load and predict default model
    start = time.time()
    model = keras.models.load_model('./weights/mnist')
    predictions_default = model.predict(input)
    return predictions_default, time.time() - start

def predict_tflite_model(input):
    # Load and predict tflite model
    start = time.time()
    interpreter = tf.lite.Interpreter(model_path='./weights/mnist_model.tflite')
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, input)
    interpreter.invoke()
    predictions_tflite = interpreter.get_tensor(output_index)
    return predictions_tflite, time.time() - start

def predict_tflite_fp16_model(input):
    # Load and predict tflite_fp16 model
    start = time.time()
    interpreter_fp16 = tf.lite.Interpreter(model_path='./weights/mnist_model_quant_f16.tflite')
    interpreter_fp16.allocate_tensors()

    input_index = interpreter_fp16.get_input_details()[0]["index"]
    output_index = interpreter_fp16.get_output_details()[0]["index"]

    interpreter_fp16.set_tensor(input_index, input)
    interpreter_fp16.invoke()
    predictions_16 = interpreter_fp16.get_tensor(output_index)
    return predictions_16, time.time() - start

def predict_onnx_model(input):
    # Load and predict onnx model
    start = time.time()
    sess = ort.InferenceSession("./weights/mnist.onnx") # providers=["CUDAExecutionProvider"])
    ort_preds = sess.run(None, {"input_1": input})
    predictions_onnx = ort_preds[0]
    return predictions_onnx, time.time() - start

if __name__ == "__main__":
    input = prepare_test_data()
    input = np.expand_dims(input[0], axis=0).astype(np.float32)

    print('Result default model: {}'.format(predict_default_model(input)[0]))
    print('Inference time: {}'.format(predict_default_model(input)[1]))

    print('Result tflite model: {}'.format(predict_tflite_model(input)[0]))
    print('Inference time: {}'.format(predict_tflite_model(input)[1]))

    print('Result tflite_16 model: {}'.format(predict_tflite_fp16_model(input)[0]))
    print('Inference time: {}'.format(predict_tflite_fp16_model(input)[1]))

    print('Result onnx model: {}'.format(predict_onnx_model(input)[0]))
    print('Inference time: {}'.format(predict_onnx_model(input)[1]))

