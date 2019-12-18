import tensorflow as tf

# you can use this simple code for converting your models to tflite, tflite models you can use in android apps
converter = tf.lite.TFLiteConverter.from_keras_model_file("trained_model.h5")      # choose your model
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)        # save your converted model