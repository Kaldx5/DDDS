import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="../models/eyes_resnet18_128x128.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
