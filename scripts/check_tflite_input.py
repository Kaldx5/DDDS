import tensorflow as tf

# Load the model from your local models folder
interpreter = tf.lite.Interpreter(model_path="../models/cnn_model.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
