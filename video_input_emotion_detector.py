import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# For TensorFlow 1.x
# List devices
from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print("\nAvailable devices:")
for device in get_available_devices():
    print(f"- {device}")

# Configure GPU memory growth if GPU is available
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the config
sess = tf.Session(config=config)

# Check if GPU is available
if tf.test.is_built_with_cuda():
    print("\nGPU is available")
else:
    print("\nNo GPU found. Using CPU")