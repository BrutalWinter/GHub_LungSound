import tensorflow as tf


G = tf.config.experimental.list_physical_devices("GPU")
C = tf.config.experimental.list_physical_devices("CPU")
print("GPU: ", G)
print("CPU: ", C)
if G:
    with tf.device("/gpu:0"):
        print("GPU is currently printing:{} ".format(G))
else:
    print("GPU: not found, CPU:", C)