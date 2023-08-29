import tensorflow_hub as hub
import numpy as np
tf.debugging.set_log_device_placement(False)

obj_model = hub.load('objectNet/')

object_tensor = []
for i, j in train_ds:
    features = obj_model.predict(i/255.0)
    img = i/255.0
    seg_map = tf.argmax(features, axis=3)
    with tf.device('/CPU:0'):
        object_tensor.extend(seg_map.numpy())
        
object_tensor = np.array(object_tensor)
np.save('training_features/training_features_objectnet.npy', object_tensor)

object_tensor = []
for i, j in test_ds:
    features = obj_model.predict(i/255.0)
    img = i/255.0
    seg_map = tf.argmax(features, axis=3)
    with tf.device('/CPU:0'):
        object_tensor.extend(seg_map.numpy())
        
object_tensor = np.array(object_tensor)
np.save('testing_features/testing_features_objectnet.npy', object_tensor)