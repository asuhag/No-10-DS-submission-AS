import argparse
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras as K 
import numpy as np
from sklearn import metrics
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import os
import pickle
import xgboost as xgb 

# Command line argument parser
parser = argparse.ArgumentParser(description='Run model with specified parameters')
parser.add_argument('--max_depth', type=int, default=5, help='max_depth for XGBoost')
parser.add_argument('--n_estimators', type=int, default=100, help='n_estimators for XGBoost')
parser.add_argument('--tree_method', type=str, default='gpu_hist', help='tree_method for XGBoost')
parser.add_argument('--predictor', type=str, default='gpu_predictor', help='predictor for XGBoost')
parser.add_argument('--grow_policy', type=str, default='lossguide', help='grow_policy for XGBoost')
args = parser.parse_args()

obj_model = hub.load('objectNet/objectNet/')
placesNet_model = K.models.load_model('placesNet/placesNet')
preprocessing_layer = K.Sequential(
    [
        K.layers.Rescaling(scale=1./127.5, offset=-1, name='Rescaling'),
        K.layers.CenterCrop(224, 224, name='CenterCrop'),
    ]
)

places_model = K.models.Sequential()
input_ = K.Input(shape=(224, 224, 3))
places_model.add(input_)
places_model.add(preprocessing_layer)
places_model.add(placesNet_model.layers[2])
places_model.add(placesNet_model.layers[3])

# run places inference
print("Running placesNet")
img = cv2.imread('IndoorData/val/bookstore/librairie2.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = tf.image.resize(img, size=[224, 224], method='nearest').numpy()
img_tensor = tf.expand_dims(img, 0)
y_place = places_model.predict(img_tensor)
labels = os.listdir('IndoorData/train/')
print("--" * 73)

# run object segment inference
print("Running object segmentation")
img_obj = tf.cast([img], tf.float32)
y_obj = obj_model.predict(img_obj)
seg_map = tf.argmax(y_obj, axis=3)
first_img_classes = np.unique(seg_map[0, :, :], return_counts=True)
area_by_class = list(zip(first_img_classes[0], 100*first_img_classes[1]/(224**2)))
X_train_obj = seg_map.numpy().reshape((-1, 224*224))
values, count = np.unique(X_train_obj, return_counts=True)

X_train_obj_top5 = []
X_train_obj_top5_area = []

for i in X_train_obj:
    values, count = np.unique(i, return_counts=True)
    X_train_obj_top5.append(values[np.argsort(-count)][:5])
    X_train_obj_top5_area.append((count[np.argsort(-count)]/(224*224))[:5])

def append(x):     
    if x.shape[0] < 5:
        return np.append(x, np.zeros(shape=(5-len(x))))
    else:
        return x
    
X_train_obj_top5 = np.array(list(map(append, X_train_obj_top5)))
X_train_obj_top5_area = np.array(list(map(append, X_train_obj_top5_area)))
print("--" * 73)

params = {
    'max_depth': args.max_depth,
    'n_estimators': args.n_estimators,
    'tree_method': args.tree_method,
    'predictor': args.predictor,
    'grow_policy': args.grow_policy
}

xgb_classifier = xgb.XGBClassifier(**params)
xgb_classifier.fit(X_train, y_train)
prediction_xgb_prob = xgb_classifier.predict_proba(X_test)

# top n metrics 
total = 1480
top_1 = [i for i in range(0, total) if top_5_classes[i][-1] == y_test[i]]
top_2 = [i for i in range(0, total) if y_test[i] in top_5_classes[i][-2:]]
top_3 = [i for i in range(0, total) if y_test[i] in top_5_classes[i][-3:]]
top_4 = [i for i in range(0, total) if y_test[i] in top_5_classes[i][-4:]]
top_5 = [i for i in range(0, total) if y_test[i] in top_5_classes[i][-5:]]

top_1_acc = len(top_1) / total
top_2_acc = len(top_2) / total
top_3_acc = len(top_3) / total
top_4_acc = len(top_4) / total
top_5_acc = len(top_5) / total

print(top_1_acc, top_2_acc, top_3_acc, top_4_acc, top_5_acc)
