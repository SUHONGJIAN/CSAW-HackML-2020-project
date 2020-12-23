import keras
import h5py
import cv2
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

# clean_data_test_filename = "drive/MyDrive/MlForCyberProject/clean_test_data.h5"
# poisoned_data_sunglasses_filename = "drive/MyDrive/MlForCyberProject/sunglasses_poisoned_data.h5"
# sunglasses_bd_model_filename = "drive/MyDrive/MlForCyberProject/sunglasses_bd_net.h5"
clean_data_test_filename = "data/clean_test_data.h5"
poisoned_data_sunglasses_filename = "data/sunglasses_poisoned_data.h5"
sunglasses_bd_model_filename = "drive/MyDrive/MlForCyberProject/sunglasses_bd_net.h5"
entropy_sunglasses_filename = "entropy_clean_sunglasses.h5"

test_clean = DataLoader(clean_data_test_filename)
test_poisoned_sunglasses = DataLoader(poisoned_data_sunglasses_filename)

test_clean.load()
test_clean.preprocess()
test_poisoned_sunglasses.load()
test_poisoned_sunglasses.preprocess()

sunglasses_bd_model = keras.models.load_model(sunglasses_bd_model_filename)

entropy_clean_sunglasses_data = h5py.File(entropy_sunglasses_filename, "r")
entropy_clean_sunglasses = np.asarray(entropy_clean_sunglasses_data["data"])


STRIP_filter = STRIP(50, 1.5, 1, 0)

pred_poisoned = STRIP_filter.predict(entropy_clean_sunglasses, np.vstack((test_poisoned_sunglasses.x[:200],test_clean.x[:800])), sunglasses_bd_model)

print(np.sum(pred_poisoned))