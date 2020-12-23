import keras
import h5py
import numpy as np
import os.path
from strip import STRIP

"""
DataLoader class is inspired by https://github.com/csaw-hackml/CSAW-HackML-2020/blob/master/eval.py
"""
class DataLoader:
  def __init__(self, file_path):
    self.file_path = file_path

  def load(self):
    data = h5py.File(self.file_path, "r")
    x_data = np.asarray(data["data"])
    self.x = x_data.transpose((0,2,3,1))
    self.y = np.asarray(data["label"])
  
  def preprocess(self):
    self.x = np.asarray(self.x/255, np.float64)

def num_to_net(num):
    numbers = {
        1: {"model": "models/sunglasses_bd_net.h5", "entropy": "entropy/entropy_clean_sunglasses.h5"},
        2: {"model": "models/anonymous_1_bd_net.h5", "entropy": "entropy/entropy_clean_anonymous1"},
        3: {"model": "models/anonymous_2_bd_net.h5", "entropy": "entropy/entropy_clean_anonymous1"},
        4: {"model": "models/multi_trigger_multi_target_bd_net.h5", "entropy": "entropy/entropy_clean_multi"},
    }
    return numbers.get(num, {"model": "models/sunglasses_bd_net", "entropy": "entropy/entropy_clean_sunglasses.h5"})


def main():
    # Choose a test BadNet
    print("Select the net: \n1. sunglasses_bd_net \n2. anonymous_1_bd_net \n3. anonymous_2_bd_net \n4. multi-trigger_multi_target_bd_net")
    net = num_to_net(input())
    model = net["model"]
    bd_model = keras.models.load_model(model)

    # Give the test poisoned data
    print("Please put the poisoned data under eval/ and name the file test.h5 (i.e. eval/poisoned.h5) \nThen click enter.")
    clean_data_test_filename = "data/clean_test_data.h5"
    poisoned_data_test_filename = "eval/poisoned.h5"
    while not os.path.isfile(poisoned_data_test_filename):
        print("Error: eval/poisoned.h5 does not exist. please try again.")
    test_clean = DataLoader(clean_data_test_filename)
    test_poisoned = DataLoader(poisoned_data_test_filename)

    # Step one: STRIP
    print("STRIP: running......")
    entropy_filename = net["entropy"]
    entropy_clean_data = h5py.File(entropy_filename, "r")
    entropy_clean = np.asarray(entropy_clean_data["data"])
    STRIP_filter = STRIP(50, 1.5, 1, 0)
    pred_poisoned = STRIP_filter.predict(entropy_clean, test_poisoned.x, bd_model)
    succ_att_rate = np.mean(np.equal(pred_poisoned, test_poisoned.y)) * 100
    print("Success attack rate after STRIP: {0}%".format(succ_att_rate))
    pred_clean = STRIP_filter.predict(entropy_clean, test_clean.x, bd_model)
    accu = np.mean(np.equal(pred_clean, test_clean.y)) * 100
    print("Accuracy after STRIP: {0}%".format(accu))
    # Step two: Fine-pruning
    print("Fine-pruning: running......")
    


if __name__ == "__main__":
    main()