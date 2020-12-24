import keras
import h5py
import numpy as np
import os.path
import tensorflow as tf
from strip import STRIP

"""
DataLoader class is inspired by https://github.com/csaw-hackml/CSAW-HackML-2020/blob/master/eval.py
"""
class DataLoader:
  def __init__(self, file_path):
    self.file_path = file_path
    self.load()
    self.preprocess()

  def load(self):
    data = h5py.File(self.file_path, "r")
    x_data = np.asarray(data["data"])
    self.x = x_data.transpose((0,2,3,1))
    self.y = np.asarray(data["label"])
  
  def preprocess(self):
    self.x = np.asarray(self.x/255, np.float64)

def num_to_net(num):
    numbers = {
      "1": {"model": "models/sunglasses_bd_net.h5", "entropy": "entropy/entropy_clean_sunglasses.h5", "fine_pruned_model": "finePruning/sunglasses_repaired_net.h5"},
      "2": {"model": "models/anonymous_1_bd_net.h5", "entropy": "entropy/entropy_clean_anonymous1.h5", "fine_pruned_model": "finePruning/anonymous_1_repaired_net.h5"},
      "3": {"model": "models/anonymous_2_bd_net.h5", "entropy": "entropy/entropy_clean_anonymous2.h5", "fine_pruned_model": "finePruning/anonymous_2_repaired_net.h5"},
      "4": {"model": "models/multi_trigger_multi_target_bd_net.h5", "entropy": "entropy/entropy_clean_multi.h5", "fine_pruned_model": "finePruning/multi_trigger_multi_target_repaired_net.h5"},
    }
    return numbers.get(num, {"model": "models/sunglasses_bd_net.h5", "entropy": "entropy/entropy_clean_sunglasses.h5", "fine_pruned_model": "finePruning/sunglasses_repaired_net.h5"})

def num_to_mode(num):
    numbers = {
      "1": "seperate",
      "2": "mixed",
    }
    return numbers.get(num, "seperate")

def main():
    # Select a test BadNet
    print('\033[0;32m' + "Select a test net: \n1. sunglasses_bd_net \n2. anonymous_1_bd_net \n3. anonymous_2_bd_net \n4. multi-trigger_multi_target_bd_net")
    net = num_to_net(input())
    model = net["model"]
    print("{0} selected!\n".format(model))
    bd_model = keras.models.load_model(model)

    # Select the test mode
    print("Select the test mode: \n1. Seperate data (clean / poisoned) \n2. Mixed data")
    test_mode = num_to_mode(input())
    if test_mode == "seperate":
        print("Seperate mode selected!\n")
        # Set the test poisoned data
        print("Please put the poisoned data under data/ and name the file poisoned_data.h5 (i.e. data/poisoned_data.h5) \nThen click enter.")
        input()
        poisoned_data_test_filename = "data/poisoned_data.h5"
        while not os.path.isfile(poisoned_data_test_filename):
            print('\033[91m' + "Error: data/poisoned_data.h5 does not exist. please try again.\nThen click enter.")
            input()
        # Set the test clean data
        print('\033[0;32m' + "Please put the clean data under data/ and name the file clean_data.h5 (i.e. data/clean_data.h5) \nThen click enter.")
        input()
        clean_data_test_filename = "data/clean_data.h5"
        while not os.path.isfile(poisoned_data_test_filename):
            print('\033[91m' + "Error: data/clean_data.h5 does not exist. please try again.\nThen click enter.")
            input()
        test_clean = DataLoader(clean_data_test_filename)
        test_poisoned = DataLoader(poisoned_data_test_filename)

        # Step one: STRIP
        print('\033[0;32m' + "STRIP: running......(maybe several minutes, even hours)")
        entropy_filename = net["entropy"]
        entropy_clean_data = h5py.File(entropy_filename, "r")
        entropy_clean = np.asarray(entropy_clean_data["data"])
        STRIP_filter = STRIP(50, 1.5, 1, 0)
        pred_poisoned = STRIP_filter.predict(entropy_clean, test_poisoned.x, bd_model)
        succ_att_rate = np.mean(np.equal(pred_poisoned, test_poisoned.y)) * 100
        print('\033[95m' + "Success attack rate after STRIP: {0}%".format(succ_att_rate))
        print('\033[0;32m' + "Please wait for accuracy...")
        pred_clean = STRIP_filter.predict(entropy_clean, test_clean.x, bd_model)
        accu = np.mean(np.equal(pred_clean, test_clean.y)) * 100
        print("Accuracy after STRIP: {0}%".format(accu))

        # Step two: Fine-pruning
        print('\033[0;32m' + "Fine-pruning: running......")
        I_poisoned_remaining = np.argwhere(pred_poisoned == test_poisoned.y)
        if I_poisoned_remaining.size != 0 :
            test_poisoned_remaining = test_poisoned.x[I_poisoned_remaining]
            I_clean_remaining = np.argwhere(pred_clean != test_clean.y)
            test_clean_remaining = test_clean.x[I_clean_remaining]
            fine_pruned_model = keras.models.load_model(net["fine_pruned_model"])
            pred_poisoned_remaining = fine_pruned_model.predict(test_poisoned_remaining)
            succ_att_rate = (np.sum(pred_poisoned_remaining == test_poisoned.y[I_poisoned_remaining]) / test_poisoned.y.shape[0]) * 100
            print('\033[95m' + "Success attack rate after fine-pruning: {0}%".format(succ_att_rate))
            print('\033[0;32m' + "Please wait for accuracy...")
            pred_clean_remaining = fine_pruned_model.predict(test_clean_remaining)
            accu = (test_clean.y.shape[0] - np.sum(pred_clean_remaining != test_clean.y[I_clean_remaining]) / test_clean.y.shape[0]) * 100
            print("Accuracy after fine-pruning: {0}%".format(accu))
            for i in I_poisoned_remaining:
              pred_poisoned[i] = pred_poisoned_remaining[i]
            for i in I_clean_remaining:
              pred_clean[i] = pred_clean_remaining[i]
        # Output the final prediction
        print("Final prediction of poisoned data: {0}".format(pred_poisoned))
        print("Final prediction of clean data: {0}".format(pred_poisoned))
    else:
        # Set the test mixed data
        print("Please put the mixed data under data/ and name the file mixed_data.h5 (i.e. data/mixed_data.h5) \nThen click enter.")
        input()
        mixed_data_test_filename = "data/mixex_data.h5"
        while not os.path.isfile(mixed_data_test_filename):
            print('\033[91m' + "Error: data/mixed_data.h5 does not exist. please try again.\nThen click enter.")
            input()
        test_mixed = DataLoader(mixed_data_test_filename)

        # Step one: STRIP
        print('\033[0;32m' + "STRIP: running......(maybe several minutes, even hours)")
        entropy_filename = net["entropy"]
        entropy_clean_data = h5py.File(entropy_filename, "r")
        entropy_clean = np.asarray(entropy_clean_data["data"])
        STRIP_filter = STRIP(50, 1.5, 1, 0)
        pred_mixed = STRIP_filter.predict(entropy_clean, test_mixed.x, bd_model)

        # Step two: Fine-pruning
        print('\033[0;32m' + "Fine-pruning: running......")
        I_mixed_remaining = np.argwhere(pred_mixed == test_mixed.y)
        if I_mixed_remaining.size != 0:
            test_mixed_remaining = test_mixed.x[I_mixed_remaining]
            fine_pruned_model = keras.models.load_model(net["fine_pruned_model"])
            pred_mixed_remaining = fine_pruned_model.predict(test_mixed_remaining)
            for i in I_mixed_remaining:
              pred_mixed[i] = pred_mixed_remaining[i]

        # Output the final prediction
        print("Final prediction of poisoned data: {0}".format(pred_mixed))


if __name__ == "__main__":
    main()
