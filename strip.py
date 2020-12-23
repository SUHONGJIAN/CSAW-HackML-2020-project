import keras
import h5py
import cv2
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt


class STRIP:
  def __init__(self, N, alpha, beta, gamma):
    self.N = N
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

  def blend_image(self, img1, img2):
    return cv2.addWeighted(img1,self.alpha,img2,self.beta,self.gamma).reshape(55,47,3)

  def calculate_entropy(self, target_img, model, input_img):
    blended_img = np.zeros((self.N,55,47,3))
    random_index = np.random.choice(np.arange(len(input_img)), self.N, replace=False)
    for i in range(self.N):
      blended_img[i] = self.blend_image(target_img, input_img[random_index[i]])
    pred_label = model.predict(blended_img)
    entropy = -np.nansum(pred_label*np.log2(pred_label))
    return entropy

  def generate_entropy_distribution(self, input_img, clean_img, model):
    l = len(input_img)
    entropy_distribution = np.zeros(l)
    for i in range(l):
      target_img = input_img[i]
      entropy_distribution[i] = self.calculate_entropy(target_img, model, clean_img)
    return entropy_distribution
  
  def predict(self, entropy_clean, input_img, model):
    mu, sigma = norm.fit(entropy_clean)
    threshold = norm.ppf(0.05, loc=mu, scale=sigma)
    l = len(input_img)
    pred_label = np.zeros(l)
    for i in range(l):
      target_entropy = self.calculate_entropy(input_img[i], model, input_img)
      if target_entropy < threshold:
        pred_label[i] = 1
    return pred_label

