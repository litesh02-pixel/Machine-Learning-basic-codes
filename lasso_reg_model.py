# -*- coding: utf-8 -*-
"""Lasso_Reg_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L7g6kUYtMrONekxcRkaSy4tpvFLGVYGk
"""

import numpy as np 

class LassoRegression():
  def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.lambda_parameter = lambda_parameter

  def fit(self, x, y):
    self.m, self.n = x.shape

    self.w = np.zeros(self.n)
    self.b = 0

    self.x = x
    self.y = y

    for i in range(self.no_of_iterations):
      self.update_weigths()
      
  def update_weigths(self):
    #linear equation of model

    y_prediction = self.predict(self.x)
    dw = np.zeros(self.n)

    for i in range (self.n):
      if self.w[i]>0:
        dw[i] = (-(2*(self.x[:,i]).dot(self.y - y_prediction)) + self.lambda_parameter)/self.m

      else:
        dw[i] = (-(2*(self.x[:,i]).dot(self.y - y_prediction)) - self.lambda_parameter)/self.m

    db = -2 * np.sum(self.y - y_prediction)/self.m
    #update the weight
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db
    
  def predict(self, x):
    return x.dot(self.w) + self.b

