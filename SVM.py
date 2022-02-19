import numpy as np

class SVM_classifier():


  # initiating the hyperparameters
  def __init__(self, learning_rate, no_of_iterations, lambda_parameter):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.lambda_parameter = lambda_parameter
  
  # fitting the dataset to SVM Classifier

  def fit(self, x, y):
    self.m, self.n = x.shape

    self.w = np.zeros(self.n)
    self.b = 0
    self.x = x
    self.y = y

    for i in range(self.no_of_iterations):
      self.update_weights()


  def update_weights(self):
      y_label = np.where(self.y<=0, -1, 1)

      for index, x_i in enumerate(self.x):
        condition = y_label[index]*(np.dot(x_i, self.w)-self.b)>=1

        if (condition==True):
          dw = 2 * self.lambda_parameter * self.w
          db = 0
        else:
          dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
          db = y_label[index] 


        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

      #predict the label for a given input value

  def predict(self,x):
      output = np.dot(x, self.w) - self.b
      predicted_labels = np.sign(output)
      y_hat = np.where(predicted_labels<=-1, 0, 1)
      return y_hat
