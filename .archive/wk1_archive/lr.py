#!/usr/bin/env python3

# Logistic regression yay.

import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CSV_FILE="Social_Network_Ads.csv"

Normalizers = {
    'Age'   :           lambda a: min(float(a)/100.0, 1.0),
    'Gender':           lambda g: 0.0 if g == 'Male' else 1.0,
    'EstimatedSalary': lambda e: (float(e) - 15000)/(150000 - 15000),
    'Purchased':        lambda p: float(p)
}

def read_csv(csv_filename: str):
  """
  Returns:
     A tuple of (X, Y) where Y is a list of output values  and X is a list of input vectors (list of list)
  """
  with open(csv_filename) as csvfile:
    reader = csv.DictReader(csvfile)
    print(reader.fieldnames)
    x_list = []
    y_list = []
    for row in reader:
        # Access data by column name
        print(f"User ID: {row['User ID']}, Gender: {row['Gender']}, Age: {row['Age']}, EstimatedSalary: {row['EstimatedSalary']}, Purchased: {row['Purchased']}")
        x = [Normalizers[f](row[f]) for f in ('Gender', 'Age', 'EstimatedSalary', 'Purchased')]
        y = Normalizers['Purchased'](row['Purchased'])
        x_list.append(x)
        y_list.append(y)
    return (x_list, y_list)

  pass

def plot_dataset(dataset):
  X, Y = dataset
  gender   = [x[0] for x in X]
  age      = [x[1] for x in X]
  salary   = [x[2] for x in X]
  colors   = ['red' if y == 1.0 else 'blue' for y in Y]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(gender, age, salary, c=colors, alpha=0.6)

  ax.set_xlabel('Gender')
  ax.set_ylabel('Age')
  ax.set_zlabel('Estimated Salary')
  ax.set_title('Social Network Ads Dataset')

  legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='Purchased'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Not Purchased'),
  ]
  ax.legend(handles=legend_handles)
  plt.show()


def main():
  """ Main entry point of the program."""
  print("Main function")
  dataset = read_csv("Social_Network_Ads.csv")
  plot_dataset(dataset)


if __name__ == "__main__":
  print(__name__)
  main()

