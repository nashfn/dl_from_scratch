#!/usr/bin/env python3

# Logistic regression yay.

import csv

CSV_FILE="Social_Network_Ads.csv"

Normalizers = {
    'age'   :           lambda a: min(a/100.0, 1.0),
    'gender':           lambda g: 0.0 if g == 'Make' else 1.0,
    'estimated_salary': lambda e: (float(e) - 15000)/(150000 - 15000),
    'purchased':        lambda p: float(p)
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
        print(f"User ID: {row['User ID']}, Gender: {row['Gender']}, EstimatedSalary: {row['EstimatedSalary']}, Purchased: {row['Purchased']}")
        x = [Normalizers[f](row[f]) for f in ('Gender', 'EstimatedSalary', 'Purchased')]
        y = Normalizers['purchased'](row['purchased'])
        x_list.append(x)
        y_list.append(y)
    return (x_list, y_list)

  pass

def main():
  """ Main entry point of the program."""
  print("Main function")
  dataset = read_csv("Social_Network_Ads.csv")
  print(dataset)


if __name__ == "__main__":
  print(__name__)
  main()

