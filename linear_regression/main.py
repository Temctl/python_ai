import pandas as pd # to read data files
from regression_model import Regression_model
import csv

def main():

    user_input = input("enter the csv file name: ")

    dataset = pd.read_csv(user_input)
    with open(user_input, newline = '') as f:
        reader = csv.reader(f)
        # gets the first row of the dataset st so it can know the x and y value name
        row1 = next(reader) 

    reg_model = Regression_model(dataset, row1[0], row1[1])

    reg_model.graph()

    reg_model.predict()


if __name__ == '__main__':
    main()
