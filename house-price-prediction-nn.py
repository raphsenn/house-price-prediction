import numpy as np
import csv


"""
price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,
hotwaterheating,airconditioning,parking,prefarea,furnishingstatus
"""


def tokenize(text: str) -> list[int]
    pass


def read_labled_data(file: str):
    with open(file, 'r') as file:
        csvreader = rsv.reader(file)
        # Skip first line.
        next(csvreader)



class NerualNetwork:
    def __init__(self, features: int) -> None:
        self.weights = np.zeros(features)

    



