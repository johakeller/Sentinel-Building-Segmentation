from params import *
from data_acquisition import *
from dataset import *
from train_apply import *



def main():
    #run_acquisition()
    dataset = DataSplit(city='Aarhus')
    dataset.visualize_test()


if __name__ == "__main__":
    main()