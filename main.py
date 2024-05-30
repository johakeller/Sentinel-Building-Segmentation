from params import *
from data_acquisition import *
from dataset import *
from train_apply import *



def main():
    #run_acquisition()
    dataset = DataSplit(building_cover = 0.4)
    dataset.visualize_test()


if __name__ == "__main__":
    main()