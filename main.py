from params import *
from data_acquisition import *
from dataset import *
from train_apply import *



def main():
    #run_acquisition()
    #train_apply('UNet')
    train_apply('ConvNet')

if __name__ == "__main__":
    main()