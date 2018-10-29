#!/usr/bin/env python3
import main as m
import fitters

def main():
    fitter = fitters.RidgeFitter(lambda_=0.02, validation_param=0.8)
    fitter.run(*m._load_data(m.DEFAULT_DATA_PATH))  # load data and run

if __name__ == '__main__':
    main()