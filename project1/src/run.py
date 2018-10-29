#!/usr/bin/env python3
import argparse

import main as m
import fitters

def main():
    parser = argparse.ArgumentParser(description='Higgs Boson ML Challenge')
    parser.add_argument('--data-path', type=str, default=m.DEFAULT_DATA_PATH,
                        help='Directory containing train.csv and test.csv.')
    args = parser.parse_args()

    fitter = fitters.RidgeFitter(
        std=False,
        lambda_=0.02,
        validation_param=0.8,
        degree=10,
        eliminate_minus_999=True)
    fitter.run(*m._load_data(args.data_path))  # load data and run

if __name__ == '__main__':
    main()