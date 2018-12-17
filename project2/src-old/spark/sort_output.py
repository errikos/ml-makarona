#!/usr/bin/env python3
import sys


def make_key(row):
    id_, _ = row.split(',')
    if id_ == 'Id':
        return 0, 0
    row, col = id_.split('_')
    row = int(row[1:])
    col = int(col[1:])
    return col, row


def main():
    if len(sys.argv) != 2:
        print('Please provide the CSV file to sort')
        sys.exit(1)
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        sorted_lines = sorted(lines, key=make_key)
    with open(sys.argv[1][:-4] + '_sorted.csv', 'w+') as f:
        f.write('Id,Prediction\n')
        f.writelines(sorted_lines)


if __name__ == '__main__':
    main()