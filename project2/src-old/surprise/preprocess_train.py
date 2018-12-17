import csv


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset, sparse_matrix=True):
    """
        Load data in text format, one rating per line, and store in a
        sparse matrix or np.array 
    """

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    data = read_txt(path_dataset)[1:]
    data = [deal_line(line) for line in data]

    csvfile = open("./data/surprise_item_based_bsln_top50_clean.csv", 'w')

    fieldnames = ['User', 'Item', 'Rating']
    writer = csv.DictWriter(csvfile, delimiter=",",
                        fieldnames=fieldnames, lineterminator = '\n')
    writer.writeheader()

    for user, item, rating in data: 
        writer.writerow({'User': user - 1, 'Item': item - 1, 'Rating': rating})

    csvfile.close()


load_data("./submissions/surprise_item_based_bsln_top50.csv")
