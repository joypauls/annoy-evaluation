# This script converts the dataset from it's original form (u.data) to a standard CSV
import csv

original_file = "./data/u.data"
processed_file = "./data/movielens_ratings.csv"
processed_file_header = ["user_id", "item_id", "rating", "timestamp"]


def preprocess_movielens_100k():
  """Movielens 100k"""
  with open(processed_file, "w") as f1:
    writer = csv.writer(f1)
    writer.writerow(processed_file_header)

    with open(original_file, "r") as f2:
      file = csv.reader(f2, delimiter="\t")
      writer.writerows(file)

if __name__ == "__main__":
  preprocess_movielens_100k()
  print("Movielens 100k dataset preprocessed")
