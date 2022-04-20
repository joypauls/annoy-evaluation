# This script converts the dataset from it's original form (u.data) to a standard CSV
import csv

original_file = "./data/u.data"
processed_file = "./data/movielens_users.csv"
processed_file_header = ["user_id", "item_id", "rating", "timestamp"]


with open('combined_file.csv', 'w', newline='') as outcsv:
  writer = csv.writer(outcsv)
  writer.writerow(processed_file_header)

  with open(original_file, "rb") as f:
    file = csv.reader(f, delimiter="\t")
    writer.writerows(f)

print("Movielens 100k dataset processed")

