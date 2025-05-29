"""
This file gets the unique classes from the train and validation datasets.
"""

import os, json
 
def get_categories(file_path):

    with open(file_path, "r") as f:
        json_loaded = json.load(f)

    for each_file_json in json_loaded:
        labels = each_file_json["labels"]
        for labs in labels:
            if labs["category"] not in categories:
                categories.append(labs["category"])

categories = []


file_path = "/Documents/personal/obd/data_bdd/bdd100k/labels/bdd100k_labels_images_val.json"
get_categories(file_path)
file_path = "/Documents/personal/obd/data_bdd/bdd100k/labels/bdd100k_labels_images_train.json"
get_categories(file_path)
print(list(set(categories)))