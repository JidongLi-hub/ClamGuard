import json
import os
from tqdm import tqdm

def add_label_into_json(file_path, json_path, label):
    with open(json_path, "r") as jf:
        cont = jf.read()
        if cont == "":
            label_dict = {}
        else:
            label_dict = json.loads(cont)

    for file in tqdm(os.listdir(file_path)):
        label_dict.update({file: label})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    file_path = "./DateSet/Malimg/"
    json_path = "DateSet/labels.json"
    label = 1
    add_label_into_json(file_path, json_path, label)


