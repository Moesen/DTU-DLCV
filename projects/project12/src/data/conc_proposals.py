import json
from projects.utils import get_project12_root
import re

if __name__ == "__main__":
    proot = get_project12_root()

    train_annot = json.load(open(proot / "data/data_wastedetection/train_data.json"))[
        "images"
    ]

    validation_annot = json.load(
        open(proot / "data/data_wastedetection/validation_data.json")
    )["images"]

    test_annot = json.load(open(proot / "data/data_wastedetection/test_data.json"))[
        "images"
    ]

    train_ids = {x["id"] for x in train_annot}
    val_ids = {x["id"] for x in train_annot}
    test_ids = {x["id"] for x in train_annot}

    proposal_path = proot / "proposals"
    data_path = proot / "data"

    train_dict = {}
    val_dict = {}
    test_dict = {}
    pattern = re.compile(r"(\d+)\_.*\.json")
    for file in map(str, proposal_path.iterdir()):
        pats = pattern.findall(file)
        if len(pats) == 0:
            continue
        if len(pats) > 1:
            print(file)

        img_id = int(pats[0])
        if img_id in train_ids:
            train_dict[str(img_id)] = json.load(open(file))  
        elif img_id in val_ids:
            val_dict[str(img_id)] = json.load(open(file))
        elif img_id in test_ids:
            test_dict[str(img_id)] = json.load(open(file))
        else:
            print(file)
            print("id not found in any of the sets??")

    with open(data_path / "train_proposals.json", "w") as f:
        json.dump(train_dict, f, indent = 2)
    with open(data_path / "val_proposals.json", "w") as f:
        json.dump(val_dict, f, indent = 2)
    with open(data_path / "test_proposals.json", "w") as f:
        json.dump(test_dict, f, indent = 2)
