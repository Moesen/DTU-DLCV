import json
from projects.utils import get_project12_root
from tqdm import tqdm
import math

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
    val_ids = {x["id"] for x in validation_annot}
    test_ids = {x["id"] for x in test_annot}

    proposal_path = proot / "proposals"
    data_path = proot / "data/data_wastedetection"

    keys = ["train", "validation", "test"]
    out_dict = {k: {} for k in keys}

    for fn in tqdm(map(str, proposal_path.iterdir()), total=1500, disable=False):
        if ".gitkeep" in fn:
            continue
        img_id = fn.split("/")[-1].split("_")[0]
        with open(fn, "r") as f:
            file: list = json.load(f)

        if len(file) > 2000:
            del file[1000 : 1000 + (len(file) - 2000)]
        elif len(file) < 2000:
            missing_number_of_samples = 2000 - len(file)
            num_found_ious = 16 - missing_number_of_samples
            if num_found_ious > 0:
                file.extend(file[:missing_number_of_samples])
            
        for proposal in file:
            for i in range(2):
                if proposal[i] < 0:
                    proposal[i] = abs(proposal[i])

        where = [int(img_id) in x for x in [train_ids, val_ids, test_ids]]
        k = [x for i, x in enumerate(keys) if where[i]]

        if len(k) != 1:
            print(k, img_id)
        
        k = k[0]
        out_dict[k][str(img_id)] = file
    
    for fn, k in zip(["train_proposals.json", "val_proposals.json", "test_proposals.json"], keys):
        with open(data_path / fn, "w") as f:
            json.dump(out_dict[k], f, indent = 2)

