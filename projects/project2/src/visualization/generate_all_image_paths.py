
import glob 

from projects.utils import get_project2_root

classes = ["hair","bald"]

PROJECT_ROOT = get_project2_root()
path1 =  PROJECT_ROOT / "data" / classes[0]
path2 =  PROJECT_ROOT / "data" / classes[1]



img_paths1 = glob.glob((path1 / "*.png").as_posix())
img_paths2 = glob.glob((path2 / "*.png").as_posix())

with open(path1.as_posix() +'.txt', 'w') as f:
    f.write('\n'.join(img_paths1))

with open(path2.as_posix() +'.txt', 'w') as f:
    f.write('\n'.join(img_paths2))
