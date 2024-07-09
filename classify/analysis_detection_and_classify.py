import pandas as pd
from pathlib import Path

path = "work_dir/ViT_B_Vehicle_c2/results.csv"

df = pd.read_csv(path)

def cal_old_res(row):
    row = Path(row).stem
    det_pred, label = list(map(int, row.split("_")[-2:]))
    if det_pred == label:
        return True
    else:
        return False
def cal_old_res2(row):
    row = Path(row).stem
    det_pred, label = list(map(int, row.split("_")[-2:]))
    if det_pred in [1, 4]:
        return 0 if det_pred == 1 else 1
    else:
        return -1
    # return det_pred


df["det_acc"] = df["file_info"].apply(cal_old_res)
df["det_pred"] = df["file_info"].apply(cal_old_res2)
df.to_csv("work_dir/ViT_B_Vehicle_c2/results2.csv")