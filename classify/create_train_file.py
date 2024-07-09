import glob, os, json
from pathlib import Path

def create_train_json(root, output_path, filter_class_names=None, add_bg_class=None):
    rgb_root = os.path.join(root, "rgb")
    tir_root = os.path.join(root, "tir")
    
    rgbs = glob.glob(os.path.join(rgb_root, "*/*.jpg"))
    
    data = []
    for i in rgbs:
        tir_path = i.replace("/rgb/", "/tir/")
        rgb_path = i
        if filter_class_names is None:
            data.append({
                "rgb": Path(rgb_path).relative_to(root).__str__(),
                "tir": Path(tir_path).relative_to(root).__str__(),
                "label": Path(rgb_path).parent.name
            })
        else:
            if Path(rgb_path).parent.name in filter_class_names:
                data.append({
                    "rgb": Path(rgb_path).relative_to(root).__str__(),
                    "tir": Path(tir_path).relative_to(root).__str__(),
                    "label": Path(rgb_path).parent.name
                })
            elif add_bg_class is not None:
                data.append({
                    "rgb": Path(rgb_path).relative_to(root).__str__(),
                    "tir": Path(tir_path).relative_to(root).__str__(),
                    "label": add_bg_class
                })
    
    with open(output_path, "w") as f:
        json.dump({
                "dataset_root": root,
                "data": data,
                "version": "v1.0"
            }, f)
        
if __name__ == "__main__":
    ## please split the dataset first
    
    ## 6 categories
    # create_train_json("/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/cropped_images/train",
    #                   "/root/workspace/data/Classify/data/train.json")
    # create_train_json("/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/cropped_images/val",
    #                   "/root/workspace/data/Classify/data/val.json")
    
    # 2 categories
    create_train_json("/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/cropped_images/train",
                      "/root/workspace/data/classify/data/train_c2.json", filter_class_names=["freight_car", "truck"])
    create_train_json("/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/cropped_images/val",
                      "/root/workspace/data/classify/data/val_c2.json", filter_class_names=["freight_car", "truck"])