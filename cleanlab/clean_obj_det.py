import copy
import json
import os
import pickle

import numpy as np
import tqdm
from pathlib import Path


# label_coco = "merged_coco_new.json"


# {'ori_shape2': (512, 640), 'ori_shape': (512, 640), 'pad_shape': (1024, 1024), 'batch_input_shape': (1024, 1024), 'img_shape2': (1024, 1024), 'scale_factor': (1.6, 1.599609375), 'img_id': 0, 'img_path2': '/root/workspace/data/GAIIC2024/train_more/tir/test-04575.jpg', 'img_path': '/root/workspace/data/GAIIC2024/train_more/rgb/test-04575.jpg', 'img_shape': (1024, 1024), 'pred_instances': {'scores': tensor([0.8758, 0.8580, 0.8526, 0.8490, 0.8428, 0.7239, 0.7143, 0.6686, 0.6344,
#         0.3733, 0.2789, 0.2543, 0.1788, 0.1307, 0.1221, 0.1189, 0.1186, 0.1081,
#         0.1043, 0.1032, 0.1005, 0.0957, 0.0940, 0.0911, 0.0896, 0.0861, 0.0860,
#         0.0760, 0.0733, 0.0733, 0.0711, 0.0611, 0.0579, 0.0579, 0.0567, 0.0565,
#         0.0526, 0.0525, 0.0516, 0.0506, 0.0498, 0.0493, 0.0487, 0.0480, 0.0449,
#         0.0415, 0.0401, 0.0386, 0.0386, 0.0386, 0.0376, 0.0373, 0.0368, 0.0364,
#         0.0358, 0.0356, 0.0347, 0.0345, 0.0344, 0.0336, 0.0332, 0.0320, 0.0315,
#         0.0313, 0.0309, 0.0308, 0.0302, 0.0300, 0.0299, 0.0294, 0.0293, 0.0291,
#         0.0283, 0.0281, 0.0281, 0.0280, 0.0279, 0.0277, 0.0276, 0.0275, 0.0267,
#         0.0260, 0.0254, 0.0246, 0.0240, 0.0232, 0.0230, 0.0229, 0.0226, 0.0225,
#         0.0224, 0.0222, 0.0220, 0.0216, 0.0215, 0.0213, 0.0212, 0.0206, 0.0206,
#         0.0204, 0.0203, 0.0203, 0.0202, 0.0202, 0.0196, 0.0195, 0.0194, 0.0192,
#         0.0190, 0.0190, 0.0189, 0.0185, 0.0184, 0.0183, 0.0181, 0.0181, 0.0177,
#         0.0169, 0.0168, 0.0167, 0.0164, 0.0163, 0.0160, 0.0155, 0.0154, 0.0154,
#         0.0153, 0.0153, 0.0153, 0.0153, 0.0151, 0.0150, 0.0149, 0.0142, 0.0141,
#         0.0140, 0.0139, 0.0138, 0.0138, 0.0138, 0.0137, 0.0135, 0.0135, 0.0133,
#         0.0133, 0.0132, 0.0131, 0.0131, 0.0129, 0.0126, 0.0126, 0.0125, 0.0124,
#         0.0124, 0.0124, 0.0123, 0.0123, 0.0122, 0.0122, 0.0122, 0.0121, 0.0121,
#         0.0120, 0.0119, 0.0118, 0.0117, 0.0114, 0.0113, 0.0113, 0.0113, 0.0112,
#         0.0112, 0.0112, 0.0110, 0.0108, 0.0108, 0.0107, 0.0105, 0.0105, 0.0104,
#         0.0103, 0.0103, 0.0103, 0.0103, 0.0103, 0.0101, 0.0101, 0.0101, 0.0100,
#         0.0099, 0.0099, 0.0098, 0.0097, 0.0097, 0.0096, 0.0096, 0.0096, 0.0096,
#         0.0095, 0.0095, 0.0095, 0.0095, 0.0094, 0.0093, 0.0092, 0.0092, 0.0091,
#         0.0091, 0.0091, 0.0091, 0.0090, 0.0090, 0.0089, 0.0089, 0.0089, 0.0088,
#         0.0088, 0.0088, 0.0087, 0.0086, 0.0086, 0.0085, 0.0085, 0.0085, 0.0084,
#         0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083,
#         0.0083, 0.0082, 0.0082, 0.0082, 0.0082, 0.0082, 0.0082, 0.0081, 0.0081,
#         0.0081, 0.0081, 0.0081, 0.0080, 0.0080, 0.0079, 0.0079, 0.0079, 0.0078,
#         0.0078, 0.0078, 0.0078, 0.0077, 0.0077, 0.0077, 0.0076, 0.0076, 0.0075,
#         0.0074, 0.0074, 0.0074, 0.0073, 0.0073, 0.0073, 0.0073, 0.0073, 0.0073,
#         0.0072, 0.0072, 0.0057, 0.0036, 0.0028, 0.0024, 0.0020, 0.0019, 0.0019,
#         0.0017, 0.0017, 0.0015, 0.0011, 0.0011]), 'labels': tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 2, 1, 1, 1, 0, 4, 0, 0, 3, 4, 4, 0, 1,
#         2, 0, 1, 0, 3, 0, 0, 0, 0, 3, 0, 0, 2, 2, 0, 3, 0, 0, 4, 0, 1, 1, 4, 1,
#         0, 0, 1, 1, 1, 3, 3, 0, 0, 4, 4, 2, 0, 1, 3, 0, 1, 2, 0, 4, 0, 0, 3, 3,
#         2, 4, 2, 2, 3, 4, 1, 4, 3, 1, 4, 4, 1, 3, 0, 1, 3, 4, 3, 0, 0, 0, 1, 0,
#         3, 2, 2, 1, 0, 2, 4, 1, 0, 2, 0, 0, 4, 2, 0, 3, 2, 2, 0, 2, 2, 0, 2, 4,
#         0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 4, 0, 4, 0, 3, 3, 0, 0, 1, 3, 0, 0, 1, 1,
#         0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 4, 0, 4, 2, 0,
#         4, 0, 2, 0, 0, 0, 0, 2, 3, 4, 0, 0, 4, 1, 1, 0, 1, 0, 1, 0, 0, 2, 3, 0,
#         1, 0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0,
#         0, 0, 3, 0, 3, 4, 0, 1, 4, 3, 0, 0, 3, 4, 0, 2, 0, 1, 0, 1, 0, 2, 0, 0,
#         0, 0, 0, 0, 3, 4, 0, 3, 0, 0, 0, 4, 0, 0, 4, 1, 0, 4, 1, 0, 0, 1, 0, 0,
#         0, 0, 1, 1, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1]), 'bboxes': tensor([[126.7741, 217.5205, 163.4435, 270.6203],
#         [ 97.2121, 433.7397, 157.7018, 473.9344],
#         [522.3201, 242.1748, 581.3476, 282.2158],
#         ...,
#         [575.1943, 501.8248, 623.0279, 511.9221],
#         [189.9367,  95.1477, 239.2880, 141.0479],
#         [ 46.2219, 296.6235, 117.7318, 339.6447]])}}


def conver2cleanlab(pred, label_path, save_path, class_num=5):
    
    import pickle as pk
    with open(pred, "rb") as f:
        pred_data = pk.load(f)
        
    with open(label_path) as f:
        label_data = json.load(f)
    
        
    label_output = []
    pred_output = []
    image2name = dict()
    for i in tqdm.tqdm(pred_data):
        img_path = i["img_path"]
        img_path2 = i["img_path2"]
        img_name = Path(img_path).name
        scale_factor = i["scale_factor"]
        batch_input_shape = i["batch_input_shape"]
        # image2name[img_name]
        
        pred_instances = i["pred_instances"]
        # for j in pred_instances:
        labels = pred_instances["labels"].numpy()
        scores = pred_instances["scores"].numpy()
        bboxes = pred_instances["bboxes"].numpy()
        labels = labels[scores > 0.1]
        bboxes = bboxes[scores > 0.1]
        scores = scores[scores > 0.1]

        k_classes = []
        for i in range(class_num):
            n_class = (labels == i)
            k_scores = scores[n_class]
            k_bboxes = bboxes[n_class]
            # print(k_scores.shape, k_bboxes.shape)
            k_instances = np.concatenate([k_bboxes, k_scores[:, None]], axis=1)
            k_classes.append(k_instances)
        pred_output.append(k_classes)
        
    
        
    # print(pred_data[0], len(pred_data))
    
    # labels is a list where for the i-th image in our dataset, 
    # labels[i] is a dictionary containing: 
    # key labels – a list of class labels for each bounding box in this image and 
    # key bboxes – a numpy array of the bounding boxes’ coordinates. 
    # Each bounding box in labels[i]['bboxes'] is in the format [x1,y1,x2,y2] format with respect to the image matrix 
    # where (x1,y1) corresponds to the top-left corner of the box and (x2,y2) the bottom-right (E.g. XYXY in Keras, Detectron 2).
    images = label_data["images"]
    annotations = label_data["annotations"]
    categories = label_data["categories"]
    print(categories)
    image2label = dict()
    for img in images:
        img_id = img["id"]
        file_name = img["file_name"]
        image2name[img_id] = file_name
        image2label[img_id] = []
    
    for anno in annotations:
        #   {"id": 125686, "category_id": 0, "iscrowd": 0, "segmentation": [[164.81, 417.51,......167.55, 410.64]], "image_id": 242287, "area": 42061.80340000001, "bbox": [19.23, 383.18, 314.5, 244.46]},
        category_id = anno["category_id"]
        image_id = anno["image_id"]
        bbox = anno["bbox"]
        x, y, w, h = bbox
        
        image2label[image_id].append([x, y, x + w, y + h, category_id])
    
    image2label2 = dict()
    for k, v in image2label.items():
        image_id = k
        bboxes = v
        # print(bboxes)
        if len(bboxes) > 0:
            bboxes = np.stack(bboxes, axis=0)
        else:
            bboxes = np.zeros((0, 5))
        image2label2[image_id] = bboxes
        
    for k, v in image2label2.items():
        # print(k)
        # print(k, image2name[k])
        label_output.append(
            {
                "labels": v[:, -1] - 1,
                "bboxes": v[:, :-1],
                "seg_map": image2name[k]
            }
        )
        # image2pred[]
    
    pickle.dump(label_output, open(os.path.join(save_path, "labels.pkl"), "wb"))
    pickle.dump(pred_output, open(os.path.join(save_path, "predictions.pkl"), "wb"))
    # print(pred_output[0].)

if __name__ == "__main__":
    from cleanlab.object_detection.filter import find_label_issues
    from cleanlab.object_detection.rank import (
        _separate_label,
        _separate_prediction,
        get_label_quality_scores,
        issues_from_scores,
    )
    from cleanlab.object_detection.summary import visualize

    # pred_mmdet = "/root/workspace/data/dual_mmdetection/mmdetection/0530_train_tta.pkl"
    label_coco = "projects/data/mmdet/gaiic/mix_all/merged_coco_new_vis_3cls.json"
    save_path = "data/"
    
    # conver2cleanlab(pred_mmdet, label_coco, save_path=save_path)

    # IMAGE_PATH = '/root/workspace/data/GAIIC2024/train_more/tir/'  # path to raw image files downloaded above
    # predictions = pickle.load(open(os.path.join(save_path, "predictions.pkl"), "rb"))
    labels = pickle.load(open(os.path.join(save_path, "labels.pkl"), "rb"))
    # print(predictions[0])
    # print(labels[0])
    # print("-" * 100)
    # print("find issues ...")
    # print(len(labels), len(predictions))
    # num_examples_to_show = 5

    ##########################################
    # label_issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True, overlapping_label_check=True)

    # print(f"issue index number: {len(label_issue_idx)}")

    # print("-" * 100)
    # print("get quality scores ...")
    # scores = get_label_quality_scores(labels, predictions)


    # issue_idx = issues_from_scores(scores, threshold=0.5)  # lower threshold will return fewer (but more confident) label issues
    # # issue_idx[:num_examples_to_show], scores[issue_idx][:num_examples_to_show]
    # pickle.dump(scores, open("/root/workspace/data/cleanlab/data/scores.pkl", "wb"))
    # pickle.dump(issue_idx, open("/root/workspace/data/cleanlab/data/issue_idx.pkl", "wb"))

    ####################################

    scores = pickle.load(open("data/scores.pkl", "rb"))
    issue_idx = pickle.load(open("data/issue_idx.pkl", "rb"))
    
    know_issue_names = [f"{i:05}.jpg" for i in range(0, 1000)]
    num_examples_to_show = 1e6
    for issue_to_visualize in range(len(scores)):
        # issue_to_visualize = issue_idx[0]  # change this to view other images
        class_names = {"0": "car", "1": "truck", "2": "bus", "3":"van", "4": "freight_car"}

        label = labels[issue_to_visualize]
        # prediction = predictions[issue_to_visualize]
        score = scores[issue_to_visualize]
        if label['seg_map'] in know_issue_names and score < 0.98:
            print(label['seg_map'], score)
        # image_path = IMAGE_PATH + label['seg_map']
        
        # image_path2 = IMAGE_PATH.replace("tir", "rgb") + label['seg_map']
        # image_name = Path(image_path).name
        # save_path1 = os.path.join("/root/workspace/data/cleanlab/data/issue_vis", image_name.replace(".jpg", "_tir.jpg"))
        # save_path2 = os.path.join("/root/workspace/data/cleanlab/data/issue_vis", image_name.replace(".jpg", "_rgb.jpg"))
        # print(image_path, '| idx', issue_to_visualize , '| label quality score:', score, '| is issue: True')
        # visualize(image_path, label=label, prediction=prediction, class_names=class_names, overlay=False, save_path=save_path1)
        # visualize(image_path2, label=label, prediction=prediction, class_names=class_names, overlay=False, save_path=save_path2)
