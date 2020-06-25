import sys
import os
import csv
from tqdm import tqdm
import numpy as np
import pickle
import warnings

import torch
import torchvision
import torchvision.models as models

from training_dataset import retrieval_dataset
from pooling import *
from features import transform_480
from net import seresnet152, densenet201
import pandas as pd
import ast

import argparse

# python search_post_opt.py --test_image_path='/home/jisijie/MM/Challenge_validation_set_2020/val_2020' --result_key='predictions'

parser = argparse.ArgumentParser()
parser.add_argument('--result_key', dest='result_key', type=str, default=None,
                    help='Name of the result file')
parser.add_argument('--test_image_path', dest='test_image_path', type=str, default=None,
                    help='Root path to test images.')
args = parser.parse_args()
# if args.result_key == None:
#     raise Exception("Need to specify name of the result file")

def load_feature(feat_name):
    with open(feat_name, "rb") as file_to_read:
        feature=pickle.load(file_to_read)
    name=feature['name']
    return name,feature

if __name__ == "__main__":
    #test_image_path=sys.argv[1]
    #result_path=sys.argv[2]
    
    test_image_path = args.test_image_path
    result_key = args.result_key

    result_path='./result/1st_result.csv'

    se_path='./pretrained/seresnet152.t7'
    dense_path='./pretrained/densenet201.t7'
    se_feature_path='./feature/feat_seresnet152.pkl'
    dense_feature_path='./feature/feat_densenet201.pkl'

    name_list,dense_feature=load_feature(dense_feature_path)
    # print(len(name_list))

    name_list,senet_feature=load_feature(se_feature_path)
    #import pdb; pdb.set_trace()
    feature={'dense201':dense_feature,'seresnet152':senet_feature}

    feat_type={'dense201':['Grmac'],'seresnet152':['Grmac']}
    weight={
        'dense201':{'Mac':1,'rmac':1,'ramac':1,'Grmac':3},
        'seresnet152':{'Mac':1,'rmac':1,'ramac':1,'Grmac':1}
    }

    # feat_type={'dense201':['Grmac'],'seresnet152':['Mac']}
    # weight={
    #     'dense201':{'Mac':0,'rmac':0,'ramac':0,'Grmac':1},
    #     'seresnet152':{'Mac':1,'rmac':0,'ramac':0,'Grmac':0}
    # }

    dim_feature={
        'dense201':1920,
        'seresnet152':2048
    }
    batch_size=1

    # similarity=torch.zeros(len(os.listdir(test_image_path)),len(name_list))
    similarity=torch.ones(len(os.listdir(test_image_path)),len(name_list))
    # print(similarity.size())

    for model_name in ['dense201','seresnet152']:
        feature_model=feature[model_name]
        for item in feat_type[model_name]:
            if model_name == 'seresnet152':
                model=seresnet152(se_path,item)
            elif model_name == 'dense201':
                model=densenet201(dense_path,item)
            else:
                pass

            feat_reserved=feature_model[item]
            # print(type(feat_reserved))

            dataset = retrieval_dataset(test_image_path,transform=transform_480)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            model=model.cpu()
            model.eval()
            
            query=torch.empty(len(os.listdir(test_image_path)),dim_feature[model_name])
            name_test=[]
            with torch.no_grad():
                for i, (inputs, names) in tqdm(enumerate(testloader)):
                    query[i*batch_size:i*batch_size+len(names)] = model(inputs).cpu()
                    name_test.extend(names)
            #import pdb; pdb.set_trace()
            feat_reserved=torch.Tensor(feat_reserved).transpose(1,0)
            query=torch.Tensor(query)
            
            # similarity+=torch.matmul(query,feat_reserved)*weight[model_name][item]
            similarity = similarity * (1 + torch.matmul(query,feat_reserved)*weight[model_name][item])

    # print(similarity.size())
    # print(name_test)
    _, predicted = similarity.topk(10)
    predicted=predicted.tolist()
    # print(predicted)
    dict_result=dict(zip(name_test,predicted))
    # print(dict_result)

    raw_results = {}
    for testing_imn, training_indices in dict_result.items():
        testing_id = testing_imn.split('.')[0]
        raw_results[testing_id] = []
        for idx in training_indices:
            training_id = name_list[idx].split('.')[0]
            raw_results[testing_id].append(training_id)

    # Search top3 results within labels for each testing image
    print("Refine the results ... ")
    # Get labels
    labels = pd.read_csv("training_matchings2.csv")
    labels_dict = {}
    for i in range(len(labels)):
        img_id = labels.iloc[i, 0]
        matchings = labels.iloc[i, 1]
        labels_dict[img_id] = ast.literal_eval(matchings)

    # Search for top3
    subsets = {}
    for k, v in raw_results.items():
        top_keys = v[:3]
        subset = set()
        for top_key in top_keys:
            subset.add(top_key)
            if top_key in labels_dict.keys():
                subset.update(labels_dict[top_key])
        subsets[k] = set(subset)
        if len(subsets[k]) < 7:
            subsets[k].update(v[3:])
        subsets[k] = list(subsets[k])
    #     print(len(subsets[k]))
    # print(subsets)


    # Re-ranking the subset results
    # Should handle the case when img_id in labels does not exist in the images ***
    img_id_to_idx = {}
    for i, imn in enumerate(name_list):
        img_id = imn.split(".")[0]
        img_id_to_idx[img_id] = i

    subset_testid_to_subtrainidx = {}
    for testing_id, training_ids in subsets.items():
        # Get training features according to selected training ids
        training_indices = []
        for training_id in training_ids:
            if training_id in img_id_to_idx:
                # Convert training_id to index
                training_idx = img_id_to_idx[training_id]
                training_indices.append(training_idx)
        subset_testid_to_subtrainidx[testing_id] = training_indices
    # print(subset_testid_to_subtrainidx)

    subset_results = {}
    for testing_id, training_indices in subset_testid_to_subtrainidx.items():
        for i, imn in enumerate(name_test):
            img_id = imn.split(".")[0]
            if testing_id == img_id:
                testing_idx = i
        testing_similarities = similarity[testing_idx]
        subset_results[testing_id] = testing_similarities[training_indices].numpy()
        sorted_indices_of_training_indices = np.argsort(subset_results[testing_id])[::-1]
        # print(testing_id, subset_results[testing_id])

        sorted_training_indices = []
        for idx in sorted_indices_of_training_indices:
            sorted_training_indices.append(training_indices[idx])
        # print(testing_id, sorted_training_indices)

        sorted_training_img_ids = []
        for idx in sorted_training_indices:
            sorted_training_img_ids.append(name_list[idx].split(".")[0])
        # print(testing_id, sorted_training_img_ids)
        subset_results[testing_id] = sorted_training_img_ids

    with open("result/%s.csv" % result_key, "w") as f:
        for testing_id, training_ids in subset_results.items():
            print(len(training_ids))
            all_ids = [testing_id]
            # all_ids.extend(training_ids)
            all_ids.extend(training_ids[:7])
            f.write(",".join(all_ids) + "\n")

    print("Finish!")

    # #saving csv
    # img_results=[]
    # name_test.sort()
    # for name in name_test:
    #     temp=[name.split('.')[0]]
    #     for idx in dict_result[name]:
    #         temp.append(name_list[idx].split('.')[0])
    #     img_results.append(temp)
    # print('saving')
    # out = open(result_path,'w')
    # csv_write = csv.writer(out)
    # csv_write.writerows(img_results)
