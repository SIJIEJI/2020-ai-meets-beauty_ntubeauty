import os
import argparse
import numpy as np
from PIL import Image
import shutil

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import warnings
from tqdm import tqdm
from pooling import *
from training_dataset import retrieval_dataset
import net


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

transform_480 = transforms.Compose([
    transforms.Resize((480,480)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

model_path={
    'densenet201':'./pretrained/densenet201.t7',
    'seresnet152':'./pretrained/seresnet152.t7',
}
feature_length={
    'densenet201':1920,
    'seresnet152':2048
}

if __name__ == "__main__":
    name_list=os.listdir('/home/jisijie/MM/Perfect500K-Beauty-and-Personal-Care-Products-Retrieval-Challenge/dataset_clean/PF-500K')   # original full dataset
    # name_list=os.listdir('/home/jisijie/MM/beauty-ir_v7.0/data/testing_images')    # jingwen 10K dataset
    
    name_list.sort()
    mode='cuda' # or 'cpu'
    batch_size = 32
    #import pdb; pdb.set_trace()
    for model_name in ['seresnet152','densenet201']:
        model=net.__dict__[model_name](model_path[model_name], 'Grmac')
        if mode=='cuda':
            model=model.cuda()
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        dataset = retrieval_dataset('/home/jisijie/MM/Perfect500K-Beauty-and-Personal-Care-Products-Retrieval-Challenge/dataset_clean/PF-500K',transform=transform_480)
        # dataset = retrieval_dataset('/home/jisijie/MM/beauty-ir_v7.0/data/testing_images',transform=transform_480)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        # build dictionary to save features
        feat_dict={
            # 'rmac':torch.empty(len(name_list),feature_length[model_name]),
            # 'ramac':torch.empty(len(name_list),feature_length[model_name]),
            'Grmac':torch.empty(len(name_list),feature_length[model_name])
            # 'SPoC':torch.empty(len(name_list),feature_length[model_name]),
            # 'Mac':torch.empty(len(name_list),feature_length[model_name])
        }
        img_list=[]
        model.eval()
        with torch.no_grad():    # save memory, don't save gradiant
            #for i, (inputs, names) in tqdm(enumerate(testloader)):
            for i, (inputs, names) in tqdm(enumerate(testloader)):
                inputs = inputs.to(mode)  # use gpu
                # feature_rmac,feature_ramac,feature_Grmac,feature_Mac,feature_SPoC, = model(inputs)
                feature_Grmac = model(inputs)
                # print(features.size())
                #import pdb; pdb.set_trace()
                # i*batch_size = number of already extracted features
                # feat_dict['rmac'][i*batch_size:i*batch_size+len(names)]=feature_rmac.cpu()
                # feat_dict['ramac'][i*batch_size:i*batch_size+len(names)]=feature_ramac.cpu()
                feat_dict['Grmac'][i*batch_size:i*batch_size+len(names)]=feature_Grmac.cpu()
                # feat_dict['SPoC'][i*batch_size:i*batch_size+len(names)]=feature_SPoC.cpu()
                # feat_dict['Mac'][i*batch_size:i*batch_size+len(names)]=feature_Mac.cpu()
                
                assert name_list[i*batch_size:i*batch_size+len(names)]==list(names)
                img_list.extend(names)  # extend more than one compare to append
                
        
        #import pdb; pdb.set_trace()
        with open("./feature/feat_{}.pkl".format(model_name), "wb") as file_to_save:  # format here is to use model_name fill in feat_{}
            pickle.dump(
                {
                'name':img_list,
                # 'rmac':feat_dict['rmac'].half().numpy(),
                # 'ramac':feat_dict['ramac'].half().numpy(),
                'Grmac':feat_dict['Grmac'].half().numpy(), # half precision
                # 'SPoC':feat_dict['SPoC'].half().numpy(),
                # 'Mac':feat_dict['Mac'].half().numpy()
                    }, 
                file_to_save, 
                -1
                )