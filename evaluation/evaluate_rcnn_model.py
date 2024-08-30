from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import pandas as pd

def drawKeyPts(im,keyp,col,th, filename):
    print(filename)
    count = 0
    for curKey in keyp:#
        x=int(curKey[0])
        y=int(curKey[1])
        size = 5
        if filename:
            cv2.circle(im,(x,y),size, col,thickness=th, lineType=8, shift=0)
        else:
            cv2.circle(im,(x,y),size, (0, 0, 255),thickness=th, lineType=8, shift=0)
        if filename:
            cv2.putText(im, str(count), (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 3)
        count += 1
    #plt.imshow(im)  
    if filename:  
        cv2.imwrite(filename, im)
    return im    

import argparse

parser=argparse.ArgumentParser(description="Evaluation")
parser.add_argument("path")
parser.add_argument("testpath")
parser.add_argument("savepath")
args=parser.parse_args()

path = args.path
test_path = args.testpath
savepath = args.savepath

model = keypointrcnn_resnet50_fpn(weights=None)

# Replace the classifier head with the number of keypoints
in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_channels=in_features, num_keypoints=11)

device = 'cuda'
# Set the model's device and data type
model.to(device=device, dtype=torch.float32)

# Add attributes to store the device and model name for later reference
model.device = device
model.name = 'model'

model.load_state_dict(torch.load(path))

img_names = os.listdir(test_path)
test_df = pd.read_csv('./data/landmarks_test.csv')
distances = []
for img_name in img_names:
    image_path = os.path.join(test_path, img_name) 
    img = cv2.imread(image_path)
    img = transforms.ToTensor()(img).to('cuda')
    imgs = [img]
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        results = model(imgs)

    original_keypoints = torch.Tensor(test_df.loc[test_df.loc[:, 'image_name'].str.contains(img_name)].iloc[:, 1:].to_numpy().reshape(-1, 2))
    pred_keypoints = torch.Tensor(results[0]['keypoints'][:, :, :2].tolist()[0])
    #print(original_keypoints)
    #print(pred_keypoints)
    #print(torch.cdist(original_keypoints, pred_keypoints).diag())
    distances.append(torch.cdist(original_keypoints, pred_keypoints).diag())
    image = cv2.imread(image_path)
    #image = drawKeyPts(image, original_keypoints,(255,0,0),2, None)
    drawKeyPts(image, pred_keypoints,(255,0,0),2, os.path.join(savepath, f'{img_name}.png'))

with open(f'{savepath}.txt', 'w') as f:
    f.write(f'Mean: {str(torch.mean(torch.stack(distances)))}\n')
    f.write(f'SD: {str(torch.std(torch.stack(distances)))}\n')
    f.write(f'Var: {str(torch.var(torch.stack(distances)))}\n\n')

    f.write(f'KP Mean:{str(torch.mean(torch.stack(distances), dim=0))}\n')
    f.write(f'KP SD{str(torch.std(torch.stack(distances), dim=0))}\n')
    f.write(f'KP VAR:{str(torch.var(torch.stack(distances), dim=0))}\n')
    f.close()