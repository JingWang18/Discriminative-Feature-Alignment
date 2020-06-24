import torch
import torch.nn as nn
from _utils import load_state_dict_from_url
from collections import OrderedDict


__all__ = ['resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}
    
def key_transformation(old_key):
    #Layer 1
    if old_key == "layer1.0.conv1.weight":
        return "conv111.weight"
    elif old_key == "layer1.0.bn1.running_mean":
        return "bn111.running_mean"
    elif old_key == "layer1.0.bn1.running_var":
        return "bn111.running_var"
    elif old_key == "layer1.0.bn1.weight":
        return "bn111.weight"
    elif old_key == "layer1.0.bn1.bias":
        return "bn111.bias"
    elif old_key == "layer1.0.conv2.weight":
        return "conv112.weight"
    elif old_key == "layer1.0.bn2.running_mean":
        return "bn112.running_mean"
    elif old_key == "layer1.0.bn2.running_var":
        return "bn112.running_var"
    elif old_key == "layer1.0.bn2.weight":
        return "bn112.weight"
    elif old_key == "layer1.0.bn2.bias":
        return "bn112.bias"
    elif old_key == "layer1.0.conv3.weight":
        return "conv113.weight"
    elif old_key == "layer1.0.bn3.running_mean":
        return "bn113.running_mean"
    elif old_key == "layer1.0.bn3.running_var":
        return "bn113.running_var"
    elif old_key == "layer1.0.bn3.weight":
        return "bn113.weight"
    elif old_key == "layer1.0.bn3.bias":
        return "bn113.bias"
    elif old_key == "layer1.0.downsample.0.weight":
        return "conv114.weight"
    elif old_key == "layer1.0.downsample.1.running_mean":
        return "bn114.running_mean"
    elif old_key == "layer1.0.downsample.1.running_var":
        return "bn114.running_var"
    elif old_key == "layer1.0.downsample.1.weight":
        return "bn114.weight"
    elif old_key == "layer1.0.downsample.1.bias":
        return "bn114.bias"
    
    elif old_key == "layer1.1.conv1.weight":
        return "conv121.weight"
    elif old_key == "layer1.1.bn1.running_mean":
        return "bn121.running_mean"
    elif old_key == "layer1.1.bn1.running_var":
        return "bn121.running_var"
    elif old_key == "layer1.1.bn1.weight":
        return "bn121.weight"
    elif old_key == "layer1.1.bn1.bias":
        return "bn121.bias"
    elif old_key == "layer1.1.conv2.weight":
        return "conv122.weight"
    elif old_key == "layer1.1.bn2.running_mean":
        return "bn122.running_mean"
    elif old_key == "layer1.1.bn2.running_var":
        return "bn122.running_var"
    elif old_key == "layer1.1.bn2.weight":
        return "bn122.weight"
    elif old_key == "layer1.1.bn2.bias":
        return "bn122.bias"
    elif old_key == "layer1.1.conv3.weight":
        return "conv123.weight"
    elif old_key == "layer1.1.bn3.running_mean":
        return "bn123.running_mean"
    elif old_key == "layer1.1.bn3.running_var":
        return "bn123.running_var"
    elif old_key == "layer1.1.bn3.weight":
        return "bn123.weight"
    elif old_key == "layer1.1.bn3.bias":
        return "bn123.bias"
    
    if old_key == "layer1.2.conv1.weight":
        return "conv131.weight"
    elif old_key == "layer1.2.bn1.running_mean":
        return "bn131.running_mean"
    elif old_key == "layer1.2.bn1.running_var":
        return "bn131.running_var"
    elif old_key == "layer1.2.bn1.weight":
        return "bn131.weight"
    elif old_key == "layer1.2.bn1.bias":
        return "bn131.bias"
    elif old_key == "layer1.2.conv2.weight":
        return "conv132.weight"
    elif old_key == "layer1.2.bn2.running_mean":
        return "bn132.running_mean"
    elif old_key == "layer1.2.bn2.running_var":
        return "bn132.running_var"
    elif old_key == "layer1.2.bn2.weight":
        return "bn132.weight"
    elif old_key == "layer1.2.bn2.bias":
        return "bn132.bias"
    elif old_key == "layer1.2.conv3.weight":
        return "conv133.weight"
    elif old_key == "layer1.2.bn3.running_mean":
        return "bn133.running_mean"
    elif old_key == "layer1.2.bn3.running_var":
        return "bn133.running_var"
    elif old_key == "layer1.2.bn3.weight":
        return "bn133.weight"
    elif old_key == "layer1.2.bn3.bias":
        return "bn133.bias"
    
    
    #Layer 2
    if old_key == "layer2.0.conv1.weight":
        return "conv211.weight"
    elif old_key == "layer2.0.bn1.running_mean":
        return "bn211.running_mean"
    elif old_key == "layer2.0.bn1.running_var":
        return "bn211.running_var"
    elif old_key == "layer2.0.bn1.weight":
        return "bn211.weight"
    elif old_key == "layer2.0.bn1.bias":
        return "bn211.bias"
    elif old_key == "layer2.0.conv2.weight":
        return "conv212.weight"
    elif old_key == "layer2.0.bn2.running_mean":
        return "bn212.running_mean"
    elif old_key == "layer2.0.bn2.running_var":
        return "bn212.running_var"
    elif old_key == "layer2.0.bn2.weight":
        return "bn212.weight"
    elif old_key == "layer2.0.bn2.bias":
        return "bn212.bias"
    elif old_key == "layer2.0.conv3.weight":
        return "conv213.weight"
    elif old_key == "layer2.0.bn3.running_mean":
        return "bn213.running_mean"
    elif old_key == "layer2.0.bn3.running_var":
        return "bn213.running_var"
    elif old_key == "layer2.0.bn3.weight":
        return "bn213.weight"
    elif old_key == "layer2.0.bn3.bias":
        return "bn213.bias"
    elif old_key == "layer2.0.downsample.0.weight":
        return "conv214.weight"
    elif old_key == "layer2.0.downsample.1.running_mean":
        return "bn214.running_mean"
    elif old_key == "layer2.0.downsample.1.running_var":
        return "bn214.running_var"
    elif old_key == "layer2.0.downsample.1.weight":
        return "bn214.weight"
    elif old_key == "layer2.0.downsample.1.bias":
        return "bn214.bias"
    
    elif old_key == "layer2.1.conv1.weight":
        return "conv221.weight"
    elif old_key == "layer2.1.bn1.running_mean":
        return "bn221.running_mean"
    elif old_key == "layer2.1.bn1.running_var":
        return "bn221.running_var"
    elif old_key == "layer2.1.bn1.weight":
        return "bn221.weight"
    elif old_key == "layer2.1.bn1.bias":
        return "bn221.bias"
    elif old_key == "layer2.1.conv2.weight":
        return "conv222.weight"
    elif old_key == "layer2.1.bn2.running_mean":
        return "bn222.running_mean"
    elif old_key == "layer2.1.bn2.running_var":
        return "bn222.running_var"
    elif old_key == "layer2.1.bn2.weight":
        return "bn222.weight"
    elif old_key == "layer2.1.bn2.bias":
        return "bn222.bias"
    elif old_key == "layer2.1.conv3.weight":
        return "conv223.weight"
    elif old_key == "layer2.1.bn3.running_mean":
        return "bn223.running_mean"
    elif old_key == "layer2.1.bn3.running_var":
        return "bn223.running_var"
    elif old_key == "layer2.1.bn3.weight":
        return "bn223.weight"
    elif old_key == "layer2.1.bn3.bias":
        return "bn223.bias"
    
    if old_key == "layer2.2.conv1.weight":
        return "conv231.weight"
    elif old_key == "layer2.2.bn1.running_mean":
        return "bn231.running_mean"
    elif old_key == "layer2.2.bn1.running_var":
        return "bn231.running_var"
    elif old_key == "layer2.2.bn1.weight":
        return "bn231.weight"
    elif old_key == "layer2.2.bn1.bias":
        return "bn231.bias"
    elif old_key == "layer2.2.conv2.weight":
        return "conv232.weight"
    elif old_key == "layer2.2.bn2.running_mean":
        return "bn232.running_mean"
    elif old_key == "layer2.2.bn2.running_var":
        return "bn232.running_var"
    elif old_key == "layer2.2.bn2.weight":
        return "bn232.weight"
    elif old_key == "layer2.2.bn2.bias":
        return "bn232.bias"
    elif old_key == "layer2.2.conv3.weight":
        return "conv233.weight"
    elif old_key == "layer2.2.bn3.running_mean":
        return "bn233.running_mean"
    elif old_key == "layer2.2.bn3.running_var":
        return "bn233.running_var"
    elif old_key == "layer2.2.bn3.weight":
        return "bn233.weight"
    elif old_key == "layer2.2.bn3.bias":
        return "bn233.bias"
    
    if old_key == "layer2.3.conv1.weight":
        return "conv241.weight"
    elif old_key == "layer2.3.bn1.running_mean":
        return "bn241.running_mean"
    elif old_key == "layer2.3.bn1.running_var":
        return "bn241.running_var"
    elif old_key == "layer2.3.bn1.weight":
        return "bn241.weight"
    elif old_key == "layer2.3.bn1.bias":
        return "bn241.bias"
    elif old_key == "layer2.3.conv2.weight":
        return "conv242.weight"
    elif old_key == "layer2.3.bn2.running_mean":
        return "bn242.running_mean"
    elif old_key == "layer2.3.bn2.running_var":
        return "bn242.running_var"
    elif old_key == "layer2.3.bn2.weight":
        return "bn242.weight"
    elif old_key == "layer2.3.bn2.bias":
        return "bn242.bias"
    elif old_key == "layer2.3.conv3.weight":
        return "conv243.weight"
    elif old_key == "layer2.3.bn3.running_mean":
        return "bn243.running_mean"
    elif old_key == "layer2.3.bn3.running_var":
        return "bn243.running_var"
    elif old_key == "layer2.3.bn3.weight":
        return "bn243.weight"
    elif old_key == "layer2.3.bn3.bias":
        return "bn243.bias"
    
    #Layer 3
    if old_key == "layer3.0.conv1.weight":
        return "conv311.weight"
    elif old_key == "layer3.0.bn1.running_mean":
        return "bn311.running_mean"
    elif old_key == "layer3.0.bn1.running_var":
        return "bn311.running_var"
    elif old_key == "layer3.0.bn1.weight":
        return "bn311.weight"
    elif old_key == "layer3.0.bn1.bias":
        return "bn311.bias"
    elif old_key == "layer3.0.conv2.weight":
        return "conv312.weight"
    elif old_key == "layer3.0.bn2.running_mean":
        return "bn312.running_mean"
    elif old_key == "layer3.0.bn2.running_var":
        return "bn312.running_var"
    elif old_key == "layer3.0.bn2.weight":
        return "bn312.weight"
    elif old_key == "layer3.0.bn2.bias":
        return "bn312.bias"
    elif old_key == "layer3.0.conv3.weight":
        return "conv313.weight"
    elif old_key == "layer3.0.bn3.running_mean":
        return "bn313.running_mean"
    elif old_key == "layer3.0.bn3.running_var":
        return "bn313.running_var"
    elif old_key == "layer3.0.bn3.weight":
        return "bn313.weight"
    elif old_key == "layer3.0.bn3.bias":
        return "bn313.bias"
    elif old_key == "layer3.0.downsample.0.weight":
        return "conv314.weight"
    elif old_key == "layer3.0.downsample.1.running_mean":
        return "bn314.running_mean"
    elif old_key == "layer3.0.downsample.1.running_var":
        return "bn314.running_var"
    elif old_key == "layer3.0.downsample.1.weight":
        return "bn314.weight"
    elif old_key == "layer3.0.downsample.1.bias":
        return "bn314.bias"
    
    elif old_key == "layer3.1.conv1.weight":
        return "conv321.weight"
    elif old_key == "layer3.1.bn1.running_mean":
        return "bn321.running_mean"
    elif old_key == "layer3.1.bn1.running_var":
        return "bn321.running_var"
    elif old_key == "layer3.1.bn1.weight":
        return "bn321.weight"
    elif old_key == "layer3.1.bn1.bias":
        return "bn321.bias"
    elif old_key == "layer3.1.conv2.weight":
        return "conv322.weight"
    elif old_key == "layer3.1.bn2.running_mean":
        return "bn322.running_mean"
    elif old_key == "layer3.1.bn2.running_var":
        return "bn322.running_var"
    elif old_key == "layer3.1.bn2.weight":
        return "bn322.weight"
    elif old_key == "layer3.1.bn2.bias":
        return "bn322.bias"
    elif old_key == "layer3.1.conv3.weight":
        return "conv323.weight"
    elif old_key == "layer3.1.bn3.running_mean":
        return "bn323.running_mean"
    elif old_key == "layer3.1.bn3.running_var":
        return "bn323.running_var"
    elif old_key == "layer3.1.bn3.weight":
        return "bn323.weight"
    elif old_key == "layer3.1.bn3.bias":
        return "bn323.bias"
    
    if old_key == "layer3.2.conv1.weight":
        return "conv331.weight"
    elif old_key == "layer3.2.bn1.running_mean":
        return "bn331.running_mean"
    elif old_key == "layer3.2.bn1.running_var":
        return "bn331.running_var"
    elif old_key == "layer3.2.bn1.weight":
        return "bn331.weight"
    elif old_key == "layer3.2.bn1.bias":
        return "bn331.bias"
    elif old_key == "layer3.2.conv2.weight":
        return "conv332.weight"
    elif old_key == "layer3.2.bn2.running_mean":
        return "bn332.running_mean"
    elif old_key == "layer3.2.bn2.running_var":
        return "bn332.running_var"
    elif old_key == "layer3.2.bn2.weight":
        return "bn332.weight"
    elif old_key == "layer3.2.bn2.bias":
        return "bn332.bias"
    elif old_key == "layer3.2.conv3.weight":
        return "conv333.weight"
    elif old_key == "layer3.2.bn3.running_mean":
        return "bn333.running_mean"
    elif old_key == "layer3.2.bn3.running_var":
        return "bn333.running_var"
    elif old_key == "layer3.2.bn3.weight":
        return "bn333.weight"
    elif old_key == "layer3.2.bn3.bias":
        return "bn333.bias"
    
    if old_key == "layer3.3.conv1.weight":
        return "conv341.weight"
    elif old_key == "layer3.3.bn1.running_mean":
        return "bn341.running_mean"
    elif old_key == "layer3.3.bn1.running_var":
        return "bn341.running_var"
    elif old_key == "layer3.3.bn1.weight":
        return "bn341.weight"
    elif old_key == "layer3.3.bn1.bias":
        return "bn341.bias"
    elif old_key == "layer3.3.conv2.weight":
        return "conv342.weight"
    elif old_key == "layer3.3.bn2.running_mean":
        return "bn342.running_mean"
    elif old_key == "layer3.3.bn2.running_var":
        return "bn342.running_var"
    elif old_key == "layer3.3.bn2.weight":
        return "bn342.weight"
    elif old_key == "layer3.3.bn2.bias":
        return "bn342.bias"
    elif old_key == "layer3.3.conv3.weight":
        return "conv343.weight"
    elif old_key == "layer3.3.bn3.running_mean":
        return "bn343.running_mean"
    elif old_key == "layer3.3.bn3.running_var":
        return "bn343.running_var"
    elif old_key == "layer3.3.bn3.weight":
        return "bn343.weight"
    elif old_key == "layer3.3.bn3.bias":
        return "bn343.bias"
    
    if old_key == "layer3.4.conv1.weight":
        return "conv351.weight"
    elif old_key == "layer3.4.bn1.running_mean":
        return "bn351.running_mean"
    elif old_key == "layer3.4.bn1.running_var":
        return "bn351.running_var"
    elif old_key == "layer3.4.bn1.weight":
        return "bn351.weight"
    elif old_key == "layer3.4.bn1.bias":
        return "bn351.bias"
    elif old_key == "layer3.4.conv2.weight":
        return "conv352.weight"
    elif old_key == "layer3.4.bn2.running_mean":
        return "bn352.running_mean"
    elif old_key == "layer3.4.bn2.running_var":
        return "bn352.running_var"
    elif old_key == "layer3.4.bn2.weight":
        return "bn352.weight"
    elif old_key == "layer3.4.bn2.bias":
        return "bn352.bias"
    elif old_key == "layer3.4.conv3.weight":
        return "conv353.weight"
    elif old_key == "layer3.4.bn3.running_mean":
        return "bn353.running_mean"
    elif old_key == "layer3.4.bn3.running_var":
        return "bn353.running_var"
    elif old_key == "layer3.4.bn3.weight":
        return "bn353.weight"
    elif old_key == "layer3.4.bn3.bias":
        return "bn353.bias"
    
    if old_key == "layer3.5.conv1.weight":
        return "conv361.weight"
    elif old_key == "layer3.5.bn1.running_mean":
        return "bn361.running_mean"
    elif old_key == "layer3.5.bn1.running_var":
        return "bn361.running_var"
    elif old_key == "layer3.5.bn1.weight":
        return "bn361.weight"
    elif old_key == "layer3.5.bn1.bias":
        return "bn361.bias"
    elif old_key == "layer3.5.conv2.weight":
        return "conv362.weight"
    elif old_key == "layer3.5.bn2.running_mean":
        return "bn362.running_mean"
    elif old_key == "layer3.5.bn2.running_var":
        return "bn362.running_var"
    elif old_key == "layer3.5.bn2.weight":
        return "bn362.weight"
    elif old_key == "layer3.5.bn2.bias":
        return "bn362.bias"
    elif old_key == "layer3.5.conv3.weight":
        return "conv363.weight"
    elif old_key == "layer3.5.bn3.running_mean":
        return "bn363.running_mean"
    elif old_key == "layer3.5.bn3.running_var":
        return "bn363.running_var"
    elif old_key == "layer3.5.bn3.weight":
        return "bn363.weight"
    elif old_key == "layer3.5.bn3.bias":
        return "bn363.bias"
    
    
    #Layer 4
    if old_key == "layer4.0.conv1.weight":
        return "conv411.weight"
    elif old_key == "layer4.0.bn1.running_mean":
        return "bn411.running_mean"
    elif old_key == "layer4.0.bn1.running_var":
        return "bn411.running_var"
    elif old_key == "layer4.0.bn1.weight":
        return "bn411.weight"
    elif old_key == "layer4.0.bn1.bias":
        return "bn411.bias"
    elif old_key == "layer4.0.conv2.weight":
        return "conv412.weight"
    elif old_key == "layer4.0.bn2.running_mean":
        return "bn412.running_mean"
    elif old_key == "layer4.0.bn2.running_var":
        return "bn412.running_var"
    elif old_key == "layer4.0.bn2.weight":
        return "bn412.weight"
    elif old_key == "layer4.0.bn2.bias":
        return "bn412.bias"
    elif old_key == "layer4.0.conv3.weight":
        return "conv413.weight"
    elif old_key == "layer4.0.bn3.running_mean":
        return "bn413.running_mean"
    elif old_key == "layer4.0.bn3.running_var":
        return "bn413.running_var"
    elif old_key == "layer4.0.bn3.weight":
        return "bn413.weight"
    elif old_key == "layer4.0.bn3.bias":
        return "bn413.bias"
    elif old_key == "layer4.0.downsample.0.weight":
        return "conv414.weight"
    elif old_key == "layer4.0.downsample.1.running_mean":
        return "bn414.running_mean"
    elif old_key == "layer4.0.downsample.1.running_var":
        return "bn414.running_var"
    elif old_key == "layer4.0.downsample.1.weight":
        return "bn414.weight"
    elif old_key == "layer4.0.downsample.1.bias":
        return "bn414.bias"
    
    elif old_key == "layer4.1.conv1.weight":
        return "conv421.weight"
    elif old_key == "layer4.1.bn1.running_mean":
        return "bn421.running_mean"
    elif old_key == "layer4.1.bn1.running_var":
        return "bn421.running_var"
    elif old_key == "layer4.1.bn1.weight":
        return "bn421.weight"
    elif old_key == "layer4.1.bn1.bias":
        return "bn421.bias"
    elif old_key == "layer4.1.conv2.weight":
        return "conv422.weight"
    elif old_key == "layer4.1.bn2.running_mean":
        return "bn422.running_mean"
    elif old_key == "layer4.1.bn2.running_var":
        return "bn422.running_var"
    elif old_key == "layer4.1.bn2.weight":
        return "bn422.weight"
    elif old_key == "layer4.1.bn2.bias":
        return "bn422.bias"
    elif old_key == "layer4.1.conv3.weight":
        return "conv423.weight"
    elif old_key == "layer4.1.bn3.running_mean":
        return "bn423.running_mean"
    elif old_key == "layer4.1.bn3.running_var":
        return "bn423.running_var"
    elif old_key == "layer4.1.bn3.weight":
        return "bn423.weight"
    elif old_key == "layer4.1.bn3.bias":
        return "bn423.bias"
    
    if old_key == "layer4.2.conv1.weight":
        return "conv431.weight"
    elif old_key == "layer4.2.bn1.running_mean":
        return "bn431.running_mean"
    elif old_key == "layer4.2.bn1.running_var":
        return "bn431.running_var"
    elif old_key == "layer4.2.bn1.weight":
        return "bn431.weight"
    elif old_key == "layer4.2.bn1.bias":
        return "bn431.bias"
    elif old_key == "layer4.2.conv2.weight":
        return "conv432.weight"
    elif old_key == "layer4.2.bn2.running_mean":
        return "bn432.running_mean"
    elif old_key == "layer4.2.bn2.running_var":
        return "bn432.running_var"
    elif old_key == "layer4.2.bn2.weight":
        return "bn432.weight"
    elif old_key == "layer4.2.bn2.bias":
        return "bn432.bias"
    elif old_key == "layer4.2.conv3.weight":
        return "conv433.weight"
    elif old_key == "layer4.2.bn3.running_mean":
        return "bn433.running_mean"
    elif old_key == "layer4.2.bn3.running_var":
        return "bn433.running_var"
    elif old_key == "layer4.2.bn3.weight":
        return "bn433.weight"
    elif old_key == "layer4.2.bn3.bias":
        return "bn433.bias"
    
    
    elif old_key == "fc.weight":
        return
    elif old_key == "fc.bias":
        return

    
    
    elif old_key == "features.3.weight":
        return "conv2.weight"
    elif old_key == "features.3.bias":
        return "conv2.bias"
    elif old_key == "features.6.weight":
        return "conv3.weight"
    elif old_key == "features.6.bias":
        return "conv3.bias"
    elif old_key == "features.8.weight":
        return "conv4.weight"
    elif old_key == "features.8.bias":
        return "conv4.bias"
    elif old_key == "features.10.weight":
        return "conv5.weight"
    elif old_key == "features.10.bias":
        return "conv5.bias"
    
    elif old_key == "classifier.1.weight":
        return
    elif old_key == "classifier.1.bias":
        return
    elif old_key == "classifier.4.weight":
        return
    elif old_key == "classifier.4.bias":
        return
    elif old_key == "classifier.6.weight":
        return
    elif old_key == "classifier.6.bias":
        return
    return old_key



def rename_key_resnet():

    target = 'model/resnet_model.pth'

    state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        if new_key == None:
            continue
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)
    new_state_dict = torch.load(target)
        
    return new_state_dict, state_dict

new_state_dict,state_dict = rename_key_resnet()
print("Model's state_dict:")
for param_tensor in new_state_dict:
    print(param_tensor, "\t", new_state_dict[param_tensor].size())
