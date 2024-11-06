import torch
import numpy as np
from utils import *
from class_rnn_1_compatibility import *
from class_rnn_1_folding import *
from torchvision import datasets,transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader

dataset = 'imagenet'
depth = 8
blockdepth = int(depth/2 - 1)
m = 'module.'
o = '.parametrizations.weight.original'

def applybn(modeldict,oldmodel,a,bn):
    modeldict[m+a+o+'0'], modeldict[m+a+o+'1'], bias = extract(m+bn,oldmodel.state_dict()[m+a+o+'0'],oldmodel.state_dict()[m+a+o+'1'])
    return bias

device = torch.device("cuda")

oldmodel = step1cordsnet(dataset=dataset,depth=depth).to(device)
oldmodel = nn.DataParallel(oldmodel, device_ids=[0,1])
oldmodel.load_state_dict(torch.load('./results_' + str(depth) + '_' + dataset + '/1_compatibility_3_100.pth'))

model = step1cordsnetfolded(dataset=dataset,depth=depth).to(device)
model = nn.DataParallel(model, device_ids=[0,1])
modeldict = model.state_dict()

def extract(bnname,oldoriginal0,oldoriginal1):
    bn_eps  = 1e-05
    bn_bias = oldmodel.state_dict()[bnname + '.bias']
    bn_w    = oldmodel.state_dict()[bnname + '.weight']
    bn_mean = oldmodel.state_dict()[bnname + '.running_mean']
    bn_var  = oldmodel.state_dict()[bnname + '.running_var']
    original0 = oldoriginal0/torch.sqrt(bn_eps + bn_var[:,None,None,None]) 
    original0 *= bn_w[:,None,None,None]
    bias = -bn_mean / torch.sqrt(bn_eps + bn_var) * bn_w + bn_bias
    original1 = oldoriginal1/torch.norm_except_dim(oldoriginal1)
    return original0, original1, bias

inpbias = applybn(modeldict,oldmodel,'inp_conv','inp_norm')
inpbias = inpbias[:,None,None] * torch.ones(64,112,112).to(device)
inpbias = torch.nn.functional.avg_pool2d(inpbias,kernel_size=3, stride=2, padding=1, ceil_mode=False)
for i in range(depth):
    modeldict['module.area_bias.' + str(i)] += applybn(modeldict,oldmodel,'area_area.' + str(i),'area_norm.' + str(i))[:,None,None]
    modeldict['module.conv_bias.' + str(i)] += applybn(modeldict,oldmodel,'area_conv.' + str(i),'area_nrm2.' + str(i))[:,None,None]

lengths = [28,14,7]
for i in range(blockdepth):
    skipbias = applybn(modeldict,oldmodel,'skip_area.' + str(i),'skip_norm.' + str(i))[:,None,None]
    skipbias = skipbias*torch.ones(1,lengths[i],lengths[i]).to(device)
    modeldict['module.area_bias.' + str(3+2*i)] += torch.nn.functional.conv2d(skipbias,modeldict[m+'area_area.'+str(3+2*i)+o+'0']*modeldict[m+'area_area.'+str(3+2*i)+o+'1'],padding=1)

modeldict[m+'inp_skip'+o+'0'] = oldmodel.state_dict()[m+'inp_skip'+o+'0']
modeldict[m+'inp_skip'+o+'1'] = oldmodel.state_dict()[m+'inp_skip'+o+'1']/torch.norm_except_dim(oldmodel.state_dict()[m+'inp_skip'+o+'1'])

inpbias1 = torch.nn.functional.conv2d(inpbias,modeldict[m+'area_area.0'+o+'0']*modeldict[m+'area_area.0'+o+'1'],padding=1)
modeldict['module.area_bias.0'] += inpbias1 

inpbias2 = torch.nn.functional.conv2d(inpbias,modeldict[m+'inp_skip'+o+'0']*modeldict[m+'inp_skip'+o+'1'],padding=1)
inpbias2 = torch.nn.functional.conv2d(inpbias2,modeldict[m+'area_area.1'+o+'0']*modeldict[m+'area_area.1'+o+'1'],padding=1)
modeldict['module.area_bias.1'] += inpbias2

modeldict[m+'out_fc.bias']  = oldmodel.state_dict()[m+'out_fc.bias'] 
modeldict[m+'out_fc'+o+'0'] = oldmodel.state_dict()[m+'out_fc'+o+'0']
modeldict[m+'out_fc'+o+'1'] = oldmodel.state_dict()[m+'out_fc'+o+'1']

model.load_state_dict(modeldict)
torch.save(model.state_dict(), './results_' + str(depth) + '_' + dataset + '/1_compatibility_fused.pth')

# --------------------------------------------------------------------------------------

batch_size = 256
num_workers= 32
dataset_test = load_dataset(dataset,'test')
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
train_size, test_size = get_size(dataset)

with torch.no_grad():
    model.eval()
    oldmodel.eval()

    correct = 0.
    correct_old = 0.
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        current_batch_size = inputs.size()[0]
        inputs = inputs.to(device)
        labels = labels.to(device)
        x0 = []
        for j in range(depth):
            x0.append(torch.zeros(current_batch_size,channels[j],sizes[j],sizes[j]).to(device))

        outputsold, x1old = oldmodel([x0[i] for i in range(depth)], inputs)
        correct_old += (outputsold.argmax(1) == labels).sum().item()/test_size

        outputs, x1 = model([x0[i] for i in range(depth)], inputs)
        correct += (outputs.argmax(1) == labels).sum().item()/test_size

    print(f'testacc: {correct:.5f}, testaccold: {correct_old:.5f}')


