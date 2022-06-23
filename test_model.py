from test_data_prep import test_taskloader
import torch
from tqdm import tqdm
import einops
from utils import change_lab,label_rearrange,DCE
from torchmetrics import ConfusionMatrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Initialize tensors for performance metrics
n_correct = 0
n_wrong = 0
Dice_score1 =torch.empty(len(test_taskloader))
Dice_score2 =torch.empty(len(test_taskloader))
Dice_score3 =torch.empty(len(test_taskloader))
Acc_out =torch.empty(len(test_taskloader))
Acc_tar =torch.empty(len(test_taskloader))

# load trained model
model =torch.load('model_framework21.pth')

model = model.to(device)

model.eval()  # model in eval mode


i= 0
loop = tqdm(enumerate(test_taskloader), total=len(test_taskloader), leave=False)
for batch_idx, (images, lung_lab, lesion_lab) in loop:

    torch.cuda.empty_cache()
    images = torch.unsqueeze(images, dim=1)
    lung_lab = torch.unsqueeze(lung_lab, dim=1)
    lesion_lab = torch.unsqueeze(lesion_lab, dim=1)

    if images.size()[0] == 1:
        images = einops.rearrange(images, 'c b h w -> b c h w')
        lung_lab = einops.rearrange(lung_lab, 'c b h w -> b c h w')
        lesion_lab = einops.rearrange(lesion_lab, 'c b h w -> b c h w')

    target_seg = label_rearrange(lung_lab, lesion_lab)
    target_class = change_lab(target_seg)


    images = images.to(device).type(torch.cuda.FloatTensor)
    target_seg = target_seg.to(device).type(torch.cuda.FloatTensor)
    target_class = target_class.to(device).type(torch.cuda.FloatTensor)

    with torch.no_grad():
        class_out, seg_out,_,_,_ = model(images)


    DS1 = DCE(seg_out[:, 0, :, :], target_seg[:, 0, :, :])
    DS2 = DCE(seg_out[:, 1, :, :], target_seg[:, 1, :, :])
    DS3 = DCE(seg_out[:, 2, :, :], target_seg[:, 2, :, :])
    Dice_score1[i] = DS1.item()
    Dice_score2[i] = DS2.item()
    Dice_score3[i] = DS3.item()

    Cl_out = torch.sigmoid(class_out)
    if torch.argmax(Cl_out) == torch.argmax(target_class):
        n_correct += 1
    else:
        n_wrong += 1

    Acc_out[i] = torch.argmax(Cl_out)
    Acc_tar[i] = torch.argmax(target_class)

    i += 1

acc = (n_correct * 1.0) / (n_correct+n_wrong)

print("Acc:", acc, "Dice1:", torch.mean(Dice_score1), "Dice2:", torch.mean(Dice_score2), "Dice3:",
      torch.mean(Dice_score3))


# Confusion matrix
preds = Acc_out.type(torch.IntTensor)
targ = Acc_tar.type(torch.IntTensor)
confmat = ConfusionMatrix(num_classes=3)
confmat(preds, targ)
print(confmat(preds, targ))



