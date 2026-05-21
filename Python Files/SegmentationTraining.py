import time
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch import nn
from torchvision import models, datasets, transforms
from AirSimDataset import AirSimData
from torchvision.transforms import InterpolationMode
#dataset = "TestImages"
dataset = "AirSimData"
#dataset = "Rellis3D"

modelSavesPath = "C:/Python/PyTorchSegmentation/ModelSaves/"

#General params
imageResize = (256, 512)
#imageResize = (400, 800)    #Big
batchSize = 4
epochs = 10

#Optimizer params
#0.01 for DeepLab, 0.001 for UNet
learningRate = 0.001
momentum = 0.9
weightDecay = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

standardTransform = transforms.Compose([
    transforms.ToTensor(),
])

resizeTransform = transforms.Compose([
    transforms.Resize(imageResize, interpolation=InterpolationMode.NEAREST),
])

#TODO Calculate proper normalization
normalizeTransform = transforms.Compose([
    transforms.Resize(imageResize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Dataset definition
datasetPath = ""
numClasses = 0
trainDataset, validationDataset = None, None

if dataset == "AirSimData":
    datasetPath = "C:/Python/PyTorchSegmentation/AirSimData/images/"
    numClasses = 5
    trainDataset = AirSimData(datasetPath, split="train", transform=normalizeTransform, target_transform=resizeTransform)
    validationDataset = AirSimData(datasetPath, split="val", transform=normalizeTransform, target_transform=resizeTransform)
else:
    print("Error: Please define a valid dataset")
    exit(0)

#Dataset samplers, train data is sampled randomly
trainSampler = torch.utils.data.RandomSampler(trainDataset)
validationSampler = torch.utils.data.SequentialSampler(validationDataset)

#Dataset loaders
trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, sampler=trainSampler, drop_last=True)
validationDataLoader = torch.utils.data.DataLoader(validationDataset, batch_size=1, sampler=validationSampler)

# Specific to UNet
# segmentationModel = smp.Unet(
#     encoder_name="resnet101",        # similar backbone to your current model
#     encoder_weights="imagenet",      # pretrained
#     in_channels=3,
#     classes=numClasses,
# )
#
# segmentationModel.to(device)
# segmentationModel.train()


# Specific to DeepLab
segmentationModel = models.segmentation.deeplabv3_resnet101(pretrained=True)
segmentationModel.classifier = DeepLabHead(2048, numClasses)
segmentationModel.train()
segmentationModel.to(device)

#Optimization parameters
#UNet
# optimizer = torch.optim.SGD(
#     segmentationModel.parameters(),
#     lr=learningRate,
#     momentum=momentum,
#     weight_decay=weightDecay
# )


modelTrainingParameters = [
     {"params": [p for p in segmentationModel.backbone.parameters() if p.requires_grad]},
     {"params": [p for p in segmentationModel.classifier.parameters() if p.requires_grad]}]

# #optimizer = torch.optim.SGD(segmentationModel.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)
optimizer = torch.optim.SGD(modelTrainingParameters, lr=learningRate, momentum=momentum, weight_decay=weightDecay)
# #optimizer = torch.optim.Adam(segmentationModel.parameters(), lr=learningRate, weight_decay=weightDecay)
learningRateScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(trainDataLoader) * epochs)) ** 0.9)
#learningRateScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

lossCalculator = nn.CrossEntropyLoss(ignore_index=255)

numBatches = len(trainDataLoader)
batchNum = 1
cumulativeLoss = 0
bestIoU = 0

#Train loop
startTime = time.time()

for epoch in range(epochs):
    segmentationModel.train()

    #Training Loop
    for imageBatch, targetBatch in trainDataLoader:
        if (batchNum % 10 == 0):
            print("Processing batch " + str(batchNum) + " / " + str(numBatches) + ".")

        #print("INPUT STATS:", imageBatch.min(), imageBatch.max())
        #print("TARGET STATS:", targetBatch.min(), targetBatch.max())

        targetBatch = targetBatch.squeeze(1).long()
        imageBatch, targetBatch = imageBatch.to(device), targetBatch.to(device)

        optimizer.zero_grad()

        #print("INPUT STATS:", imageBatch.min(), imageBatch.max())
        #print("TARGET STATS:", targetBatch.min(), targetBatch.max())

        #Unet Implement
        #outputBatch = segmentationModel(imageBatch)

        # DeepLab
        outputBatch = segmentationModel(imageBatch)["out"]

        #print("OUTPUT STATS:", outputBatch.min(), outputBatch.max())
        #print("OUTPUT ARGMAX STATS:", outputBatch.argmax(1).min(), outputBatch.argmax(1).max())

        #TODO Maybe use Aux loss, maybe not

        print("Target classes:", torch.unique(targetBatch))

        outputBatchPredictions = outputBatch.argmax(1)

        if device.type == "cpu":
            loss = lossCalculator(outputBatch, targetBatch.type(torch.LongTensor))
        else:
            loss = lossCalculator(outputBatch, targetBatch.type(torch.cuda.LongTensor))

        cumulativeLoss += loss.item()

        print("Batch Loss: " + '{:.6f}'.format(loss.item()), "Running Loss: " + '{:.6f}'.format(cumulativeLoss / (batchNum + (epoch * numBatches))))

        """
        imageBatch, targetBatch, outputBatchPredictions = imageBatch.to("cpu"), targetBatch.to("cpu"), outputBatchPredictions.to("cpu")

        #print(imageBatch.shape, targetBatch.shape, outputBatch.shape)

        #Display Images
        if batchNum == numBatches:
            for image, target, output in zip(imageBatch, targetBatch, outputBatchPredictions):
                print("In Loop Image", image.permute(1, 2, 0).shape)
                plt.imshow(image.permute(1, 2, 0))
                plt.show()

                print("In Loop Target", target.shape)
                plt.imshow(target)
                plt.show()

                print("In Loop Output.argmax", output.shape)
                plt.imshow(output)
                plt.show()
        """

        loss.backward()
        optimizer.step()
        learningRateScheduler.step()

        batchNum += 1

    print("Validating model...")
    segmentationModel.eval()

    classIntersectionCounts = torch.zeros(numClasses)
    classUnionCounts = torch.zeros(numClasses)

    #Validation Loop
    for validationBatch, targetBatch in validationDataLoader:
        validationBatch = validationBatch.to(device)

        with torch.no_grad():
            #outputBatch = segmentationModel(validationBatch)
            outputBatch = segmentationModel(validationBatch)["out"]

        outputBatchPredictions = outputBatch.argmax(1)

        targetBatch = targetBatch.long()

        validationImage, targetImage, outputImage = validationBatch[0].to("cpu"), targetBatch[0].to("cpu"), outputBatchPredictions[0].to("cpu")

        #Calculate IOUS
        for classIndex in range(numClasses):
            targetClassMask = (classIndex == targetImage)
            predictedClassMask = (classIndex == outputImage)

            classIntersection = torch.logical_and(targetClassMask, predictedClassMask)
            classUnion = torch.logical_or(targetClassMask, predictedClassMask)

            classIntersectionCounts[classIndex] += torch.count_nonzero(classIntersection).item()
            classUnionCounts[classIndex] += torch.count_nonzero(classUnion).item()

    overallIoU = (torch.sum(classIntersectionCounts) / torch.sum(classUnionCounts)).item()

    batchNum = 1

    torch.save(segmentationModel.state_dict(), modelSavesPath + "DeeplabV3" + dataset + "-" + str(epoch) + "-" + str(overallIoU) + ".pth")

totalTime = time.time() - startTime
totalTimeStr = str(datetime.timedelta(seconds=int(totalTime)))
print('Training time {}'.format(totalTimeStr))