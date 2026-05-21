
import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
from scipy.ndimage import binary_fill_holes
from torchvision.transforms import InterpolationMode
from PathingProcessing import PathingProcessor
from AirSimDataset import AirSimData
import pandas as pd

# Type can be
# StraightLine
# MyAlg
# AStar
# MaxSafe
pathingType = "AStar"

pathingProcessor = PathingProcessor()
# ================================
# Safety Margin
# ================================

SAFETY_KERNEL_SIZE = 7


# ================================
# Kernel Builder
# ================================

def get_kernel(kernelShape, kernelSize):

    if kernelShape == "square":

        return np.ones(
            (kernelSize, kernelSize),
            np.uint8
        )

    elif kernelShape == "circle":

        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernelSize, kernelSize)
        )


# ================================
# Safe Class-wise Morphology
# ================================

def apply_morphology(
    mask,
    morphOperation,
    kernelShape,
    kernelSize,
    applyErosion,
    applyImfill
):

# Applies morph operations using the parameters defined above
    kernel = get_kernel(kernelShape, kernelSize)

    refinedMask = np.zeros_like(mask)

    for c in range(numClasses):

        classMask = (mask == c).astype(np.uint8)

        if morphOperation == "open":
            classMask = cv2.morphologyEx(
                classMask,
                cv2.MORPH_OPEN,
                kernel
            )

        elif morphOperation == "close":
            classMask = cv2.morphologyEx(
                classMask,
                cv2.MORPH_CLOSE,
                kernel
            )

        elif morphOperation == "openclose":

            classMask = cv2.morphologyEx(
                classMask,
                cv2.MORPH_OPEN,
                kernel
            )

            classMask = cv2.morphologyEx(
                classMask,
                cv2.MORPH_CLOSE,
                kernel
            )

        if applyErosion:
            classMask = cv2.erode(classMask, kernel)

        if applyImfill:
            classMask = binary_fill_holes(classMask).astype(np.uint8)

        refinedMask[classMask == 1] = c

    return refinedMask


 # Start of code
# ================================
# Dataset Setup
# ================================

dataset = "AirSimData"
modelset = "AirSimData"

modelSavesPath = "C:/Python/PyTorchSegmentation/ModelSaves/"
segmentationsPath = "C:/Python/PyTorchSegmentation/Segmentations/"

preFolder = os.path.join(segmentationsPath,"pre")
morphFolder = os.path.join(segmentationsPath,"morph")
pathFolder = os.path.join(segmentationsPath,"path")

os.makedirs(preFolder,exist_ok=True)
os.makedirs(morphFolder,exist_ok=True)
os.makedirs(pathFolder,exist_ok=True)

imageResize = (256,512)

device = torch.device("cpu") # Define device

normalizeTransform = transforms.Compose([
    transforms.Resize(imageResize),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

resizeTransform = transforms.Compose([
    transforms.Resize(imageResize, interpolation=InterpolationMode.NEAREST),
])


# path definitions for image and model locations

datasetPath = "C:/Python/PyTorchSegmentation/AirSimData/images/"
numClasses = 5

testDataset = AirSimData(
    datasetPath,
    split="test",
    transform=normalizeTransform, target_transform= resizeTransform
)

testLoader = DataLoader(
    testDataset,
    batch_size=1,
    shuffle=False
)


# ================================
# Load Model
# ================================

segmentationModel = models.segmentation.deeplabv3_resnet101(pretrained=True)
segmentationModel.classifier = DeepLabHead(2048, numClasses)

segmentationModel.load_state_dict(
    torch.load(
        modelSavesPath +
        "DeeplabV3" +
        modelset +
        "-9-0.8561012744903564.pth",
        map_location=device
    )
)

segmentationModel.eval()
segmentationModel.to(device)


# ================================
# Color Lookup Table
# ================================

color_table = np.array([
    [135, 206, 235],    # Sky Blue
    [139, 69, 19],      # Brown
    [128, 128, 128],    # Grey
    [144, 238, 144],    # Light Green
    [255, 105, 180]     # Pink
], dtype=np.uint8)


# ================================
# Traversability Mapping
# ================================

traversabilityLookupTable = [
    [0,0,0],
    [0,0,255],
    [0,0,0],
    [0,0,0],
    [0,0,0]
]

for _ in range(256 - len(traversabilityLookupTable)):
    traversabilityLookupTable.append([0,0,0])

traversabilityLookupTable = np.array([traversabilityLookupTable], dtype=np.uint8)


# ================================
# IoU Counters
# ================================

rawIntersectionCounts = torch.zeros(numClasses)
rawUnionCounts = torch.zeros(numClasses)

morphIntersectionCounts = torch.zeros(numClasses)
morphUnionCounts = torch.zeros(numClasses)


# ================================
# Path Metrics
# ================================

cumulativePathingCalculationTime = 0
totalPathLength = 0
totalNumPaths = 0

overallStart = time.time()
imagesTested = 0

kernelSizes = [5]

kernelShapes = [
    "square"
]

morphOperations = [
    "openclose"
]

erosionOptions = [
    False
]

imfillOptions = [
    False
]
results = []
precomputedData = []

# ================================
# PRECOMPUTE SEGMENTATIONS
# ================================

print("\n====================================")
print("PRECOMPUTING SEGMENTATIONS")
print("====================================")

for i, (testBatch, targetBatch) in enumerate(testLoader):

    testBatch = testBatch.to(device)

    with torch.no_grad():
        outputBatch = segmentationModel(testBatch)["out"]

    prediction = outputBatch.argmax(1)[0].cpu()

    target = targetBatch[0].cpu().long()

    rawMask = prediction.numpy()

    filename = os.path.splitext(
        testDataset.imagesList[i]
    )[0]

    # SAVE PRE SEGMENTATION ONCE
    rgbRawMask = color_table[rawMask]

    plt.imsave(
        os.path.join(
            preFolder,
            filename + "_seg_pre.png"
        ),
        rgbRawMask
    )

    precomputedData.append({

        "prediction": prediction,

        "target": target,

        "rawMask": rawMask,

        "filename": filename
    })

    print("Precomputed:", filename)


for kernelSize in kernelSizes:

    for kernelShape in kernelShapes:

        for morphOperation in morphOperations:

            for applyErosion in erosionOptions:

                for applyImfill in imfillOptions:

                    print("\n====================================")
                    print("STARTING NEW EXPERIMENT")
                    print("====================================")

                    print("Operation:", morphOperation)
                    print("Kernel Shape:", kernelShape)
                    print("Kernel Size:", kernelSize)
                    print("Erosion:", applyErosion)
                    print("Imfill:", applyImfill)

                    experimentName = (
                        f"{morphOperation}_"
                        f"{kernelShape}_"
                        f"{kernelSize}_"
                        f"E{applyErosion}_"
                        f"F{applyImfill}"
                    )

                    experimentMorphFolder = os.path.join(
                        morphFolder,
                        experimentName
                    )

                    experimentPathFolder = os.path.join(
                        pathFolder,
                        experimentName
                    )

                    os.makedirs(
                        experimentMorphFolder,
                        exist_ok=True
                    )

                    os.makedirs(
                        experimentPathFolder,
                        exist_ok=True
                    )

                    # RESET METRICS FOR THIS EXPERIMENT

                    rawIntersectionCounts = torch.zeros(numClasses)
                    rawUnionCounts = torch.zeros(numClasses)

                    morphIntersectionCounts = torch.zeros(numClasses)
                    morphUnionCounts = torch.zeros(numClasses)

                    cumulativePathingCalculationTime = 0

                    totalPathLength = 0
                    totalNumPaths = 0

                    imagesTested = 0

                    overallStart = time.time()

                    # ================================
                    # Evaluation Loop
                    # ================================

                    for data in precomputedData:

                        individualStart = time.time()

                        prediction = data["prediction"]

                        target = data["target"]

                        rawMask = data["rawMask"]

                        filename = data["filename"]


                        # IoU BEFORE MORPH
                        for c in range(numClasses):

                            targetMask = (target == c)
                            predMask = (prediction == c)

                            inter = torch.logical_and(targetMask,predMask)
                            union = torch.logical_or(targetMask,predMask)

                            rawIntersectionCounts[c]+=torch.count_nonzero(inter)
                            rawUnionCounts[c]+=torch.count_nonzero(union)


                        # MORPHOLOGY
                        morphMask = apply_morphology(
                            rawMask.astype(np.uint8),
                            morphOperation,
                            kernelShape,
                            kernelSize,
                            applyErosion,
                            applyImfill
                        )
                        morphMaskTensor = torch.from_numpy(morphMask)


                        # POST MORPH SAVE
                        rgbMorphMask = color_table[morphMask]
                        plt.imsave(
                            os.path.join(
                                experimentMorphFolder,
                                filename + "_seg_morph.png"
                            ),
                            rgbMorphMask
                        )


                        # IoU AFTER MORPH
                        for c in range(numClasses):

                            targetMask=(target==c)
                            predMask=(morphMaskTensor==c)

                            inter=torch.logical_and(targetMask,predMask)
                            union=torch.logical_or(targetMask,predMask)

                            morphIntersectionCounts[c]+=torch.count_nonzero(inter)
                            morphUnionCounts[c]+=torch.count_nonzero(union)


                        # TRAVERSABILITY
                        segRGB=cv2.cvtColor(morphMask.astype(np.uint8),cv2.COLOR_GRAY2RGB)

                        traversabilityImage=cv2.LUT(segRGB,traversabilityLookupTable)
                        traversabilityImage=cv2.cvtColor(traversabilityImage,cv2.COLOR_RGB2GRAY)
                        traversabilityImage=cv2.threshold(traversabilityImage,1,1,cv2.THRESH_BINARY)[1]


                        # SAFETY MARGIN
                        kernel=np.ones((SAFETY_KERNEL_SIZE,SAFETY_KERNEL_SIZE),np.uint8)
                        traversabilityImage=cv2.erode(traversabilityImage,kernel)
                        h, w = traversabilityImage.shape


                        pathingStart = time.time()
                        pathingAreaImage = pathingProcessor.calculatePathingAreaFromTraversableArea(
                            traversabilityImage
                        )

                        combinedTraversabilityAndPathingAreaImage = (
                                traversabilityImage + pathingAreaImage
                        )

                        numLabels, labels, stats, centerPoints = \
                            cv2.connectedComponentsWithStats(
                                pathingAreaImage,
                                connectivity=4
                            )

                        if pathingType == "AStar":
                            pathingImage, numPaths, pathLength = \
                                pathingProcessor.AStarPathing(
                                    combinedTraversabilityAndPathingAreaImage,
                                    numLabels,
                                    stats
                                )
                            # print("Number of Paths:", numPaths)
                            # print("Path Length:", pathLength)

                        individualPathTime=time.time()-pathingStart

                        cumulativePathingCalculationTime += individualPathTime

                        totalPathLength += pathLength

                        totalNumPaths += numPaths

                        # =========================
                        # VISUALIZATION
                        # =========================

                        # IMAGE 1 — ALL TRAVERSABLE AREAS
                        allTraversable = np.zeros((h, w, 3), dtype=np.uint8)

                        allTraversable[traversabilityImage == 1] = [255, 255, 0]

                        plt.imsave(
                            os.path.join(experimentPathFolder, filename + "_all_traversable.png"),
                            allTraversable
                        )

                        # IMAGE 2 — PATHING AREA
                        # (safe/usable planner region)

                        pathingAreaVis = np.zeros((h, w, 3), dtype=np.uint8)

                        # planner-valid region
                        pathingAreaVis[pathingAreaImage > 0] = [255, 255, 0]

                        # traversable but excluded from planner region
                        removedMask = (
                                (traversabilityImage == 1) &
                                (pathingAreaImage == 0)
                        )

                        pathingAreaVis[removedMask] = [0, 200, 0]

                        plt.imsave(
                            os.path.join(experimentPathFolder, filename + "_pathing_area.png"),
                            pathingAreaVis
                        )

                        # IMAGE 3 — FINAL PATH RESULT
                        # already generated by PathingProcessor

                        plt.imsave(
                            os.path.join(experimentPathFolder, filename + "_path_map.png"),
                            np.uint8(pathingImage)
                        )

                        # ================================
                        # Final Metrics
                        # ================================

                        # rawClassIoU = rawIntersectionCounts / rawUnionCounts
                        # morphClassIoU = morphIntersectionCounts / morphUnionCounts
                        #
                        # rawIoU = torch.sum(rawIntersectionCounts) / torch.sum(rawUnionCounts)
                        # morphIoU = torch.sum(morphIntersectionCounts) / torch.sum(morphUnionCounts)
                        #
                        # print("\nRaw IoU per class:", rawClassIoU.tolist())
                        # print("Morph IoU per class:", morphClassIoU.tolist())
                        #
                        # print("\nRaw Overall IoU:", rawIoU.item())
                        # print("Morphology Overall IoU:", morphIoU.item())
                        #
                        # print("Improvement:", (morphIoU - rawIoU).item())
                        #
                        # individualTime=time.time()-individualStart
                        #
                        # print("Image:",filename)
                        # print("Path Length:",pathLength)
                        # print("Path Time:",individualPathTime)
                        # print("Total Image Time:",individualTime)
                        # print()

                        imagesTested+=1
                        print(imagesTested)



                    # ================================
                    # FINAL DATASET STATISTICS
                    # ================================

                    overallTime = time.time() - overallStart


                    # IoU calculations
                    rawClassIoU = rawIntersectionCounts / rawUnionCounts
                    morphClassIoU = morphIntersectionCounts / morphUnionCounts

                    rawIoU = torch.sum(rawIntersectionCounts) / torch.sum(rawUnionCounts)

                    morphIoU = (
                        torch.sum(morphIntersectionCounts) /
                        torch.sum(morphUnionCounts)
                    )


                    # Path statistics
                    if totalNumPaths != 0:

                        AvgPathLength = (
                            totalPathLength / totalNumPaths
                        )

                    else:
                        AvgPathLength = 0


                    if imagesTested != 0:

                        AvgRuntimePerImage = (
                            overallTime / imagesTested
                        )

                        AvgPathTimePerImage = (
                            cumulativePathingCalculationTime /
                            imagesTested
                        )

                    else:

                        AvgRuntimePerImage = 0
                        AvgPathTimePerImage = 0

                    results.append({

                        "Operation": morphOperation,

                        "Kernel Shape": kernelShape,

                        "Kernel Size": kernelSize,

                        "Erosion": applyErosion,

                        "Imfill": applyImfill,

                        "Morph IoU": morphIoU.item(),

                        "Average Path Length": AvgPathLength,

                        "Successful Paths": totalNumPaths,

                        "Path Success Rate":
                            totalNumPaths / imagesTested,

                        "Average Runtime Per Image":
                            AvgRuntimePerImage,

                        "Average Path Time":
                            AvgPathTimePerImage,

                        "Images Processed":
                            imagesTested
                    })

                    pd.DataFrame(results).to_csv(
                        "MorphologyExperimentResults.csv",
                        index=False
                    )

                    print("\n====================================")
                    print("EXPERIMENT COMPLETE")
                    print("====================================")

                    print("Morph IoU:", morphIoU.item())

                    print("Average Path Length:", AvgPathLength)

                    print("Average Runtime Per Image:",
                          AvgRuntimePerImage)

                    print("Average Path Time:",
                          AvgPathTimePerImage)

                    print("Images Processed:",
                          imagesTested)



resultsTable = pd.DataFrame(results)

print(resultsTable)

resultsTable.to_csv(
    "MorphologyExperimentResults.csv",
    index=False
)