import math
import numpy as np
import cv2
from skimage.draw import line

# Changed iterations on line 12 from 20 to 7
# Changed erosion kernel size on line 16 from 55,55 to 35,35
# Changed line width on line 230 from 3 to 1
# Changed bounds at 245 from 20000 and 0.9 to 3000 and 0.6

class PathingProcessor():

    def calculatePathingAreaFromTraversableArea(self, traversabilityImage):

        #pathingAreaImage = cv2.erode(image, np.ones((55, 55), np.uint8))

        pathingAreaImage2 = cv2.morphologyEx(traversabilityImage, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=7)
        pathingAreaImage2 = cv2.erode(pathingAreaImage2, np.ones((35, 35), np.uint8))

        return pathingAreaImage2

    def straightLinePathing(self, traversabilityImage, combinedTraversabilityAndPathingAreaImage, numLabels, stats):
        pathingAreaValue = 2
        pathValue = 3

        completePaths = 0
        combinedPathLengths = 0
        unsafePathPixels = 0

        for i in range(1, numLabels):
            leftBound = stats[i, cv2.CC_STAT_LEFT]
            topBound = stats[i, cv2.CC_STAT_TOP]
            rightBound = leftBound + stats[i, cv2.CC_STAT_WIDTH]
            bottomBound = topBound + stats[i, cv2.CC_STAT_HEIGHT]

            # Calculate
            if stats[i, cv2.CC_STAT_AREA] >= 20000 and bottomBound > combinedTraversabilityAndPathingAreaImage.shape[0] * 0.9:
                startY = bottomBound - 1
                endY = topBound
                positivePositionSum = 0
                positiveCount = 0

                # Find starting pixel
                for xPosition in range(leftBound, rightBound):
                    if combinedTraversabilityAndPathingAreaImage[startY][xPosition] == pathingAreaValue:
                        positivePositionSum += xPosition
                        positiveCount += 1

                if positiveCount != 0:
                    startX = math.floor(positivePositionSum / positiveCount)

                else:
                    continue

                #while image[startY][startX] != pathingAreaValue:
                #    startX = startX - 1

                positivePositionSum = 0
                positiveCount = 0

                # Find ending pixel
                for xPosition in range(leftBound, rightBound):
                    if combinedTraversabilityAndPathingAreaImage[endY][xPosition] == pathingAreaValue:
                        positivePositionSum += xPosition
                        positiveCount += 1

                if positiveCount != 0:
                    endX = math.floor(positivePositionSum / positiveCount)

                else:
                    continue

                #while image[endY][endX] != pathingAreaValue:
                #    endX = endX - 1

                xCoords, yCoords = line(startX, startY, endX, endY)

                for xCoord, yCoord in zip(xCoords, yCoords):
                    combinedPathLengths += 1

                    if combinedTraversabilityAndPathingAreaImage[yCoord][xCoord] != pathingAreaValue:
                        unsafePathPixels += 1

                    combinedTraversabilityAndPathingAreaImage[yCoord][xCoord] = pathValue

                completePaths += 1

        return combinedTraversabilityAndPathingAreaImage, completePaths, combinedPathLengths, unsafePathPixels


    def MyAlgPathing(self, image, numLabels, stats):
        pathingAreaValue = 2
        pathValue = 3

        completePaths = 0
        combinedPathLengths = 0
        unsafePathPixels = 0

        for i in range(1, numLabels):
            leftBound = stats[i, cv2.CC_STAT_LEFT]
            topBound = stats[i, cv2.CC_STAT_TOP]
            rightBound = leftBound + stats[i, cv2.CC_STAT_WIDTH]
            bottomBound = topBound + stats[i, cv2.CC_STAT_HEIGHT]

            averagePositivePosition = 0
            previousPosition = 0
            previousShift = 0
            j = 0

            #Calculate
            if stats[i, cv2.CC_STAT_AREA] >= 20000 and bottomBound > image.shape[0] * 0.9:
                completePaths += 1

                for yPosition in range(bottomBound - 1, topBound, -1):
                    positivePositionSum = 0
                    positiveCount = 0

                    for xPosition in range(leftBound, rightBound):
                        if image[yPosition][xPosition] == pathingAreaValue:
                            positivePositionSum += xPosition
                            positiveCount += 1

                    if positiveCount != 0:
                        averagePositivePosition = math.floor(positivePositionSum / positiveCount)

                        #Check that selected point is in the pathing region
                        if image[yPosition][averagePositivePosition] == pathingAreaValue:
                            #First point of path
                            if j == 0:
                                image[yPosition][averagePositivePosition] = pathValue
                                combinedPathLengths += 1

                            #Line connecting point to next
                            else:
                                direction = 1 if averagePositivePosition >= previousPosition else -1
                                for k in range(previousPosition, averagePositivePosition + direction, direction):
                                    if image[yPosition][k] != pathingAreaValue:
                                        unsafePathPixels += 1
                                        image[yPosition][k] = 5
                                    else:
                                        image[yPosition][k] = pathValue

                                #cv2.line(image, (previousPosition, yPosition), (averagePositivePosition, yPosition), pathValue, 1)
                                combinedPathLengths += abs(previousPosition - averagePositivePosition)

                            previousPosition = averagePositivePosition
                            previousShift = 0

                        #If it is not, shift it into the pathing region
                        else:
                            #Shift left if previous shift was left
                            if previousShift == 1:
                                leftOffset = 1

                                while averagePositivePosition - leftOffset >= leftBound and image[yPosition][averagePositivePosition - leftOffset] != pathingAreaValue:
                                    leftOffset += 1

                                cv2.line(image, (previousPosition, yPosition), (averagePositivePosition - leftOffset, yPosition), pathValue, 1)
                                combinedPathLengths += abs(previousPosition - (averagePositivePosition - leftOffset))
                                previousPosition = averagePositivePosition - leftOffset

                            #Shift right if previous shift was right
                            elif previousShift == 2:
                                rightOffset = 1

                                while averagePositivePosition + rightOffset <= rightBound and image[yPosition][averagePositivePosition + rightOffset] != pathingAreaValue:
                                    rightOffset += 1

                                cv2.line(image, (previousPosition, yPosition), (averagePositivePosition + rightOffset, yPosition), pathValue, 1)
                                combinedPathLengths += abs(previousPosition - (averagePositivePosition + rightOffset))
                                previousPosition = averagePositivePosition + rightOffset

                            #Calculate shortest shift in pixels and shift there
                            else:
                                leftOffset = 1
                                rightOffset = 1
                                while averagePositivePosition - leftOffset >= leftBound and image[yPosition][averagePositivePosition - leftOffset] != pathingAreaValue:
                                    leftOffset += 1

                                while averagePositivePosition + rightOffset <= rightBound and image[yPosition][averagePositivePosition + rightOffset] != pathingAreaValue:
                                    rightOffset += 1

                                leftDistance = abs((averagePositivePosition - leftOffset) - previousPosition)
                                rightDistance = abs((averagePositivePosition + rightOffset) - previousPosition)

                                if leftDistance < rightDistance:
                                    if j == 0:
                                        image[yPosition][averagePositivePosition - leftOffset] = pathValue
                                        combinedPathLengths += 1

                                    else:
                                        cv2.line(image, (previousPosition, yPosition), (averagePositivePosition - leftOffset, yPosition), pathValue, 1)
                                        combinedPathLengths += abs(previousPosition - (averagePositivePosition - leftOffset))

                                    previousPosition = averagePositivePosition - leftOffset
                                    previousShift = 1

                                else:
                                    if j == 0:
                                        image[yPosition][averagePositivePosition + rightOffset] = pathValue
                                        combinedPathLengths += 1

                                    else:
                                        cv2.line(image, (previousPosition, yPosition), (averagePositivePosition + rightOffset, yPosition), pathValue, 1)
                                        combinedPathLengths += abs(previousPosition - (averagePositivePosition + rightOffset))

                                    previousPosition = averagePositivePosition + rightOffset
                                    previousShift = 2

                    j += 1

        return image, completePaths, combinedPathLengths, unsafePathPixels

    class AStarNode():
        """A node class for A* Pathfinding"""

        def __init__(self, parent=None, position=None, g=0, h=0, f=0):
            self.parent = parent
            self.position = position

            self.g = 0
            self.h = 0
            self.f = 0

        def __eq__(self, otherNode):
            return self.position == otherNode.position

    def AStarPathing(self, image, numLabels, stats):
        imgHeight = image.shape[0]
        imgWidth = image.shape[1]
        lineWidth = 3

        pathingAreaValue = 2
        pathValue = 3

        completePaths = 0
        combinedPathLengths = 0

        for i in range(1, numLabels):
            leftBound = stats[i, cv2.CC_STAT_LEFT]
            topBound = stats[i, cv2.CC_STAT_TOP]
            rightBound = leftBound + stats[i, cv2.CC_STAT_WIDTH]
            bottomBound = topBound + stats[i, cv2.CC_STAT_HEIGHT]

            # print(
            #     "Component:", i,
            #     "Area:", stats[i, cv2.CC_STAT_AREA],
            #     "BottomBound:", bottomBound,
            #     "ImageHeight:", image.shape[0]
            # )

            #Calculate
            if stats[i, cv2.CC_STAT_AREA] >= 3000 and bottomBound > image.shape[0] * 0.6:
                startY = bottomBound - 1
                endY = topBound
                positivePositionSum = 0
                positiveCount = 0

                """
                #Find starting pixel
                for xPosition in range(leftBound, rightBound):
                    if image[startY][xPosition] == pathingAreaValue:
                        positivePositionSum += xPosition
                        positiveCount += 1

                if positiveCount != 0:
                    startX = math.floor(positivePositionSum / positiveCount)

                else:
                    continue
                """

                startX = math.floor(image.shape[1] / 2)

                closestDistance = 999999
                bestX = None

                for xPosition in range(leftBound, rightBound):

                    if image[startY][xPosition] == pathingAreaValue:

                        distance = abs(xPosition - image.shape[1] // 2)

                        if distance < closestDistance:
                            closestDistance = distance
                            bestX = xPosition

                if bestX is None:
                    continue

                startX = bestX

                positivePositionSum = 0
                positiveCount = 0

                #Find ending pixel
                for xPosition in range(leftBound, rightBound):
                    if image[endY][xPosition] == pathingAreaValue:
                        positivePositionSum += xPosition
                        positiveCount += 1

                if positiveCount != 0:
                    endX = math.floor(positivePositionSum / positiveCount)

                else:
                    continue

                while image[endY][endX] != pathingAreaValue:
                    endX = endX - 1

                #Begin A* Pathing - Taken from https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
                """Returns a list of tuples as a path from the given start to the given end in the given maze"""

                # Create start and end node
                start_node = self.AStarNode(None, (startX, startY), 0, 0, 0)
                end_node = self.AStarNode(None, (endX, endY), 0, 0, 0)

                # Initialize both open and closed list
                open_list = []
                closed_list = []

                # Add the start node
                open_list.append(start_node)

                iterations = 0
                maxIterations = 5000

                # Loop until you find the end
                while len(open_list) > 0:
                    if iterations >= maxIterations:
                        break

                    # Get the current node
                    current_node = open_list[0]
                    current_index = 0

                    for index, item in enumerate(open_list):
                        if item.f < current_node.f:
                            current_node = item
                            current_index = index

                    # Pop current off open list, add to closed list
                    open_list.pop(current_index)
                    closed_list.append(current_node)

                    # Found the goal
                    if current_node == end_node:
                        current = current_node
                        previousPos = (current.position[0], current.position[1])
                        completePaths += 1

                        while current is not None:
                            image[current.position[1]][current.position[0]] = pathValue

                            #Left diagonal move
                            if previousPos[0] != current.position[0] and previousPos[1] != current.position[0]:
                                for widthIncrement in range(1, lineWidth):
                                    #if current.position[1] + widthIncrement < imgHeight and current.position[0] - widthIncrement > 0:
                                    #    image[current.position[1] + widthIncrement][current.position[0] - widthIncrement] = pathValue

                                        #if current.position[1] + widthIncrement + 1 < imgHeight:
                                        #    image[current.position[1] + widthIncrement + 1][current.position[0] - widthIncrement] = pathValue

                                    #if current.position[1] - widthIncrement > 0 and current.position[0] + widthIncrement < imgWidth:
                                    #    image[current.position[1] - widthIncrement][current.position[0] + widthIncrement] = pathValue

                                        #if current.position[0] + widthIncrement + 1 < imgWidth:
                                        #    image[current.position[1] - widthIncrement][current.position[0] + widthIncrement + 1] = pathValue

                                    if current.position[0] - widthIncrement > 0:
                                        image[current.position[1]][current.position[0] - widthIncrement] = pathValue

                                    if current.position[0] + widthIncrement < imgWidth:
                                        image[current.position[1]][current.position[0] + widthIncrement] = pathValue

                                    if current.position[1] + widthIncrement < imgHeight:
                                        image[current.position[1] + widthIncrement][current.position[0]] = pathValue

                                    if current.position[1] - widthIncrement > 0:
                                        image[current.position[1] - widthIncrement][current.position[0]] = pathValue

                            #Vertical move
                            elif previousPos[1] != current.position[1]:
                                for widthIncrement in range(1, lineWidth):
                                    if current.position[0] - widthIncrement > 0:
                                        image[current.position[1]][current.position[0] - widthIncrement] = pathValue

                                    if current.position[0] + widthIncrement < imgWidth:
                                        image[current.position[1]][current.position[0] + widthIncrement] = pathValue

                            #Horizontal move
                            elif previousPos[0] != current.position[0]:
                                for widthIncrement in range(1, lineWidth):
                                    if current.position[1] - widthIncrement > 0:
                                        image[current.position[1] - widthIncrement][current.position[0]] = pathValue

                                    if current.position[1] + widthIncrement < imgHeight:
                                        image[current.position[1] + widthIncrement][current.position[0]] = pathValue

                            previousPos = (current.position[0], current.position[1])
                            current = current.parent
                            combinedPathLengths += 1

                        #break
                        return image, completePaths, combinedPathLengths

                    # Generate children
                    children = []
                    for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Adjacent squares

                        # Get node position
                        node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                        # Make sure within range
                        if node_position[0] > (rightBound - 1) or node_position[0] < 0 or node_position[1] > (bottomBound - 1) or node_position[1] < 0:
                            continue

                        # Make sure walkable terrain
                        if image[node_position[1]][node_position[0]] != pathingAreaValue:
                            continue

                        # Create new node
                        new_node = self.AStarNode(current_node, node_position)

                        # Append
                        children.append(new_node)

                    # Loop through children
                    for child in children:
                        childOnClosed = False
                        childOnOpen = False

                        # Child is on the closed list
                        for closed_child in closed_list:
                            if child == closed_child:
                                childOnClosed = True
                                break

                        if childOnClosed:
                            continue

                        # Create the f, g, and h values
                        child.g = current_node.g + 1
                        child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                        child.f = child.g + child.h

                        # Child is already in the open list
                        for open_node in open_list:
                            if child == open_node and child.g > open_node.g:
                                childOnOpen = True
                                break

                        if childOnOpen:
                            continue

                        # Add the child to the open list
                        open_list.append(child)

                    iterations += 1


        return image, completePaths, combinedPathLengths

    def MaxSafePathing(self, image, numLabels, stats):
        pathingAreaValue = 1
        pathValue = 2

        imgWidth = image.shape[1]

        allPaths = []
        maxOffset = 100

        completePaths = 0
        combinedPathLengths = 0

        for i in range(1, numLabels):
            leftBound = stats[i, cv2.CC_STAT_LEFT]
            topBound = stats[i, cv2.CC_STAT_TOP]
            rightBound = leftBound + stats[i, cv2.CC_STAT_WIDTH]
            bottomBound = topBound + stats[i, cv2.CC_STAT_HEIGHT]

            # Calculate
            if stats[i, cv2.CC_STAT_AREA] >= 3000 and bottomBound > image.shape[0] * 0.6:
                startY = bottomBound - 1
                endY = topBound
                positivePositionSum = 0
                positiveCount = 0

                # Find starting pixel
                for xPosition in range(leftBound, rightBound):
                    if image[startY][xPosition] == pathingAreaValue:
                        positivePositionSum += xPosition
                        positiveCount += 1

                if positiveCount != 0:
                    startX = math.floor(positivePositionSum / positiveCount)

                else:
                    continue

                while image[startY][startX] != pathingAreaValue:
                    startX = startX - 1

                positivePositionSum = 0
                positiveCount = 0

                # Find ending pixel
                for xPosition in range(leftBound, rightBound):
                    if image[endY][xPosition] == pathingAreaValue:
                        positivePositionSum += xPosition
                        positiveCount += 1

                if positiveCount != 0:
                    endX = math.floor(positivePositionSum / positiveCount)

                else:
                    continue

                while image[endY][endX] != pathingAreaValue:
                    endX = endX - 1

                # Begin A* Pathing - Taken from https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
                """Returns a list of tuples as a path from the given start to the given end in the given maze"""

                # Create start and end node
                start_node = self.AStarNode(None, (startX, startY), 0, 0, 0)
                end_node = self.AStarNode(None, (endX, endY), 0, 0, 0)

                # Initialize both open and closed list
                open_list = []
                closed_list = []

                # Add the start node
                open_list.append(start_node)

                iterations = 0
                maxIterations = 2000

                # Loop until you find the end
                while len(open_list) > 0:
                    if iterations >= maxIterations:
                        break

                    # Get the current node
                    current_node = open_list[0]
                    current_index = 0

                    for index, item in enumerate(open_list):
                        if item.f < current_node.f:
                            current_node = item
                            current_index = index

                    # Pop current off open list, add to closed list
                    open_list.pop(current_index)
                    closed_list.append(current_node)

                    # Found the goal
                    if current_node == end_node:
                        newPath = []
                        current = current_node
                        completePaths += 1

                        while current is not None:
                            newPath.append((current.position[0], current.position[1]))
                            current = current.parent
                            combinedPathLengths += 1

                        allPaths.append(newPath)
                        break

                    # Generate children
                    children = []
                    for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1),(1, 1)]:  # Adjacent squares

                        # Get node position
                        node_position = (
                        current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                        # Make sure within range
                        if node_position[0] > (rightBound - 1) or node_position[0] < 0 or node_position[1] > (bottomBound - 1) or node_position[1] < 0:
                            continue

                        # Make sure walkable terrain
                        if image[node_position[1]][node_position[0]] != pathingAreaValue:
                            continue

                        # Create new node
                        new_node = self.AStarNode(current_node, node_position)

                        # Append
                        children.append(new_node)

                    # Loop through children
                    for child in children:
                        childOnClosed = False
                        childOnOpen = False

                        # Child is on the closed list
                        for closed_child in closed_list:
                            if child == closed_child:
                                childOnClosed = True
                                break

                        if childOnClosed:
                            continue

                        # Create the f, g, and h values
                        child.g = current_node.g + 1
                        child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                        child.f = child.g + child.h

                        # Child is already in the open list
                        for open_node in open_list:
                            if child == open_node and child.g > open_node.g:
                                childOnOpen = True
                                break

                        if childOnOpen:
                            continue

                        # Add the child to the open list
                        open_list.append(child)

                    iterations += 1


        for path in allPaths:
            previousXPos = None
            for xPos, yPos in path:

                if previousXPos is None:
                    image[yPos][xPos] = pathValue
                    previousXPos = xPos
                    continue

                positionSum = xPos
                numPositions = 1

                positionSums = [xPos]

                offset = 1
                while image[yPos][xPos + offset] != 0 and (xPos + offset) < imgWidth and offset < maxOffset:
                    positionSum += (xPos + offset)
                    positionSums.append(xPos + offset)
                    numPositions += 1
                    offset += 1

                offset = 1
                while image[yPos][xPos - offset] != 0 and (xPos - offset) > 0 and offset < maxOffset:
                    positionSum += (xPos - offset)
                    positionSums.append(xPos - offset)
                    numPositions += 1
                    offset += 1

                maxSafeXPos = positionSum / numPositions

                print(maxSafeXPos, xPos)
                print(positionSums)

                if maxSafeXPos > xPos:
                    image[yPos][previousXPos + 1] = pathValue
                    previousXPos = previousXPos + 1

                elif maxSafeXPos < xPos:
                    image[yPos][previousXPos - 1] = pathValue
                    previousXPos = previousXPos - 1

                else:
                    image[yPos][previousXPos] = pathValue

        return image, completePaths, combinedPathLengths