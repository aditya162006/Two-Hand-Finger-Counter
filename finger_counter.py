# Hold one or both hands where the webcam can see your fingers.
import cv2 as computervision2
import time
import math
import HandTrackingModule as handtracking

# Set the webcam picture size to 640 by 480 pixels.
cameraWidth, cameraHeight = 640, 480
camera = computervision2.VideoCapture(0)
camera.set(3, cameraWidth)
camera.set(4, cameraHeight)

previousTime = 0
detector = handtracking.handDetector(maxHands=2, detectionCon=0.75)

def countFingers(points):
    # Use the line from the pinky base to the index base as the thumb direction.
    indexBase = points[5][1:]
    pinkyBase = points[17][1:]
    thumbTip = points[4][1:]
    palmX = indexBase[0] - pinkyBase[0]
    palmY = indexBase[1] - pinkyBase[1]
    palmWidth = math.hypot(palmX, palmY)
    total = 0

    if palmWidth > 0:
        # Measure how far the thumb sticks past the index side as the hand turns.
        thumbX = thumbTip[0] - indexBase[0]
        thumbY = thumbTip[1] - indexBase[1]
        thumbReach = (thumbX * palmX + thumbY * palmY) / palmWidth
        if thumbReach > 0.3 * palmWidth:
            total += 1

    for tip in [8, 12, 16, 20]:
        base = points[tip - 3][1:]
        joint = points[tip - 2][1:]
        nextJoint = points[tip - 1][1:]
        fingertip = points[tip][1:]

        # On a straight finger, the direct distance is close to the length along the joints.
        reach = math.dist(base, fingertip)
        firstPart = math.dist(base, joint)
        length = firstPart + math.dist(joint, nextJoint) + math.dist(nextJoint, fingertip)
        # A curled finger can look straight when its joints overlap on camera.
        # Its tip should also reach farther from the wrist than its knuckle.
        wrist = points[0][1:]
        palmLength = math.dist(wrist, base)
        tipFromWrist = math.dist(wrist, fingertip)
        reachesPastKnuckle = tipFromWrist > palmLength + 0.25 * firstPart
        if reachesPastKnuckle and reach > 0.85 * length and reach > 1.3 * firstPart:
            total += 1
    return total


while True:
    success, picture = camera.read()
    if not success:
        break
    picture = computervision2.flip(picture, 1)
    picture = detector.findHands(picture)

    # Add up the fingers on every hand found in this picture.
    hands = detector.results.multi_hand_landmarks
    if hands:
        totalFingers = 0
        for handNumber in range(len(hands)):
            points = detector.findPosition(picture, handNo=handNumber, draw=False)
            totalFingers += countFingers(points)

        print(totalFingers)

        # Draw a green box and write the finger count inside it.
        computervision2.rectangle(
            picture, (20, 225), (170, 425), (0, 255, 0), computervision2.FILLED
        )
        computervision2.putText(
            picture, str(totalFingers), (25, 365),
            computervision2.FONT_HERSHEY_PLAIN, 7, (255, 0, 0), 12
        )

    # Use the time between two frames to work out the frames per second.
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    computervision2.putText(
        picture, f"FPS: {int(fps)}", (400, 70),
        computervision2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    # Show the webcam picture and give the window time to update.
    computervision2.imshow("Image", picture)
    computervision2.waitKey(1)