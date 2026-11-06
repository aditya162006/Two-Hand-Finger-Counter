import cv2 as computervision2
import mediapipe
import time

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Set up the hand tracker. By default, it looks for up to two hands.
        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )

        # We use this to draw the dots and lines on the hand.
        self.mpDraw = mediapipe.solutions.drawing_utils

        # This will hold the result for the latest picture.
        self.results = None

    def findHands(self, img, draw=True):
        # OpenCV reads colors in BGR order. MediaPipe needs them in RGB order.
        rgbPicture = computervision2.cvtColor(img, computervision2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgbPicture)

        # Draw the dots and lines on each hand we find.
        if draw and self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        # Call findHands first so we have a picture to get the points from.
        if self.results is None or not self.results.multi_hand_landmarks:
            return []

        hands = self.results.multi_hand_landmarks

        # Hand 0 is the first hand. Hand 1 is the second hand.
        if handNo < 0 or handNo >= len(hands):
            return []

        hand = hands[handNo]
        height, width = img.shape[:2]
        points = []

        for pointNumber, point in enumerate(hand.landmark):
            # Turn the position into pixels. For example, x = 0.5 means halfway across.
            x = int(point.x * width)
            y = int(point.y * height)

            # Save each point as [point number, x, y].
            points.append([pointNumber, x, y])

            if draw:
                computervision2.circle(img, (x, y), 1, (255, 0, 255), -1)
        return points

def main():
    # Open the webcam to try the hand tracker on its own.
    camera = computervision2.VideoCapture(0)
    detector = handDetector()
    previousTime = time.perf_counter()

    while True:
        success, picture = camera.read()
        if not success:
            print("Could not read the webcam.")
            break

        # Flip the picture so it looks like a mirror.
        picture = computervision2.flip(picture, 1)
        picture = detector.findHands(picture)
        points = detector.findPosition(picture)

        # Point 4 is the tip of the thumb. Point 8 is the tip of the index finger.
        if points:
            print("Thumb tip:", points[4])

        # Work out how many pictures the camera is showing per second.
        currentTime = time.perf_counter()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        computervision2.putText(
            picture, f"FPS: {int(fps)}", (10, 70),
            computervision2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        computervision2.imshow("Hand Tracker", picture)
        if computervision2.waitKey(1) & 0xFF == ord("q"):
            break

    # Turn off the webcam and close the window.
    camera.release()
    computervision2.destroyAllWindows()


# Run the webcam demo only when we start this file directly.
if __name__ == "__main__":
    main()
