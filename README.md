# Two-Hand Finger Counter

I built a Python webcam app that counts extended fingers across one or both hands, from 0 to 10. My hand-tracking module uses MediaPipe’s pretrained model to find the finger joints. The counter uses the positions of those joints to decide which fingers are open. For example, two thumbs pointing down should still count as two extended fingers.

## Project files

| File | What it does |
| --- | --- |
| `finger_counter.py` | Opens the webcam, counts the fingers on each visible hand, and adds the counts together. It displays the total in a green box and shows the frame rate (FPS). |
| `HandTrackingModule.py` | Runs MediaPipe on each frame, draws the hand outline, and returns the pixel coordinates of the joints for the selected hand. |

## Run it on Windows

You need a webcam and 64-bit Python 3.11 or 3.12. Place both Python files in the same folder and open that folder in the VS Code desktop app. Use a **local PowerShell terminal** so the script can access your laptop’s webcam.

If you already have the virtual environment from the hand gesture volume project, you can reuse it. Otherwise, run these setup commands once. Use `py -3.12` in the first command if you have Python 3.12.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "mediapipe==0.10.21" "opencv-contrib-python==4.11.0.86" "numpy<2"
```

Start the counter:

```powershell
.\.venv\Scripts\python.exe finger_counter.py
```

Hold your hand where the webcam can see your wrist and fingertips. The count appears in the webcam window and prints in the terminal. A visible closed fist should show **0**. When both hands leave the frame, the green box disappears. To stop the program, focus the terminal and press **Ctrl+C**.

## What makes this finger counter different?

My first version counted a finger as open when its tip was higher on the screen than its middle joint. When I turned my open hand sideways, the counter showed **3** even though all **5** fingers were extended. I changed the counting rules to handle poses like that.

- **Sideways fingers.** The code measures from the base knuckle to the fingertip twice: once in a straight line and once along the joints. An extended finger has a straight-line distance greater than 85% of the path along its joints. Turning the hand sideways in the image preserves those distances.
- **Folded fingers.** A curled finger can look straight when its joints overlap on camera. The tip must also reach past the first finger segment and farther from the wrist than its base knuckle. I added the wrist check after two thumbs down showed **4**.
- **Left and right thumbs.** The code uses the line from the pinky base to the index base to find the thumb side of each palm. A thumb counts when its tip extends past the index side by more than 30% of the palm width. The same rule works when the hands face opposite directions in the image.
- **Two-hand totals.** The counter checks every detected hand in the frame and adds the results. Three raised fingers on one hand and two on the other give a total of **5**.

These rules use the joints visible in a flat webcam image. Fingers that overlap or point directly toward the camera can still be difficult to count.

## Poses to try

| Pose | Expected total |
| --- | --- |
| One closed fist | 0 |
| One sideways open hand | 5 |
| Both thumbs down, other fingers folded | 2 |
| Both hands open | 10 |

by **Aditya Mishra**.
