import cv2

background = None

# размеры для ROI
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


def calc_accumulated_avg(frame, accumulated_weight=0.5):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    # Grab the external contours for the image
    image, contours, hierarchy = cv2.findContours(thresholded.copy(),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:

        hand_segment_max_cont = max(contours, key=cv2.contourArea)

        return thresholded, hand_segment_max_cont
