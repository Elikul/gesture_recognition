import json

import cv2
import mediapipe as mp
import os

from public.coordinates_utils import get_coord_landmarks, find_delta_xy, get_position_wrists

DATA_PATH = os.path.join('../Hands_Data')
PHOTO_PATH = os.path.join('../photos')
HANDS_PATH = os.path.join('../mediapipe_photo')

ASL_PATH = os.path.join('../ASL_Hands_Data')
ASL_PHOTO_PATH = os.path.join('../ASL_digits')
ASL_HANDS_PATH = os.path.join('../ASL_mediapipe_photo')

LANDMARK_THICKNESS = 2

numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
asl_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def make_dirs():
    if not os.path.exists(ASL_PATH):
        os.mkdir(ASL_PATH)

    if not os.path.exists(ASL_HANDS_PATH):
        os.mkdir(ASL_HANDS_PATH)

    for number in asl_numbers:
        path1 = os.path.join(ASL_PATH, number)
        path2 = os.path.join(ASL_HANDS_PATH, number)
        if not os.path.exists(path1):
            os.mkdir(path1)
        if not os.path.exists(path2):
            os.mkdir(path2)


def draw_landmarks(annotated_image, hand_landmarks, number, file_name):
    mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        os.path.join(ASL_HANDS_PATH, number, file_name), cv2.flip(annotated_image, 1))


def extract_keypoints_from_photo():
    for number in asl_numbers:
        folder_path = os.path.join(ASL_PHOTO_PATH, number)
        photos_name = [f for f in os.listdir(folder_path)]

        # установить модель mediapipe
        with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6) as hands:
            for file_name in photos_name:
                file = os.path.join(ASL_PHOTO_PATH, number, file_name)

                image = cv2.flip(cv2.imread(file), 1)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results is None or results.multi_hand_landmarks is None:
                    print('NO DATA for ', file_name)
                else:
                    image_height, image_width, _ = image.shape
                    annotated_image = image.copy()

                    lm = get_coord_landmarks(results.multi_hand_landmarks)

                    for hand_landmarks in results.multi_hand_landmarks:

                        draw_landmarks(annotated_image, hand_landmarks, number, file_name)

                        path_coord = os.path.join(ASL_PATH, number, file_name.replace('.jpeg', '') + ".json")

                        if len(results.multi_hand_landmarks) == 1 or (
                                len(results.multi_hand_landmarks) == 2 and len(lm) == 42):

                            with open(path_coord, 'w') as jf:
                                # if len(results.multi_hand_landmarks) == 2 and len(lm) == 42:
                                #     max_x, min_x, max_y, min_y = find_delta_xy(lm)
                                #
                                #     dx, dy = max_x - min_x, max_y - min_y
                                #
                                #     for i in range(len(lm)):
                                #         lm[i][0] = (lm[i][0] - min_x) / dx
                                #         lm[i][1] = (lm[i][1] - min_y) / dy
                                #
                                #     lm = lm + [get_position_wrists(results)]
                                #
                                # # чтоб везде было 42
                                # if len(results.multi_hand_landmarks) == 1:
                                #     lm = lm + [[0.0] * 3] * 22

                                json.dump({'num_hands': len(results.multi_hand_landmarks), 'landmarks': lm}, jf)
                                jf.close()


if __name__ == "__main__":
    make_dirs()
    extract_keypoints_from_photo()
