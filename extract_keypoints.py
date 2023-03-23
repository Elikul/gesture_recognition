import json

import cv2
import numpy as np
import mediapipe as mp
import os

DATA_PATH = os.path.join('Hands_Data')
CUT_VIDEOS_PATH = os.path.join('cut')
PHOTO_PATH = os.path.join('photos')
HANDS_PATH = os.path.join('mediapipe_photo')
LANDMARK_THICKNESS = 2

numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
videos = {'ALY': 2, 'ELINA': 4, 'ILYA': 4, 'NASTYA': 2, 'NIKITA': 2, 'OLYA': 2, 'POLINA': 2, 'SACHA': 1, 'SACHA_GOL': 2,
          'SERGEY': 2, 'SLAVA': 3, 'TATYANA': 1, 'TIMOFEY': 2, 'VASYA': 2, 'WOMAN': 2}

interaction_type = {'-1': 'none', '1': 'close', '0': 'far'}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # пометить образ как недоступный для записи, чтобы повысить производительность
    image.flags.writeable = False

    # обработать изображения и определить ориентиры
    results = model.process(image)

    # пометить образ как доступный для записи, чтобы нарисовать аннотации ориентиров на изображении
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results, frame_width, frame_height):
    output = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            output.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * frame_width,
                           hand_landmarks.landmark[
                               mp_hands.HandLandmark.RING_FINGER_TIP].y * frame_height] if hand_landmarks else np.zeros(
                21 * 3))
    return np.array(output)


def extact_keypoints_from_photo():
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    if not os.path.exists(HANDS_PATH):
        os.mkdir(HANDS_PATH)

    for number in numbers:
        path1 = os.path.join(DATA_PATH, number)
        path2 = os.path.join(HANDS_PATH, number)
        if not os.path.exists(path1):
            os.mkdir(path1)
        if not os.path.exists(path2):
            os.mkdir(path2)

    for number in numbers:
        folder_path = os.path.join(PHOTO_PATH, number)
        photos_name = [f for f in os.listdir(folder_path)]

        # установить модель mediapipe
        with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6) as hands:
            for file_name in photos_name:
                file = os.path.join(PHOTO_PATH, number, file_name)

                image = cv2.flip(cv2.imread(file), 1)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results is None or results.multi_hand_landmarks is None:
                    print('NO DATA for ', file_name)
                else:
                    image_height, image_width, _ = image.shape
                    annotated_image = image.copy()

                    lm = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(21):
                            lm.append([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y,
                                       hand_landmarks.landmark[i].z])

                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        cv2.imwrite(
                            os.path.join(HANDS_PATH, number, file_name), cv2.flip(annotated_image, 1))

                        path_coord = os.path.join(DATA_PATH, number, file_name.replace('.png', '') + ".json")
                        if len(results.multi_handedness) == 1 or (len(results.multi_handedness) == 2 and len(lm) == 42):
                            with open(path_coord, 'w') as jf:
                                intr_type = interaction_type['-1']
                                if len(results.multi_handedness) == 2 and len(lm) == 42:
                                    max_x, max_y = 0, 0
                                    min_x, min_y = 100, 100
                                    print('Two hands')
                                    for coord in lm:
                                        if coord[0] > max_x:
                                            max_x = coord[0]
                                        if coord[0] < min_x:
                                            min_x = coord[0]
                                        if coord[1] > max_y:
                                            max_y = coord[1]
                                        if coord[1] < min_y:
                                            min_y = coord[1]

                                    dx, dy = max_x - min_x, max_y - min_y

                                    for i in range(len(lm)):
                                        lm[i][0] = (lm[i][0] - min_x) / dx
                                        lm[i][1] = (lm[i][1] - min_y) / dy

                                    hand1 = np.array(
                                        [[marks.x, marks.y] for marks in results.multi_hand_landmarks[0].landmark])
                                    hand2 = np.array(
                                        [[marks.x, marks.y] for marks in results.multi_hand_landmarks[1].landmark])

                                    center1 = np.mean(hand1, axis=0)
                                    center2 = np.mean(hand2, axis=0)

                                    distance = np.linalg.norm(center1 - center2)
                                    print(number, distance)
                                    if 0 < distance <= 0.3:
                                        intr_type = interaction_type['1']
                                    else:
                                        intr_type = interaction_type['0']

                                #чтоб везде было 42
                                if len(results.multi_handedness) == 1:
                                    lm = lm + [[0.0] * 3] * 21

                                json.dump(
                                    {'num_hands': len(results.multi_handedness), 'landmarks': lm,
                                     'interaction': intr_type},
                                    jf)
                                jf.close()


def extract_keypoints_from_video():
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    for number in numbers:
        path = os.path.join(DATA_PATH, number)
        if not os.path.exists(path):
            os.mkdir(path)
        for video_name in videos:
            path = os.path.join(DATA_PATH, number, video_name)
            if not os.path.exists(path):
                os.mkdir(path)
            for video_idx in range(videos[video_name]):
                path = os.path.join(DATA_PATH, number, video_name, str(video_idx))
                if not os.path.exists(path):
                    os.mkdir(path)

    for number in numbers:

        folder_path = os.path.join(CUT_VIDEOS_PATH, number)

        for video_name in videos:
            for video_idx in range(videos[video_name]):

                video_path = os.path.join(folder_path,
                                          video_name + '_' + number + '_' + str(video_idx) + '.mp4')

                cap = cv2.VideoCapture(video_path)

                # установить модель mediapipe
                with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.8) as hands:

                    # перебрать каждый кадр
                    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):

                        ret, frame = cap.read()

                        if frame is None or not ret:
                            break

                        # определить ориентиры
                        image, results = mediapipe_detection(frame, hands)
                        print(results)

                        # нарисовать аннотации
                        draw_styled_landmarks(image, results)

                        frame_height, frame_width, z = image.shape

                        # экспортировать ключевых точек
                        keypoints = extract_keypoints(results, frame_width, frame_height)
                        npy_path = os.path.join(DATA_PATH, number, video_name, str(video_idx), str(frame_idx))
                        np.save(npy_path, keypoints)

                        cv2.imshow('MediaPipe Hands', image)

                        if cv2.waitKey(5) & 0xFF == 27:  # Esc
                            break

                # Close down everything
                cap.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    extact_keypoints_from_photo()
