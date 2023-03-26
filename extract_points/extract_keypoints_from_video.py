import cv2
import numpy as np
import mediapipe as mp
import os

DATA_PATH = os.path.join('Hands_Data_Video')
CUT_VIDEOS_PATH = os.path.join('../cut')
LANDMARK_THICKNESS = 2

numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
videos = {'ALY': 2, 'ELINA': 4, 'ILYA': 4, 'NASTYA': 2, 'NIKITA': 2, 'OLYA': 2, 'POLINA': 2, 'SACHA': 1, 'SACHA_GOL': 2,
          'SERGEY': 2, 'SLAVA': 3, 'TATYANA': 1, 'TIMOFEY': 2, 'VASYA': 2, 'WOMAN': 2}

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


def make_dies():
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


def extract_keypoints_from_video():
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
                        # npy_path = os.path.join(DATA_PATH, number, video_name, str(video_idx), str(frame_idx))
                        # np.save(npy_path, keypoints)

                        cv2.imshow('MediaPipe Hands', image)

                        if cv2.waitKey(5) & 0xFF == 27:  # Esc
                            break

                # Close down everything
                cap.release()
                cv2.destroyAllWindows()
