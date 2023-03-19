import cv2
import numpy as np
import mediapipe as mp
import os

DATA_PATH = os.path.join('Holistic_Data')
CUT_VIDEOS_PATH = os.path.join('cut')
PHOTO_PATH = os.path.join('photos')
LANDMARK_THICKNESS = 2

numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
videos = {'ALY': 2, 'ELINA': 4, 'ILYA': 4, 'NASTYA': 2, 'NIKITA': 2, 'OLYA': 2, 'POLINA': 2, 'SACHA': 1, 'SACHA_GOL': 2,
          'SERGEY': 2, 'SLAVA': 3, 'TATYANA': 1, 'TIMOFEY': 2, 'VASYA': 2, 'VITAL': 2, 'WOMAN': 2}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


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
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=LANDMARK_THICKNESS, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=LANDMARK_THICKNESS, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=LANDMARK_THICKNESS,
                                                     circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=LANDMARK_THICKNESS,
                                                     circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=LANDMARK_THICKNESS,
                                                     circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=LANDMARK_THICKNESS,
                                                     circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    left_hand = np.array([[res.x, res.y, res.z] for res in
                          results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    return np.concatenate([pose, left_hand, right_hand])


def extact_keypoints_from_photo():
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    for number in numbers:
        path = os.path.join(DATA_PATH, number)
        if not os.path.exists(path):
            os.mkdir(path)

    for number in numbers:
        folder_path = os.path.join(PHOTO_PATH, number)
        photos_name = [f for f in os.listdir(folder_path)]

        for file_name in photos_name:

            # установить модель mediapipe
            with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.6) as holistic:
                file = os.path.join(PHOTO_PATH, number, file_name)

                original_image = cv2.imread(file)

                image, results = mediapipe_detection(original_image, holistic)
                print(results)

                # нарисовать аннотации
                draw_styled_landmarks(image, results)

                # экспортировать ключевых точек
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, number, file_name.replace('.png', ''))
                np.save(npy_path, keypoints)


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
                with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.8) as holistic:

                    # перебрать каждый кадр
                    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):

                        ret, frame = cap.read()

                        if frame is None or not ret:
                            break

                        # определить ориентиры
                        image, results = mediapipe_detection(frame, holistic)
                        print(results)

                        # нарисовать аннотации
                        draw_styled_landmarks(image, results)

                        # экспортировать ключевых точек
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, number, video_name, str(video_idx), str(frame_idx))
                        np.save(npy_path, keypoints)

                        cv2.imshow('MediaPipe Holistic', image)

                        if cv2.waitKey(5) & 0xFF == 27:  # Esc
                            break

                # Close down everything
                cap.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    extact_keypoints_from_photo()
