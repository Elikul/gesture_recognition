import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np

from public.utils import position, find_delta_xy, get_coord_landmarks

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

AMBER_COLOR = (0, 191, 255)
BLUE_COLOR = (255, 191, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)

NUM_IMG_POSITION = (10, 30)
TEXT_TOP_POSITION = (50, 30)
FONT_SCALE = 0.5
TEXT_THICKNESS = 1

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 720
THICKNESS_CONTOUR = 1
CONTOUR_IDX = -1
RECTANGLE_THICKNESS = 3

label_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10'}


def load_model():
    model = tf.keras.models.load_model('model/my_model')
    return model


def extract_keypoints(frame, model):
    # установить модель mediapipe
    with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6) as hands:

        image = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results is None or results.multi_hand_landmarks is None:
            print('NO DATA')
        else:
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            lm = get_coord_landmarks(results.multi_hand_landmarks)

            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if len(results.multi_hand_landmarks) == 1 or (
                        len(results.multi_hand_landmarks) == 2 and len(lm) == 42):

                    if len(results.multi_hand_landmarks) == 2 and len(lm) == 42:
                        max_x, min_x, max_y, min_y = find_delta_xy(lm)

                        dx, dy = max_x - min_x, max_y - min_y

                        for i in range(len(lm)):
                            lm[i][0] = (lm[i][0] - min_x) / dx
                            lm[i][1] = (lm[i][1] - min_y) / dy

                        lm = lm + [position(results)]

                    # чтоб везде было 42
                    if len(results.multi_hand_landmarks) == 1:
                        lm = lm + [[0.0] * 3] * 22

                    prediction = model.predict([lm])
                    print(label_map[np.argmax(prediction)])
                return label_map[np.argmax(prediction)], cv2.flip(annotated_image, 1)
        return -1, cv2.flip(image, 1)


def video_capture(model):
    num_frames = 0

    camera = cv2.VideoCapture(0)
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    if camera is None or not camera.isOpened():
        print("Предупреждение: невозможно открыть источник видео")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Не удается получить кадр. Выход ...")
            break
        # переворачивание кадра для предотвращения перевернутого изображения захваченного кадра
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        frame_copy = cv2.resize(frame_copy, (WINDOW_WIDTH, WINDOW_HEIGHT))

        cv2.putText(frame_copy, str(num_frames), NUM_IMG_POSITION,
                    cv2.FONT_HERSHEY_COMPLEX, FONT_SCALE, GREEN_COLOR, TEXT_THICKNESS)

        dig, frame_copy = extract_keypoints(frame_copy, model)

        cv2.putText(frame_copy, str(dig), TEXT_TOP_POSITION,
                    cv2.FONT_HERSHEY_COMPLEX, FONT_SCALE, RED_COLOR, TEXT_THICKNESS)

        # отобразить кадр с сегментированной рукой
        cv2.imshow("Video", frame_copy)

        # увеличить количество кадров для отслеживания
        num_frames += 1
        # закрытие окон с помощью клавиши Esc (можно использовать и любую другую клавишу с ord)
        if cv2.waitKey(1) == ord('q'):
            break
    # освобождение камеры и разрушение всех окон
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_model()
    video_capture(model)
