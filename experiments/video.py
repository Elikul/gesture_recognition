import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

AMBER_COLOR = (0, 191, 255)
BLUE_COLOR = (255, 191, 0)
GREEN_COLOR = (0, 255, 0)

NUM_IMG_POSITION = (10, 30)
TEXT_TOP_POSITION = (100, 50)
MIN_FONT_SCALE = 0.5
BIG_FONT_SCALE = 0.8
TEXT_THICKNESS = 1

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 720
THICKNESS_CONTOUR = 1

label_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10'}


def load_model():
    model = tf.keras.models.load_model('model/CNN')
    return model


def draw_landmarks(hands, frame):
    if hands.multi_hand_landmarks:
        for hand_landmarks in hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


def video_capture(model):
    num_frames = 0

    cap = cv2.VideoCapture(0)
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)

    with mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.6) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # переворачивание кадра для предотвращения перевернутого изображения захваченного кадра
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()

            frame_copy = cv2.resize(frame_copy, size)

            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_copy)

            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)

            draw_landmarks(results, frame_copy)

            cv2.putText(frame_copy, str(num_frames), NUM_IMG_POSITION,
                        cv2.FONT_HERSHEY_COMPLEX, MIN_FONT_SCALE, GREEN_COLOR, TEXT_THICKNESS)

            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) == 1:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                        x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                        y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                        y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                        if x_max > 1:
                            x_max = 1
                        if y_max > 1:
                            y_max = 1
                        if x_min < 0:
                            x_min = 0
                        if y_min < 0:
                            y_min = 0
                        x_min, y_min = mp_drawing._normalized_to_pixel_coordinates(x_min, y_min, frame_copy.shape[1],
                                                                                   frame_copy.shape[0])
                        x_max, y_max = mp_drawing._normalized_to_pixel_coordinates(x_max, y_max, frame_copy.shape[1],
                                                                                   frame_copy.shape[0])
                        cropped_img = frame_copy[y_min:y_max, x_min:x_max]
                        cropped_img = cv2.resize(cropped_img, (64, 64), interpolation=cv2.INTER_AREA)
                        prediction = model.predict(cropped_img)
                        digit = label_map[np.argmax(prediction)]

                        cv2.putText(frame_copy, digit, TEXT_TOP_POSITION,
                                    cv2.FONT_HERSHEY_COMPLEX, BIG_FONT_SCALE, AMBER_COLOR, TEXT_THICKNESS)

                if len(results.multi_hand_landmarks) == 2:
                    lm = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                        x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                        y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                        y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                        lm.append([x_min, x_max, y_min, y_max])
                    x_min, y_min = min(lm[0][0], lm[1][0]), min(lm[0][2], lm[1][2])
                    x_max, y_max = max(lm[0][1], lm[1][1]), max(lm[0][3], lm[1][3])

                    if x_max > 1:
                        x_max = 1
                    if y_max > 1:
                        y_max = 1
                    if x_min < 0:
                        x_min = 0
                    if y_min < 0:
                        y_min = 0
                    x_min, y_min = mp_drawing._normalized_to_pixel_coordinates(x_min, y_min, frame_copy.shape[1],
                                                                               frame_copy.shape[0])
                    x_max, y_max = mp_drawing._normalized_to_pixel_coordinates(x_max, y_max, frame_copy.shape[1],
                                                                               frame_copy.shape[0])
                    cropped_img = frame_copy[y_min:y_max, x_min:x_max]
                    cropped_img = cv2.resize(cropped_img, (64, 64), interpolation=cv2.INTER_AREA)
                    prediction = model.predict([cropped_img])
                    digit = label_map[np.argmax(prediction)]

                    cv2.putText(frame_copy, digit, TEXT_TOP_POSITION,
                                cv2.FONT_HERSHEY_COMPLEX, BIG_FONT_SCALE, AMBER_COLOR, TEXT_THICKNESS)

            # отобразить кадр с сегментированной рукой
            cv2.imshow("Video", frame_copy)

            # увеличить количество кадров для отслеживания
            num_frames += 1
            # закрытие окон с помощью клавиши Esc (можно использовать и любую другую клавишу с ord)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    # освобождение камеры и разрушение всех окон
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_model()
    video_capture(model)
