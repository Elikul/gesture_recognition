import cv2
import imutils

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

PATH_TO_SAVE = "resources/dataset/"
person = "SLAVA"
video = "videos/"
photo = "photo/"


def video_capture():
    num_frames = 0
    num_imgs_taken = 0

    camera = cv2.VideoCapture(0)
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(PATH_TO_SAVE + video + person + '.avi', fourcc, 20, size)
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

        out.write(frame_copy)
        cv2.imwrite(PATH_TO_SAVE + photo + person + '/' + str(num_imgs_taken) + '.png', frame_copy)
        num_imgs_taken += 1

        # отобразить кадр с сегментированной рукой
        cv2.imshow("Video", frame_copy)

        # увеличить количество кадров для отслеживания
        num_frames += 1
        # закрытие окон с помощью клавиши Esc (можно использовать и любую другую клавишу с ord)
        if cv2.waitKey(1) == ord('q'):
            break
    # освобождение камеры и разрушение всех окон
    camera.release()
    out.release()
    cv2.destroyAllWindows()
