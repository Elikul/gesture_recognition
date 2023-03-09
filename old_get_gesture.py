import cv2
import imutils

background = None

# размеры для ROI
ROI_TOP = 100
ROI_BOTTOM = 400
ROI_RIGHT = 550
ROI_LEFT = 850

AMBER_COLOR = (0, 191, 255)
BLUE_COLOR = (255, 191, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)

NUM_IMG_POSITION = (10, 30)
TEXT_TOP_POSITION = (50, 30)
FONT_SCALE = 0.5
TEXT_THICKNESS = 1

PATH_TO_SAVE = "resources/test/"
WINDOW_WIDTH = 1020
THICKNESS_CONTOUR = 1
CONTOUR_IDX = -1
RECTANGLE_THICKNESS = 3

elements = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


# путём вычисления накопленного веса для некоторых кадров (для 60 кадров), вычисляем накопленное среднее для фона
def calc_accumulated_avg(frame, accumulated_weight=0.5):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)


# если есть рука в ROI
def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    # захватить внешние контуры для изображения
    contours, hierarchy = cv2.findContours(thresholded.copy(),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment_max_cont


# воспроизведение видео для сегментации руки
def video_capture():
    num_frames = 0
    num_imgs_taken = 0

    camera = cv2.VideoCapture(0)
    if camera is None or not camera.isOpened():
        print("Предупреждение: невозможно открыть источник видео")
        return

    while True:
        ret, frame = camera.read()
        frame = imutils.resize(frame, width=WINDOW_WIDTH)
        # переворачивание кадра для предотвращения перевернутого изображения захваченного кадра
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_RIGHT:ROI_LEFT]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 60:
            calc_accumulated_avg(gray_frame)
            cv2.putText(frame_copy, "Определение фона... пожалуйста, подождите",
                        TEXT_TOP_POSITION, cv2.FONT_HERSHEY_COMPLEX, FONT_SCALE, AMBER_COLOR, TEXT_THICKNESS)
        elif num_frames >= 100:
            element = elements[4]
            cv2.putText(frame_copy, "Жест для " +
                        str(element), TEXT_TOP_POSITION, cv2.FONT_HERSHEY_COMPLEX, FONT_SCALE,
                        AMBER_COLOR, TEXT_THICKNESS)

            # сегментирование области руки
            hand = segment_hand(gray_frame)

            # проверяем, удалось ли нам обнаружить руку
            if hand is not None:
                # распаковать пороговое изображение и контур max_contour
                thresholded, hand_segment = hand
                # нарисовать контуры вокруг сегмента руки
                cv2.drawContours(frame_copy, [hand_segment + (ROI_RIGHT,
                                                              ROI_TOP)], CONTOUR_IDX, RED_COLOR, THICKNESS_CONTOUR)
                # отображение порогового изображения
                cv2.imshow("Thresholded Hand Image", thresholded)

                # рука присутствует в зоне охвата, можно сохранять изображение зоны охвата в обучающий и тестовый набор
                cv2.imwrite(PATH_TO_SAVE + element + '/' + str(num_imgs_taken) + '.jpg', thresholded)
                num_imgs_taken += 1
            else:
                cv2.putText(frame_copy, 'Рука не обнаружена', TEXT_TOP_POSITION,
                            cv2.FONT_HERSHEY_COMPLEX, FONT_SCALE, RED_COLOR, TEXT_THICKNESS)

        cv2.putText(frame_copy, str(num_frames), NUM_IMG_POSITION,
                    cv2.FONT_HERSHEY_COMPLEX, FONT_SCALE, GREEN_COLOR, TEXT_THICKNESS)

        # Нарисовать ROI на копии кадра
        cv2.rectangle(frame_copy, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), BLUE_COLOR, RECTANGLE_THICKNESS)

        # увеличить количество кадров для отслеживания
        num_frames += 1
        # отобразить кадр с сегментированной рукой
        cv2.imshow("Video", frame_copy)
        # закрытие окон с помощью клавиши Esc (можно использовать и любую другую клавишу с ord)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # освобождение камеры и разрушение всех окон
    cv2.destroyAllWindows()
    camera.release()
