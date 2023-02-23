import cv2

background = None

# размеры для ROI
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


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
    image, contours, hierarchy = cv2.findContours(thresholded.copy(),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment_max_cont


# рука присутствует в зоне охвата, можно сохранять изображение зоны охвата в обучающий и тестовый набор
def video_capture():
    cam = cv2.VideoCapture(0)
    num_frames = 0
    element = 10
    num_imgs_taken = 0
    savedPath = "..\\resourses\\train\\"

    while True:
        ret, frame = cam.read()
        # переворачивание кадра для предотвращения перевернутого изображения захваченного кадра
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        if num_frames < 60:
            calc_accumulated_avg(gray_frame)
            if num_frames <= 59:
                cv2.putText(frame_copy, "Определение фона... пожалуйста, подождите",
                            (80, 400), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)

        # настроить определение руки конкретно в ROI
        elif num_frames <= 300:
            hand = segment_hand(gray_frame)

            cv2.putText(frame_copy, "Жест для " +
                        str(element), (200, 400), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 2)

            # проверка факта обнаружения руки путем подсчета количества обнаруженных контуров
            if hand is not None:
                thresholded, hand_segment = hand
                # нарисовать контуры вокруг сегмента руки
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
                                                              ROI_top)], -1, (255, 0, 0), 1)

                cv2.putText(frame_copy, str(num_frames) + "для" + str(element),
                            (70, 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                # отобразим пороговое изображение
                cv2.imshow("Пороговое изображение руки", thresholded)

        else:
            # сегментирование области руки
            hand = segment_hand(gray_frame)

            # проверяем, удалось ли нам обнаружить руку
            if hand is not None:
                # распаковать пороговое изображение и контур max_contour
                thresholded, hand_segment = hand
                # нарисовать контуры вокруг сегмента руки
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
                                                              ROI_top)], -1, (255, 0, 0), 1)

                cv2.putText(frame_copy, str(num_frames), (70, 45),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                cv2.putText(frame_copy, str(num_imgs_taken) + 'images' + "для"
                            + str(element), (200, 400), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)
                # отображение порогового изображения
                cv2.imshow("Пороговое изображение руки", thresholded)
                if num_imgs_taken <= 300:
                    cv2.imwrite(savedPath + str(element) + "\\" +
                                str(num_imgs_taken + 300) + '.jpg', thresholded)
                else:
                    break
                num_imgs_taken += 1
            else:
                cv2.putText(frame_copy, 'Рука не обнаружена', (200, 400),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # Нарисовать ROI на копии кадра
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)

        cv2.putText(frame_copy, "Распознавание жеста ", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (51, 255, 51), 1)

        # увеличить количество кадров для отслеживания
        num_frames += 1
        # отобразить кадр с сегментированной рукой
        cv2.imshow("Обнаружение жеста", frame_copy)
        # закрытие окон с помощью клавиши Esc (можно использовать и любую другую клавишу с ord)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # освобождение камеры и разрушение всех окон
    cv2.destroyAllWindows()
    cam.release()
