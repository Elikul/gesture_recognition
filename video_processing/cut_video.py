from moviepy.editor import *
import os

VIDEO_NAME = ""
SRC_VIDEO_PATH = os.path.join('../videos', VIDEO_NAME + '.mp4')
CUT_VIDEOS_PATH = os.path.join('../cut')

numbers = {'1': (0, 0.8), '2': (1, 1.6), '3': (2, 2.5), '4': (2.8, 3.2),
           '5': (3.6, 4), '6': (4.5, 5.2), '7': (5.4, 6.2),
           '8': (6.4, 7), '9': (7.3, 8.2), '10': (8.5, 10)}


def cut_video_for_dataset():
    if not os.path.exists(CUT_VIDEOS_PATH):
        os.mkdir(CUT_VIDEOS_PATH)

    for number in numbers:
        path = os.path.join(CUT_VIDEOS_PATH, number)
        if not os.path.exists(path):
            os.mkdir(path)

    src_video = VideoFileClip(SRC_VIDEO_PATH)

    for number in numbers:
        write_path = os.path.join(CUT_VIDEOS_PATH, number)

        start_point = numbers[number][0]
        end_point = numbers[number][1]

        cutted_video = src_video.subclip(start_point, end_point)
        cutted_video.write_videofile(os.path.join(write_path, VIDEO_NAME + '_' + str(3) + '.mp4'))


if __name__ == "__main__":
    cut_video_for_dataset()
