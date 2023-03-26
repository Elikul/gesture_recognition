import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def get_coord_landmarks(landmsrks):
    lm = []

    for hand_landmarks in landmsrks:
        for i in range(21):
            lm.append([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y,
                       hand_landmarks.landmark[i].z])

    return lm


def find_delta_xy(landmarks):
    max_x, max_y = 0, 0
    min_x, min_y = 100, 100

    for coord in landmarks:
        if coord[0] > max_x:
            max_x = coord[0]
        if coord[0] < min_x:
            min_x = coord[0]
        if coord[1] > max_y:
            max_y = coord[1]
        if coord[1] < min_y:
            min_y = coord[1]

    return max_x, min_x, max_y, min_y


def get_position_wrists(hands):
    hand_0 = hands.multi_hand_world_landmarks[0]
    hand_1 = hands.multi_hand_world_landmarks[1]

    wrist_0_norm = hand_0.landmark[mp_hands.HandLandmark.WRIST]
    wrist_1_norm = hand_1.landmark[mp_hands.HandLandmark.WRIST]

    lm = get_coord_landmarks(hands.multi_hand_world_landmarks)

    max_x, min_x, max_y, min_y = find_delta_xy(lm)

    dx, dy = max_x - min_x, max_y - min_y

    dist_x = (wrist_1_norm.x - wrist_0_norm.x) / dx
    dist_y = (wrist_1_norm.y - wrist_0_norm.y) / dy
    dist_z = wrist_1_norm.z - wrist_0_norm.z

    return [dist_x, dist_y, dist_z]
