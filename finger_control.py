import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

class Finger(Enum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4

class HandSide(Enum):
    LEFT = 0
    RIGHT = 1

# Config
MAX_VALUE = 100
MIN_VALUE = -100
CHANGE_RATE = 1
DELAY_SECONDS = 1

selected_finger = None
finger_values = {finger: 0 for finger in Finger}
last_update_time = 0
current_direction = None

def get_hand_side(hand_landmarks, handedness):
    return HandSide.LEFT if handedness.classification[0].label == 'Left' else HandSide.RIGHT


def is_finger_extended(hand_landmarks, finger):
    landmarks = hand_landmarks.landmark
    tips = {
        Finger.THUMB: 4,
        Finger.INDEX: 8,
        Finger.MIDDLE: 12,
        Finger.RING: 16,
        Finger.PINKY: 20
    }
    pips = {
        Finger.THUMB: 2,
        Finger.INDEX: 6,
        Finger.MIDDLE: 10,
        Finger.RING: 14,
        Finger.PINKY: 18
    }

    tip = landmarks[tips[finger]]
    pip = landmarks[pips[finger]]

    if finger == Finger.THUMB:
        return tip.x > pip.x
    return tip.y < pip.y


def is_fist(hand_landmarks):
    for finger in Finger:
        if is_finger_extended(hand_landmarks, finger):
            return False
    return True


def check_single_finger_up(hand_landmarks):
    extended = [finger for finger in Finger if is_finger_extended(hand_landmarks, finger)]
    return extended[0] if len(extended) == 1 else None


def update_finger_value():
    global finger_values, last_update_time, current_direction

    current_time = time.time()
    if current_time - last_update_time >= DELAY_SECONDS:
        if current_direction == "up":
            finger_values[selected_finger] = min(finger_values[selected_finger] + CHANGE_RATE, MAX_VALUE)
        elif current_direction == "down":
            finger_values[selected_finger] = max(finger_values[selected_finger] - CHANGE_RATE, MIN_VALUE)
        last_update_time = current_time


def process_frame(frame):
    global selected_finger, current_direction, last_update_time

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    left_hand = right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            side = get_hand_side(hand_landmarks, results.multi_handedness[i])

            if side == HandSide.LEFT:
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if left_hand:
        new_selection = check_single_finger_up(left_hand)
        if new_selection and new_selection != selected_finger:
            selected_finger = new_selection
            current_direction = None
            last_update_time = 0

    if right_hand and selected_finger:
        if is_fist(right_hand):
            current_direction = None
        else:
            is_index_up = is_finger_extended(right_hand, Finger.INDEX)
            current_direction = "up" if is_index_up else "down"

            update_finger_value()

    cv2.putText(frame, "Precision Finger Control", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    if selected_finger and current_direction:
        time_left = max(0, DELAY_SECONDS - (time.time() - last_update_time))
        cv2.putText(frame, f"Next update in: {time_left:.1f}s", (frame.shape[1] - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    if selected_finger:
        cv2.putText(frame, f"Selected: {selected_finger.name}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Raise ONE left finger to select", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for i, finger in enumerate(Finger):
        value = finger_values[finger]
        color = (255, 255, 255)
        symbol = " "

        if value > 0:
            color = (0, 255, 0)
            symbol = "+"
        elif value < 0:
            color = (0, 0, 255)
            symbol = "-"

        if finger == selected_finger:
            color = (0, 255, 255)
            symbol = "▶" if current_direction == "up" else "◀" if current_direction == "down" else "="

        cv2.putText(frame, f"{finger.name}: {symbol}{abs(value):03d}", (10, 110 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Finger Controller", processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()