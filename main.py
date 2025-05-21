import cv2
import mediapipe as mp
import numpy as np
from enum import Enum

# Initialize MediaPipe Hands
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


# Configuration
MOVEMENT_THRESHOLD = 0.02  # Sensitivity for value changes
MAX_VALUE = 100
MIN_VALUE = -100
CHANGE_RATE = 2  # Speed of value change

# System state
selected_finger = None
finger_values = {finger: 0 for finger in Finger}
last_right_pos = None
accumulated_movement = 0


def get_hand_side(hand_landmarks, img_width):
    """Improved hand side detection using handedness"""
    # Use x-coordinate of wrist and pinky MCP for reliable detection
    wrist_x = hand_landmarks.landmark[0].x * img_width
    pinky_mcp_x = hand_landmarks.landmark[17].x * img_width
    return HandSide.LEFT if wrist_x < pinky_mcp_x else HandSide.RIGHT


def is_finger_extended(hand_landmarks, finger):
    """Precise finger extension detection with joint angle checking"""
    landmarks = hand_landmarks.landmark
    finger_joints = {
        Finger.THUMB: [1, 2, 3, 4],  # mcp, pip, dip, tip
        Finger.INDEX: [5, 6, 7, 8],
        Finger.MIDDLE: [9, 10, 11, 12],
        Finger.RING: [13, 14, 15, 16],
        Finger.PINKY: [17, 18, 19, 20]
    }

    joints = finger_joints[finger]
    vectors = []

    # Create vectors between joints
    for i in range(len(joints) - 1):
        x1, y1 = landmarks[joints[i]].x, landmarks[joints[i]].y
        x2, y2 = landmarks[joints[i + 1]].x, landmarks[joints[i + 1]].y
        vectors.append((x2 - x1, y2 - y1))

    # Calculate angles between vectors
    extended = True
    for i in range(len(vectors) - 1):
        v1 = vectors[i]
        v2 = vectors[i + 1]
        angle = np.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))
        if abs(angle) > 30:  # If angle between segments is too large, finger is bent
            extended = False
            break

    return extended


def check_single_finger_up(hand_landmarks):
    """Strict check for exactly one finger up (others must be clearly down)"""
    extended_fingers = []

    for finger in Finger:
        if is_finger_extended(hand_landmarks, finger):
            extended_fingers.append(finger)
            if len(extended_fingers) > 1:
                return None  # More than one finger up

    return extended_fingers[0] if extended_fingers else None


def update_value(current_pos, last_pos):
    """Calculate value change based on precise vertical movement"""
    global accumulated_movement

    if last_pos is None:
        return 0

    # Calculate vertical movement (normalized to screen height)
    dy = (current_pos[1] - last_pos[1])

    # Accumulate small movements until threshold is reached
    accumulated_movement += dy

    value_change = 0
    if accumulated_movement > MOVEMENT_THRESHOLD:
        value_change = -CHANGE_RATE  # Up movement (decrease negative or increase positive)
        accumulated_movement = 0
    elif accumulated_movement < -MOVEMENT_THRESHOLD:
        value_change = CHANGE_RATE  # Down movement
        accumulated_movement = 0

    return value_change


def process_frame(frame):
    global selected_finger, last_right_pos, finger_values, accumulated_movement

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_height, img_width = frame.shape[:2]

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Determine hand side
            side = get_hand_side(hand_landmarks, img_width)

            if side == HandSide.LEFT:
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks

            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Finger selection with left hand (strict single finger check)
    if left_hand:
        new_selection = check_single_finger_up(left_hand)
        if new_selection:
            selected_finger = new_selection
            # Reset tracking when selection changes
            last_right_pos = None
            accumulated_movement = 0

    # Value adjustment with right hand (only when a finger is selected)
    if right_hand and selected_finger is not None:
        wrist = right_hand.landmark[0]
        current_pos = (wrist.x, wrist.y)

        if last_right_pos is None:
            last_right_pos = current_pos
        else:
            # Calculate value change based on movement
            value_change = update_value(current_pos, last_right_pos)
            if value_change != 0:
                # Apply change with bounds checking
                new_value = finger_values[selected_finger] + value_change
                if (value_change > 0 and new_value <= MAX_VALUE) or \
                        (value_change < 0 and new_value >= MIN_VALUE):
                    finger_values[selected_finger] = new_value

            # Update last position for next frame
            last_right_pos = current_pos

    # Display UI
    cv2.putText(frame, "Precision Finger Control", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    # Display selected finger
    if selected_finger:
        cv2.putText(frame, f"Selected: {selected_finger.name}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Raise ONE left finger to select", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display finger values with directional indicators
    for i, finger in enumerate(Finger):
        value = finger_values[finger]

        # Color and symbol based on value
        if value == 0:
            color = (255, 255, 255)  # White
            symbol = " "
        elif value > 0:
            intensity = min(255, int(255 * (value / MAX_VALUE)))
            color = (0, intensity, 0)  # Green
            symbol = "+"
        else:
            intensity = min(255, int(255 * (abs(value) / abs(MIN_VALUE))))
            color = (intensity, 0, 0)  # Red
            symbol = "-"

        # Highlight selected finger
        if finger == selected_finger:
            color = (0, 255, 255)  # Yellow
            symbol = "▶" if value >= 0 else "◀"

        value_text = f"{finger.name}: {symbol}{abs(value):03d}"
        cv2.putText(frame, value_text, (10, 110 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Right hand instructions
    if selected_finger:
        current_value = finger_values[selected_finger]
        direction = "UP(+)  " if current_value >= 0 else "DOWN(-)"
        cv2.putText(frame, f"Right hand: {direction} Value: {current_value:+04d}",
                    (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        cv2.imshow('Precision Finger Control', processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()