import cv2
import time
import math
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_draw

# ==================== SETUP ====================
hands_detector = mp_hands_module.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

FINGER_TIPS = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks, handedness):
    fingers = []
    lm = hand_landmarks.landmark
    if handedness == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    for tip_id in FINGER_TIPS[1:]:
        fingers.append(1 if lm[tip_id].y < lm[tip_id - 2].y else 0)
    return sum(fingers)

def calculate(num1, operator, num2):
    try:
        if operator == '+': return num1 + num2, None
        elif operator == '-': return num1 - num2, None
        elif operator == '*': return num1 * num2, None
        elif operator == '/':
            if num2 == 0:
                return None, "ERROR: Div by 0\nCannot divide by zero"
            return round(num1 / num2, 2), None
    except Exception as e:
        return None, str(e)

def draw_ui(frame, state, num1, operator, num2, result, error, stable_count, stable_threshold, fps, total_fingers):
    h, w, _ = frame.shape

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Finger Calculator by Kaif Kohli 18 - Complete Calculator",
                (15, 47), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps}", (w - 120, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ---- EQUATION DISPLAY (live) ----
    eq_bg = frame.copy()
    cv2.rectangle(eq_bg, (0, 75), (w, 145), (10, 10, 40), -1)
    cv2.addWeighted(eq_bg, 0.85, frame, 0.15, 0, frame)

    if state == 'num1':
        eq_text = f"[ {total_fingers} ]   ?   ?"
    elif state == 'operator':
        eq_text = f"[ {num1} ]   [ ? ]   ?"
    elif state == 'num2':
        eq_text = f"[ {num1} ]  {operator}  [ {total_fingers} ]"
    else:
        if error:
            eq_text = f"{num1}  {operator}  {num2}  =  ERROR"
        else:
            eq_text = f"{num1}  {operator}  {num2}  =  {result}"

    cv2.putText(frame, eq_text, (20, 125),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)

    # ---- STEP INDICATORS ----
    steps = ['1: NUM1', '2: OPER', '3: NUM2', '4: RESULT']
    states_order = ['num1', 'operator', 'num2', 'result']
    sw = w // 4
    for i, (step, st) in enumerate(zip(steps, states_order)):
        x1 = i * sw + 5
        is_active = (st == state)
        is_done = states_order.index(state) > i
        color = (0, 200, 80) if is_done else ((0, 120, 255) if is_active else (50, 50, 50))
        cv2.rectangle(frame, (x1, 150), (x1 + sw - 10, 185), color, -1)
        cv2.putText(frame, step, (x1 + 8, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255) if (is_active or is_done) else (120, 120, 120), 2)

    # ---- INSTRUCTIONS ----
    instructions = {
        'num1':     ("Show NUM1 with fingers & hold still", (0, 255, 255)),
        'operator': ("Show OPERATOR:  1=+   2=-   3=*   4=/", (0, 200, 255)),
        'num2':     ("Show NUM2 with fingers & hold still", (0, 255, 255)),
        'result':   ("Done Press 'r' to restart", (0, 255, 100)),
    }
    instr, instr_color = instructions.get(state, ("", (255, 255, 255)))

    instr_bg = frame.copy()
    cv2.rectangle(instr_bg, (0, h - 130), (w, h - 80), (0, 0, 0), -1)
    cv2.addWeighted(instr_bg, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, instr, (20, h - 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, instr_color, 2)

    # ---- COUNTDOWN TIMER ----
    if state != 'result':
        progress = stable_count / stable_threshold
        secs_left = max(0, math.ceil((stable_threshold - stable_count) / 30))

        # Ring
        cx, cy, r = w - 75, h - 200, 50
        cv2.circle(frame, (cx, cy), r, (50, 50, 50), 6)
        angle = int(360 * progress)
        ring_color = (0, 255, 0) if progress > 0.6 else (0, 165, 255)
        for deg in range(angle):
            rad = math.radians(deg - 90)
            x = int(cx + r * math.cos(rad))
            y = int(cy + r * math.sin(rad))
            cv2.circle(frame, (x, y), 4, ring_color, -1)

        # Number inside ring
        label = "GO!" if progress >= 1.0 else str(secs_left)
        font_scale = 0.7 if progress >= 1.0 else 1.1
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)[0][0]
        cv2.putText(frame, label, (cx - tw // 2, cy + 12),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)
        cv2.putText(frame, "HOLD", (cx - 22, cy + r + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Progress bar
        bar_x1, bar_y1 = 10, h - 55
        bar_x2, bar_y2 = w - 110, h - 30
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)
        fill_w = int((bar_x2 - bar_x1) * progress)
        bar_col = (0, 255, 0) if progress > 0.6 else (0, 165, 255)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), bar_col, -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (100, 100, 100), 2)
        cv2.putText(frame, "Hold still...", (bar_x1 + 5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ---- FINGER COUNT DISPLAY ----
    count_bg = frame.copy()
    cv2.rectangle(count_bg, (0, h - 80), (200, h - 130), (0,0,0), -1)
    cv2.addWeighted(count_bg, 0.0, frame, 1.0, 0, frame)
    cv2.putText(frame, f"Fingers: {total_fingers}", (15, h - 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2)

    # ---- RESULT BOX (center) ----
    if state == 'result':
        bx, by = w // 2 - 260, h // 2 - 90
        bw2, bh2 = 520, 180

        if error:
            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), (0, 0, 150), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), (0, 0, 255), 3)
            cv2.putText(frame, "RESULT", (bx + 185, by + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            lines = error.split('\n')
            cv2.putText(frame, f"{num1} / {num2} = {lines[0]}",
                        (bx + 20, by + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            if len(lines) > 1:
                cv2.putText(frame, lines[1], (bx + 100, by + 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 1)
        else:
            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), (0, 70, 0), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), (0, 255, 0), 3)
            cv2.putText(frame, "RESULT", (bx + 185, by + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            res_str = f"{num1}  {operator}  {num2}  =  {result}"
            cv2.putText(frame, res_str, (bx + 20, by + 110),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        cv2.putText(frame, "Press 'r' = New Calculation  |  'q' = Quit",
                    (bx + 30, by + bh2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    return frame


# ==================== MAIN ====================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = 'num1'
    num1, num2 = 0, 0
    operator = None
    result = None
    error = None

    stable_value = -1
    stable_count = 0
    STABLE_THRESHOLD = 90  # ~3 seconds at 30fps

    prev_time = 0

    print("\n" + "="*50)
    print("   Finger Calculator Started")
    print("   Hold fingers still for 3 seconds to confirm")
    print("   Operator: 1=+  2=-  3=*  4=/")
    print("   'r' = Restart  |  'q' = Quit")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time + 0.001))
        prev_time = curr_time

        detection = hands_detector.process(rgb)
        total_fingers = 0

        if detection.multi_hand_landmarks and detection.multi_handedness:
            for hand_lm, hand_info in zip(detection.multi_hand_landmarks, detection.multi_handedness):
                label = hand_info.classification[0].label
                count = count_fingers(hand_lm, label)
                total_fingers += count
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands_module.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=5),
                    mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        # State machine
        if state != 'result':
            if total_fingers == stable_value:
                stable_count += 1
            else:
                stable_value = total_fingers
                stable_count = 0

            if stable_count >= STABLE_THRESHOLD:
                if state == 'num1':
                    num1 = total_fingers
                    state = 'operator'
                    stable_count = 0
                    stable_value = -1
                    print(f"✓ NUM1 = {num1}")

                elif state == 'operator':
                    op_map = {1: '+', 2: '-', 3: '*', 4: '/'}
                    if total_fingers in op_map:
                        operator = op_map[total_fingers]
                        state = 'num2'
                        stable_count = 0
                        stable_value = -1
                        print(f"✓ OPERATOR = {operator}")

                elif state == 'num2':
                    num2 = total_fingers
                    result, error = calculate(num1, operator, num2)
                    state = 'result'
                    stable_count = 0
                    print(f"✓ NUM2 = {num2}")
                    print(f"✓ Result: {num1} {operator} {num2} = {result if result is not None else error}")

        frame = draw_ui(frame, state, num1, operator, num2,
                        result, error, stable_count, STABLE_THRESHOLD, fps, total_fingers)

        cv2.imshow("Finger Calculator by Kaif Kohli 18", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state = 'num1'
            num1, num2 = 0, 0
            operator = None
            result = None
            error = None
            stable_value = -1
            stable_count = 0
            print("\n↺ Restarted")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()