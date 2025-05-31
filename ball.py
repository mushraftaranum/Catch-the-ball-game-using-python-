import cv2
import mediapipe as mp
import random
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Basket settings
basket_x = 250
basket_y = 400
basket_width = 100
basket_height = 30
basket_color = (0, 255, 0)

# Ball class
class Ball:
    def __init__(self):
        self.x = random.randint(0, 600)
        self.y = 0
        self.radius = 20
        self.color = (0, 0, 255)
        self.speed = random.randint(5, 8)
        self.caught = False

    def move(self):
        if not self.caught:
            self.y += self.speed

    def draw(self, frame):
        if not self.caught:
            cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)

# Game variables
balls = [Ball()]
score = 0
lives = 3
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Create black background
    black_bg = np.zeros_like(img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = img.shape
            points = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append([x, y])
            points = np.array(points, dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            hand_only = cv2.bitwise_and(img, img, mask=mask)
            black_bg = cv2.bitwise_or(black_bg, hand_only)
    else:
        black_bg = np.zeros_like(img)

    # If hand detected, get the x of the wrist to move basket
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = img.shape
            wrist = hand_landmarks.landmark[0]
            basket_x = int(wrist.x * w) - basket_width // 2
            # Clamp basket_x inside window width
            basket_x = max(0, min(basket_x, w - basket_width))

    # Move and draw balls
    for ball in balls:
        ball.move()
        # Check if ball caught by basket
        if (basket_x < ball.x < basket_x + basket_width) and (basket_y < ball.y + ball.radius < basket_y + basket_height):
            if not ball.caught:
                ball.caught = True
                score += 1
                # Add new ball after catching
                balls.append(Ball())
        # Remove ball if it falls beyond screen
        if ball.y - ball.radius > img.shape[0]:
            if not ball.caught:
                lives -= 1
                balls.remove(ball)
                balls.append(Ball())
        ball.draw(black_bg)

    # Draw basket on black background
    cv2.rectangle(black_bg, (basket_x, basket_y), (basket_x + basket_width, basket_y + basket_height), basket_color, -1)

    # Draw score and lives
    cv2.putText(black_bg, f"Score: {score}", (20, 40), font, 1, (255, 255, 255), 2)
    cv2.putText(black_bg, f"Lives: {lives}", (500, 40), font, 1, (0, 0, 255), 2)

    # Draw hand landmarks on black background
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(black_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Game Over condition
    if lives <= 0:
        cv2.putText(black_bg, "GAME OVER", (200, 250), font, 2, (0, 0, 255), 5)
        cv2.imshow("Catch the Ball - Hand Highlighted", black_bg)
        cv2.waitKey(3000)
        break

    cv2.imshow("Catch the Ball - Hand Highlighted", black_bg)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
