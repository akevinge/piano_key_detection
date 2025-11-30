import cv2
import mediapipe as mp
from tqdm import tqdm


class PianoHandTracker:
    def __init__(self, video_path):
        self.video_path = video_path

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(video_path)

    def process_video(self, show=True):
        """Process video frame by frame"""

        if not self.cap.isOpened():
            self.cleanup()
            return []

        results = []
        with tqdm(
            total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Video"
        ) as pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe Hands
                results_hands = self.hands.process(rgb_frame)
                results.append(results_hands)

                # Draw MediaPipe hand landmarks
                if show and results_hands.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(
                        results_hands.multi_hand_landmarks
                    ):
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(
                                color=(0, 255, 255), thickness=2, circle_radius=2
                            ),
                            self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2),
                        )

                        # Get hand label (Left/Right)
                        hand_label = (
                            results_hands.multi_handedness[hand_idx]
                            .classification[0]
                            .label
                        )

                        # Get wrist position for label placement
                        h, w, _ = frame.shape
                        wrist = hand_landmarks.landmark[0]
                        x, y = int(wrist.x * w), int(wrist.y * h)

                        cv2.putText(
                            frame,
                            f"{hand_label} Hand",
                            (x - 50, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )

                if show:
                    # Display frame
                    cv2.imshow("Piano Hand Tracking", frame)

                    # Press 'q' to quit, 'space' to pause
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord(" "):
                        cv2.waitKey(0)  # Wait for another key press

                pbar.update(1)

        self.cleanup()
        return results

    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    # Replace with your video file path
    video_path = "videos/bach-preludes.mp4"

    tracker = PianoHandTracker(video_path)
    tracker.process_video()
