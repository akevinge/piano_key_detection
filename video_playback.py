import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2


class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Slow Motion Video Player")

        # Video variables
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.playback_speed = 0.1

        # UI Setup
        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Frame label
        self.frame_label = tk.Label(
            root, text="Frame: 0", font=("Arial", 12), bg="black", fg="white"
        )
        self.frame_label.place(x=10, y=10)

        # Control buttons
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(control_frame, text="Open Video", command=self.open_video).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        tk.Button(
            control_frame, text="Play/Pause (Space)", command=self.toggle_play
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Frame jump controls
        tk.Label(control_frame, text="Jump to frame:").pack(side=tk.LEFT, padx=5)
        self.frame_entry = tk.Entry(control_frame, width=10)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Go", command=self.jump_to_frame).pack(
            side=tk.LEFT, padx=5
        )

        # Speed control
        tk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_label = tk.Label(control_frame, text="0.10x", width=6)
        self.speed_label.pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(
            control_frame,
            from_=0.01,
            to=2.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self.update_speed,
            showvalue=0,
        )
        self.speed_slider.set(0.1)
        self.speed_slider.pack(side=tk.LEFT, padx=5)

        # Key bindings
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.go_back())
        self.root.bind("<Right>", lambda e: self.go_forward())

        # Start update loop
        self.update_frame()

    def open_video(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(filepath)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.is_playing = False
            self.display_current_frame()

    def toggle_play(self):
        if self.cap:
            self.is_playing = not self.is_playing

    def go_back(self):
        if self.cap and self.current_frame > 0:
            self.current_frame -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.display_current_frame()

    def go_forward(self):
        if self.cap and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.display_current_frame()

    def jump_to_frame(self):
        if not self.cap:
            return

        try:
            target_frame = int(self.frame_entry.get())
            if 0 <= target_frame < self.total_frames:
                self.current_frame = target_frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.display_current_frame()
            else:
                self.frame_entry.delete(0, tk.END)
                self.frame_entry.insert(0, f"Range: 0-{self.total_frames-1}")
        except ValueError:
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, "Invalid number")

    def update_speed(self, value):
        self.playback_speed = float(value)
        self.speed_label.config(text=f"{self.playback_speed:.2f}x")

    def display_current_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                h, w = frame.shape[:2]
                aspect = w / h

                if canvas_width / canvas_height > aspect:
                    new_h = canvas_height
                    new_w = int(canvas_height * aspect)
                else:
                    new_w = canvas_width
                    new_h = int(canvas_width / aspect)

                frame = cv2.resize(frame, (new_w, new_h))

            # Convert to PhotoImage
            img = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=img)

            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.photo,
                anchor=tk.CENTER,
            )

            # Update frame label
            self.frame_label.config(
                text=f"Frame: {self.current_frame} / {self.total_frames}"
            )

    def update_frame(self):
        if self.is_playing and self.cap:
            if self.current_frame < self.total_frames - 1:
                self.current_frame += 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.display_current_frame()
            else:
                self.is_playing = False

        # Calculate delay for 0.1x speed
        # Assuming 30 fps, delay should be 1000ms / (30 * 0.1) ≈ 333ms
        delay = int(1000 / (30 * self.playback_speed))
        self.root.after(delay, self.update_frame)

    def __del__(self):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    player = VideoPlayer(root)
    root.mainloop()
