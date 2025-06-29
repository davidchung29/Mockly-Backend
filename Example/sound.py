import tkinter as tk
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import signal
import sys
import datetime
import wave
from scipy.signal import medfilt
import cv2
import mediapipe as mp
import os

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024

def handle_sigint(sig, frame):
    print("Exiting via Ctrl+C...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

class AudioVideoAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Mock Interview Audio & Video Analyzer")
        master.protocol("WM_DELETE_WINDOW", self.cleanup_and_exit)

        self.label = tk.Label(master, text="Press SPACE to start/stop recording", font=("Arial", 14))
        self.label.pack(pady=10)

        self.results_label = tk.Label(master, text="", fg="blue", font=("Arial", 12))
        self.results_label.pack(pady=10)

        self.play_button = tk.Button(master, text="🔁 Play Recording", command=self.play_recording, state=tk.DISABLED)
        self.play_button.pack(pady=5)

        # Audio data
        self.recording = False
        self.audio_data = []
        self.pitch_data = []
        self.time_data = []
        self.start_time = None
        self.plot_running = False

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = None
        self.face_mesh = None

        self.cap = None
        self.video_writer = None
        self.hand_thread = None
        self.hand_thread_running = False

        self.video_filename = None
        self.audio_filename = None

        master.bind("<space>", self.toggle_recording)

    def audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            samples = indata[:, 0]
            self.audio_data.extend(samples)

            # Calculate dominant frequency for pitch
            fft = np.fft.rfft(samples)
            freqs = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
            magnitude = np.abs(fft)
            dom_freq = freqs[np.argmax(magnitude)]

            current_time = time.time() - self.start_time
            self.time_data.append(current_time)
            self.pitch_data.append(dom_freq)

    def hand_eye_detection_loop(self):
        self.cap = cv2.VideoCapture(0)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = f"hand_eye_recording_{timestamp}.mp4"
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20
        self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, fps, (frame_width, frame_height))
        print(f"Saving webcam video to {self.video_filename}")

        cv2.namedWindow("Live Hand & Eye Detection", cv2.WINDOW_NORMAL)

        # Indices for eye landmarks (from MediaPipe Face Mesh 468 landmarks)
        left_eye_indices = [33, 133, 159, 145, 153, 154, 155, 133, 173]
        right_eye_indices = [362, 263, 386, 374, 380, 381, 382, 263, 398]

        def eye_openness(landmarks, eye_indices):
            top_lid = landmarks.landmark[eye_indices[2]]
            bottom_lid = landmarks.landmark[eye_indices[4]]
            return abs(top_lid.y - bottom_lid.y)

        while self.hand_thread_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands
            results_hands = self.hands.process(image_rgb)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Draw bounding box around hand
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    x_min = int(min(x_coords) * frame_width)
                    x_max = int(max(x_coords) * frame_width)
                    y_min = int(min(y_coords) * frame_height)
                    y_max = int(max(y_coords) * frame_height)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Process face mesh (eyes)
            results_face = self.face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # Draw face landmarks (optional)
                    # mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)

                    # Draw eye landmarks
                    for idx in left_eye_indices + right_eye_indices:
                        lm = face_landmarks.landmark[idx]
                        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                    # Calculate eye openness (optional for your logic)
                    left_open = eye_openness(face_landmarks, left_eye_indices)
                    right_open = eye_openness(face_landmarks, right_eye_indices)
                    avg_open = (left_open + right_open) / 2
                    # Optionally show eye openness on frame
                    cv2.putText(frame, f"Eye Open: {avg_open:.3f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Live Hand & Eye Detection", frame)
            self.video_writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # You can add toggle logic here if you want
                print("Press SPACE to stop recording")
            elif key == 27:  # ESC to quit detection early (optional)
                self.hand_thread_running = False
                break

        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print(f"Video saved: {self.video_filename}")

    def toggle_recording(self, event=None):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        print("Start recording")
        self.label.config(text="🎙️ Recording... Press SPACE to stop.")
        self.results_label.config(text="")

        self.audio_data = []
        self.pitch_data = []
        self.time_data = []

        self.recording = True
        self.start_time = time.time()
        self.play_button.config(state=tk.DISABLED)

        self.stream = sd.InputStream(callback=self.audio_callback,
                                     channels=1,
                                     samplerate=SAMPLE_RATE,
                                     blocksize=BUFFER_SIZE)
        self.stream.start()

        self.plot_thread = threading.Thread(target=self.update_plot, daemon=True)
        self.plot_running = True
        self.plot_thread.start()

        self.hand_thread_running = True
        self.hand_thread = threading.Thread(target=self.hand_eye_detection_loop, daemon=True)
        self.hand_thread.start()

    def stop_recording(self):
        print("Stop recording")
        self.recording = False
        try:
            self.stream.stop()
        except Exception as e:
            print("Error stopping stream:", e)

        self.plot_running = False
        self.plot_thread.join()

        self.hand_thread_running = False
        if self.hand_thread is not None:
            self.hand_thread.join()

        self.label.config(text="🔍 Analyzing...")

        threading.Thread(target=self.analyze_audio, daemon=True).start()

    def analyze_audio(self):
        print("Analyze audio")

        data = np.array(self.audio_data)
        skip_samples = 2 * SAMPLE_RATE
        if len(data) > skip_samples:
            data_to_analyze = data[skip_samples:]
        else:
            data_to_analyze = data

        positive_samples = data_to_analyze[data_to_analyze > 0]
        rms = np.sqrt(np.mean(positive_samples ** 2)) if len(positive_samples) > 0 else 0

        filtered_pitch = [p for t, p in zip(self.time_data, self.pitch_data) if t >= 2]
        if len(filtered_pitch) >= 5:
            filtered_pitch = medfilt(filtered_pitch, kernel_size=5)
        else:
            filtered_pitch = np.array(filtered_pitch)

        pitch_variation = np.std(filtered_pitch) if len(filtered_pitch) > 0 else 0

        results = []
        if rms < 0.005:
            results.append("🔈 Too quiet")
        elif rms > 0.1:
            results.append("🔊 Too loud")
        else:
            results.append("✅ Volume: Normal")

        results.append(f"Pitch variation: {pitch_variation:.2f} Hz")

        extra_info = f"RMS (volume): {rms:.4f}"

        self.results_label.config(text="\n".join(results) + "\n\n" + extra_info)
        self.label.config(text="Press SPACE to start another recording")
        self.play_button.config(state=tk.NORMAL)

    def update_plot(self):
        print("Starting plot thread")
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("Live Pitch Tracking (Smoothed)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pitch (Hz)")
        line, = ax.plot([], [], 'b-')

        while self.plot_running:
            min_len = min(len(self.time_data), len(self.pitch_data))
            if min_len > 0:
                pitch_smoothed = medfilt(self.pitch_data[:min_len], kernel_size=5) if min_len >= 5 else self.pitch_data[:min_len]

                line.set_data(self.time_data[:min_len], pitch_smoothed)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
            time.sleep(0.1)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pitch_graph_{timestamp}.png"
        fig.savefig(filename)
        print(f"Pitch graph saved as {filename}")
        plt.close(fig)

    def play_recording(self):
        if self.audio_data:
            print("Playing recording")
            sd.play(np.array(self.audio_data), samplerate=SAMPLE_RATE)

    def cleanup_and_exit(self):
        print("Closing app...")
        self.plot_running = False
        self.recording = False
        self.hand_thread_running = False
        try:
            self.stream.stop()
        except:
            pass
        if self.cap is not None:
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        self.master.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("460x350")
    app = AudioVideoAnalyzerApp(root)
    root.mainloop()
