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
import tempfile

# MoviePy imports (will be imported when needed)
try:
    from moviepy.editor import ImageSequenceClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not installed. Video saving will use OpenCV fallback.")
    print("Install with: pip install moviepy")

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
        self.video_frames = []  # Store frames for MoviePy
        self.hand_thread = None
        self.hand_thread_running = False

        self.video_filename = None
        self.audio_filename = None
        
        # Eye gaze tracking variables
        self.gaze_data = []
        self.looking_at_camera_count = 0
        self.total_gaze_samples = 0

        master.bind("<KeyPress>", self.on_key_press)
        master.focus_set()

    def on_key_press(self, event):
        if event.keysym == "space":
            self.toggle_recording()

    def calculate_eye_gaze_direction(self, face_landmarks, frame_width, frame_height):
        """
        Calculate eye gaze direction with STRICT camera detection
        Returns: (gaze_x, gaze_y, is_looking_at_camera)
        """
        # Use reliable eye landmarks that definitely exist in MediaPipe Face Mesh
        # Left eye landmarks
        left_eye_outer = [face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]   # Left eye outer corner
        left_eye_inner = [face_landmarks.landmark[133].x, face_landmarks.landmark[133].y] # Left eye inner corner
        left_eye_top = [face_landmarks.landmark[159].x, face_landmarks.landmark[159].y]    # Left eye top
        left_eye_bottom = [face_landmarks.landmark[145].x, face_landmarks.landmark[145].y] # Left eye bottom
        
        # Right eye landmarks  
        right_eye_inner = [face_landmarks.landmark[362].x, face_landmarks.landmark[362].y] # Right eye inner corner
        right_eye_outer = [face_landmarks.landmark[263].x, face_landmarks.landmark[263].y] # Right eye outer corner
        right_eye_top = [face_landmarks.landmark[386].x, face_landmarks.landmark[386].y]    # Right eye top
        right_eye_bottom = [face_landmarks.landmark[374].x, face_landmarks.landmark[374].y] # Right eye bottom
        
        # Calculate eye centers (geometric center of eye landmarks)
        left_eye_center_x = (left_eye_outer[0] + left_eye_inner[0]) / 2
        left_eye_center_y = (left_eye_top[1] + left_eye_bottom[1]) / 2
        
        right_eye_center_x = (right_eye_inner[0] + right_eye_outer[0]) / 2
        right_eye_center_y = (right_eye_top[1] + right_eye_bottom[1]) / 2
        
        # Calculate eye dimensions
        left_eye_width = abs(left_eye_inner[0] - left_eye_outer[0])
        left_eye_height = abs(left_eye_bottom[1] - left_eye_top[1])
        
        right_eye_width = abs(right_eye_outer[0] - right_eye_inner[0])
        right_eye_height = abs(right_eye_bottom[1] - right_eye_top[1])
        
        # Use pupil/iris center landmarks (these should exist in MediaPipe)
        # If these don't exist, we'll estimate from eye geometry
        try:
            # Try to get more precise iris points
            left_iris_x = face_landmarks.landmark[468].x if len(face_landmarks.landmark) > 468 else left_eye_center_x
            left_iris_y = face_landmarks.landmark[468].y if len(face_landmarks.landmark) > 468 else left_eye_center_y
            
            right_iris_x = face_landmarks.landmark[473].x if len(face_landmarks.landmark) > 473 else right_eye_center_x  
            right_iris_y = face_landmarks.landmark[473].y if len(face_landmarks.landmark) > 473 else right_eye_center_y
        except:
            # Fallback to geometric centers
            left_iris_x = left_eye_center_x
            left_iris_y = left_eye_center_y
            right_iris_x = right_eye_center_x
            right_iris_y = right_eye_center_y
        
        # Calculate gaze direction (normalized to -1 to 1)
        left_gaze_x = (left_iris_x - left_eye_center_x) / (left_eye_width / 2) if left_eye_width > 0 else 0
        left_gaze_y = (left_iris_y - left_eye_center_y) / (left_eye_height / 2) if left_eye_height > 0 else 0
        
        right_gaze_x = (right_iris_x - right_eye_center_x) / (right_eye_width / 2) if right_eye_width > 0 else 0
        right_gaze_y = (right_iris_y - right_eye_center_y) / (right_eye_height / 2) if right_eye_height > 0 else 0
        
        # Average both eyes
        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2
        avg_gaze_y = (left_gaze_y + right_gaze_y) / 2
        
        # VERY STRICT thresholds for camera detection
        horizontal_threshold = 0.05  # Very strict horizontal threshold
        vertical_threshold = 0.1    # Strict vertical threshold
        
        # Face orientation check using nose
        try:
            nose_tip = face_landmarks.landmark[1]  # Nose tip
            face_center_x = 0.5  # Center of frame
            nose_deviation = abs(nose_tip.x - face_center_x)
            face_orientation_threshold = 0.1  # Face must be mostly centered
            face_facing_forward = nose_deviation < face_orientation_threshold
        except:
            face_facing_forward = True  # Default to true if nose detection fails
        
        # Determine if looking at camera with STRICT criteria
        eyes_centered_horizontally = abs(avg_gaze_x) < horizontal_threshold
        eyes_centered_vertically = abs(avg_gaze_y) < vertical_threshold
        
        is_looking_at_camera = eyes_centered_horizontally and eyes_centered_vertically and face_facing_forward
        
        return avg_gaze_x, avg_gaze_y, is_looking_at_camera

    def save_video_with_moviepy(self, frames, fps=20):
        """Save video frames using MoviePy if available, otherwise fallback to OpenCV"""
        if not frames:
            print("No frames to save")
            return None
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_video_path = f"hand_eye_recording_{timestamp}.mp4"
        
        if MOVIEPY_AVAILABLE:
            try:
                # Convert frames to the format expected by MoviePy
                # MoviePy expects RGB format
                rgb_frames = []
                for frame in frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frames.append(rgb_frame)
                
                # Create video clip from frames
                clip = ImageSequenceClip(rgb_frames, fps=fps)
                
                # Write video file
                clip.write_videofile(final_video_path, codec='libx264', verbose=False, logger=None)
                
                print(f"Video saved using MoviePy: {final_video_path}")
                return final_video_path
                
            except Exception as e:
                print(f"Error saving video with MoviePy: {e}")
                print("Falling back to OpenCV VideoWriter...")
                return self.save_video_with_opencv(frames, fps, final_video_path)
        else:
            # Fallback to OpenCV VideoWriter
            return self.save_video_with_opencv(frames, fps, final_video_path)
    
    def save_video_with_opencv(self, frames, fps, filename):
        """Fallback method using OpenCV VideoWriter"""
        try:
            if not frames:
                return None
                
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            for frame in frames:
                video_writer.write(frame)
            
            video_writer.release()
            print(f"Video saved using OpenCV: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving video with OpenCV: {e}")
            return None

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

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_frames = []  # Reset frames list
        
        cv2.namedWindow("Live Hand & Eye Gaze Detection", cv2.WINDOW_NORMAL)

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

            # Process face mesh for eye gaze tracking
            results_face = self.face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # Calculate eye gaze direction
                    gaze_x, gaze_y, is_looking_at_camera = self.calculate_eye_gaze_direction(
                        face_landmarks, frame_width, frame_height)
                    
                    # Store gaze data for analysis
                    self.gaze_data.append((gaze_x, gaze_y, is_looking_at_camera))
                    self.total_gaze_samples += 1
                    if is_looking_at_camera:
                        self.looking_at_camera_count += 1
                    
                    # Draw eye landmarks
                    left_eye_indices = [33, 133, 159, 145, 153, 154, 155, 173]
                    right_eye_indices = [362, 263, 386, 374, 380, 381, 382, 398]
                    
                    for idx in left_eye_indices + right_eye_indices:
                        lm = face_landmarks.landmark[idx]
                        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                    # Display gaze information with more detail
                    gaze_status = "👁️ CAMERA" if is_looking_at_camera else "👁️ AWAY"
                    color = (0, 255, 0) if is_looking_at_camera else (0, 0, 255)
                    
                    cv2.putText(frame, f"Gaze: {gaze_status}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"X: {gaze_x:.3f}, Y: {gaze_y:.3f}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Show thresholds for debugging
                    cv2.putText(frame, f"Threshold: ±0.1 (H), ±0.15 (V)", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

            cv2.imshow("Live Hand & Eye Gaze Detection", frame)
            
            # Store frame for MoviePy
            self.video_frames.append(frame.copy())

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Press SPACE to stop recording")
            elif key == 27:  # ESC to quit detection early
                self.hand_thread_running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save video using MoviePy
        self.video_filename = self.save_video_with_moviepy(self.video_frames)

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
        self.gaze_data = []
        self.looking_at_camera_count = 0
        self.total_gaze_samples = 0

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

        threading.Thread(target=self.analyze_audio_and_gaze, daemon=True).start()

    def analyze_audio_and_gaze(self):
        print("Analyze audio and gaze")

        # Audio analysis
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

        # Calculate pitch variation and subtract baseline of 430
        raw_pitch_variation = np.std(filtered_pitch) if len(filtered_pitch) > 0 else 0
        pitch_variation = max(0, raw_pitch_variation - 430)  # Subtract baseline, don't go negative

        # Gaze analysis
        camera_attention_percentage = 0
        if self.total_gaze_samples > 0:
            camera_attention_percentage = (self.looking_at_camera_count / self.total_gaze_samples) * 100

        results = []
        
        # Audio results
        if rms < 0.005:
            results.append("🔈 Too quiet")
        elif rms > 0.1:
            results.append("🔊 Too loud")
        else:
            results.append("✅ Volume: Normal")

        # Show both raw and adjusted pitch variation for comparison
        results.append(f"Pitch variation (raw): {raw_pitch_variation:.2f} Hz")
        results.append(f"Pitch variation (adjusted): {pitch_variation:.2f} Hz")
        
        # Gaze results with stricter criteria
        if camera_attention_percentage >= 80:
            results.append("👁️ ✅ Excellent eye contact")
        elif camera_attention_percentage >= 60:
            results.append("👁️ ✅ Good eye contact")
        elif camera_attention_percentage >= 40:
            results.append("👁️ ⚠️ Fair eye contact")
        elif camera_attention_percentage >= 20:
            results.append("👁️ ❌ Poor eye contact")
        else:
            results.append("👁️ ❌ Very poor eye contact")
            
        results.append(f"Camera attention: {camera_attention_percentage:.1f}%")

        extra_info = f"RMS (volume): {rms:.4f}\nGaze samples: {self.total_gaze_samples}\nBaseline subtracted: 430 Hz"

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
        cv2.destroyAllWindows()
        self.master.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("460x350")
    app = AudioVideoAnalyzerApp(root)
    root.mainloop()