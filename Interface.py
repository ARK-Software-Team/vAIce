import os
import csv
import time
import librosa
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import matplotlib.pyplot as plt
import speech_recognition as sr
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Konu analizi için kaydedilen konu kelimeleri yüklemesi 
def load_topics():
    topics = {}
    with open('topics.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            category = row[0]
            keywords = row[1:]
            topics[category] = keywords
    return topics
TOPICS = load_topics()

class AudioHistogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Amplitude and Frequency")
        self.is_running = False
        self.audio_data = np.array([])
        self.start_time = None
        self.sample_rate = 22050
        self.stop_recognition = False

        # Matplotlib figürleri oluştur
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 5), dpi=100)
        self.fig.tight_layout(pad=3.0)

        # Time serisi grafiği (Amplitude)
        self.ax1.set_title("Amplitude - Time Graph")
        self.ax1.set_xlabel("Time")
        self.ax1.set_ylabel("Amplitude")

        # Frequency spektrumu grafiği (Spectrogram)
        self.ax2.set_title("Frequency Spectrogram - Color Histogram")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("Frequency (Hz)")

        # Matplotlib'i tkinter canvas içine ekle
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Konu Tablosu oluşturma
        self.table_frame = ttk.Frame(root)
        self.table_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.table = ttk.Treeview(self.table_frame, columns=('Category', 'Precision', 'Support'), show='headings')
        self.table.heading('Category', text='Category')
        self.table.heading('Precision', text='Precision')
        self.table.heading('Support', text='Support')
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Start/Stop butonu
        self.toggle_button = ttk.Button(
            root, text="Start", command=self.toggle_recording
        )
        self.toggle_button.pack(side=tk.BOTTOM)
        self.update_id = None
        self.colorbar = None

        self.recognizer = sr.Recognizer()
        self.text_output = tk.Text(root, height=5, width=50)
        self.text_output.pack(side=tk.BOTTOM)

    def toggle_recording(self):
        if not self.is_running:
            self.is_running = True
            self.stop_recognition = False
            self.toggle_button.config(text="Stop")
            self.audio_data = np.array([])
            self.start_time = time.time()
            self.full_text = ""  # Reset full text
            self.record_audio_loop()
            threading.Thread(target=self.speech_to_text_loop, daemon=True).start()
        else:
            self.is_running = False
            self.stop_recognition = True
            self.toggle_button.config(text="Start")
            if self.update_id:
                self.root.after_cancel(self.update_id)

    def record_audio_loop(self):
        if self.is_running:
            self.update_visualizations()
            self.update_id = self.root.after(100, self.record_audio_loop)

    def update_visualizations(self):
        duration = 0.1
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()
        self.audio_data = np.concatenate((self.audio_data, audio))

        current_time = time.time() - self.start_time
        times = np.linspace(0, current_time, len(self.audio_data))

        self.ax1.clear()
        self.ax1.plot(times, self.audio_data)
        self.ax1.set_title("Amplitude - Time Serisi")
        self.ax1.set_xlabel("Time")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.set_xlim(0, max(10, current_time))

        s_data = librosa.stft(self.audio_data)
        s_database = librosa.amplitude_to_db(np.abs(s_data), ref=np.max)

        self.ax2.clear()
        img = self.ax2.imshow(
            s_database,
            aspect="auto",
            cmap="plasma",
            origin="lower",
            extent=[0, current_time, 0, self.sample_rate / 2],
        )
        self.ax2.set_title("Frequency Spektrumu - Renk Histogrami")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("Frequency (Hz)")
        self.ax2.set_xlim(0, max(10, current_time))

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(img, ax=self.ax2, format="%+2.0f dB")
        else:
            self.colorbar.update_normal(img)

        self.canvas.draw()

    def speech_to_text_loop(self):
        while self.is_running and not self.stop_recognition:
            try:
                with sr.Microphone() as source:
                    audio = self.recognizer.listen(
                        source, phrase_time_limit=10, timeout=1
                    )
                if not self.is_running:
                    break
                try:
                    text = self.recognizer.recognize_google(audio, language="tr-TR")
                except sr.UnknownValueError:
                    text = self.recognizer.recognize_google(audio, language="en-US")
                if self.is_running:
                    self.root.after(0, self.update_text, text)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("API unavailable")

    def calculate_metrics(self, true_labels, predicted_labels):
        categories = list(TOPICS.keys())
        metrics = precision_recall_fscore_support(true_labels, predicted_labels, labels=categories, average=None, zero_division=0)
        return {cat: [metric[0], metric[3]] for cat, metric in zip(categories, zip(*metrics))}

    def update_table(self):
        topic_counts = {topic: 0 for topic in TOPICS.keys()}
        words = self.full_text.lower().split()
        for word in words:
            for topic, keywords in TOPICS.items():
                if word in keywords:
                    topic_counts[topic] += 1
        
        total_count = sum(topic_counts.values())
        
        self.table.delete(*self.table.get_children())
        for category, count in topic_counts.items():
            precision = count / total_count if total_count > 0 else 0
            self.table.insert('', 'end', values=[category, f"{precision:.2f}", count])

    def update_text(self, text):
        self.text_output.insert(tk.END, text + "\n")
        self.text_output.see(tk.END)
        self.full_text += " " + text  # Accumulate full text
        self.update_table()

# Ana uygulama penceresini oluştur
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioHistogramApp(root)
    root.mainloop()
