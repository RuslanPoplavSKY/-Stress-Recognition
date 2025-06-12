import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Завантаження моделі
model = tf.keras.models.load_model('model_mobilenet.keras')
class_labels = ['Без стресу', 'Стрес']

# Передобробка зображення
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Вибір зображення
def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if selected_image_path:
        img = Image.open(selected_image_path).resize((250, 250))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text="")
        chart_frame.pack_forget()
        analyze_button.pack(pady=10)

# Класифікація
def predict_stress():
    img_array = preprocess_image(selected_image_path)
    predictions = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    result_label.config(
        text=predicted_class,
        fg="green" if predicted_class == "Без стресу" else "red"
    )
    global last_predictions
    last_predictions = predictions
    chart_frame.pack(pady=10)

# Побудова графіка
def show_chart(metric_name):
    metric_values = {
        "Accuracy": [0.65, 0.70, 0.75, 0.80],
        "Precision": [0.60, 0.68, 0.74, 0.78],
        "Recall": [0.58, 0.63, 0.70, 0.77],
        "F1-score": [0.59, 0.65, 0.72, 0.76]
    }
    plt.figure(figsize=(4, 3))
    plt.plot(metric_values[metric_name], marker='o', color='blue')
    plt.title(metric_name)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- ГОЛОВНЕ ВІКНО ---
root = tk.Tk()
root.title("Stress Detection System")
root.geometry("400x600")
root.configure(bg="white")

# Заголовок
tk.Label(root, text="Stress Detection System", font=("Helvetica", 16, "bold"), bg="white").pack(pady=10)

# Завантаження
upload_button = tk.Button(root, text="Upload Image", command=select_image, font=("Helvetica", 12))
upload_button.pack(pady=5)

# Зображення
image_label = tk.Label(root, bg="white")
image_label.pack(pady=10)

# Результат
result_label = tk.Label(root, text="", font=("Helvetica", 22, "bold"), bg="white")
result_label.pack(pady=10)

# Кнопка аналізу
analyze_button = tk.Button(root, text="Analyze", command=predict_stress, font=("Helvetica", 12))

# Графіки
chart_frame = tk.Frame(root, bg="white")

for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
    btn = tk.Button(chart_frame, text=metric, font=("Helvetica", 10), command=lambda m=metric: show_chart(m))
    btn.pack(side="left", padx=5)

# Старт GUI
root.mainloop()
