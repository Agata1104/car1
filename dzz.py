import torch
import numpy as np
import cv2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageFilter

class DigitRecognizer:
    def __init__(self, learning_rate=0.01, random_state=42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.layer1 = None
        self.layer2 = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def activation(self, x):
        return torch.sigmoid(x)

    def _one_hot_encode(self, y):
        return torch.eye(10)[y]

    def loss(self, result, y):
        target = self._one_hot_encode(y)
        return torch.sum((result - target) ** 2)

    def _gradient(self, result, x, hidden_output, y):
        target = self._one_hot_encode(y)
        delta_output = (result - target) * result * (1 - result)
        grad_layer2 = torch.outer(delta_output, hidden_output)

        delta_hidden = (self.layer2.t() @ delta_output) * hidden_output * (1 - hidden_output)
        grad_layer1 = torch.outer(delta_hidden, x)

        return grad_layer1, grad_layer2

    def _back_propagation(self, result, x, hidden_output, y):
        grad_layer1, grad_layer2 = self._gradient(result, x, hidden_output, y)
        self.layer1 -= self.learning_rate * grad_layer1
        self.layer2 -= self.learning_rate * grad_layer2

    def fit(self, X, y, epochs=50, test_size=0.2, verbose=True):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state
        )

        torch.manual_seed(self.random_state)
        self.layer1 = torch.randn(32, 64, dtype=torch.float32) * 0.1
        self.layer2 = torch.randn(10, 32, dtype=torch.float32) * 0.1

        self.history = {'train_accuracy': [], 'test_accuracy': [], 'train_loss': []}

        for epoch in range(epochs):
            train_count = total_loss = 0
            indices = np.random.permutation(len(X_train))

            for i in indices:
                x, y_true = X_train[i], y_train[i]
                x_tensor = torch.tensor(x, dtype=torch.float32)

                hidden_output = self.activation(self.layer1 @ x_tensor)
                result = self.activation(self.layer2 @ hidden_output)

                total_loss += self.loss(result, y_true).item()
                self._back_propagation(result, x_tensor, hidden_output, y_true)
                train_count += torch.argmax(result) == y_true

            test_count = self._evaluate(X_test, y_test)
            train_acc = train_count / len(X_train) * 100
            test_acc = test_count / len(X_test) * 100
            avg_loss = total_loss / len(X_train)

            self.history['train_accuracy'].append(train_acc)
            self.history['test_accuracy'].append(test_acc)
            self.history['train_loss'].append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch + 1}: Train Acc: {train_acc:.1f}%, Test Acc: {test_acc:.1f}%, Loss: {avg_loss:.4f}")

        self.is_trained = True
        return self.history

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Модель не обучена! Сначала вызовите fit()")

        X_scaled = self.scaler.transform(X)
        predictions, probabilities = [], []

        for x in X_scaled:
            x_tensor = torch.tensor(x, dtype=torch.float32)
            hidden_output = self.activation(self.layer1 @ x_tensor)
            result = self.activation(self.layer2 @ hidden_output)

            predictions.append(torch.argmax(result).item())
            probabilities.append(result.detach().numpy())

        return np.array(predictions), np.array(probabilities)

    def predict_single(self, x):
        predictions, probabilities = self.predict([x])
        return predictions[0], np.max(probabilities[0])

    def _evaluate(self, X, y):
        count = 0
        for x, y_true in zip(X, y):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            hidden_output = self.activation(self.layer1 @ x_tensor)
            result = self.activation(self.layer2 @ hidden_output)
            count += torch.argmax(result) == y_true
        return count

    def score(self, X, y):
        predictions, _ = self.predict(X)
        return (predictions == y).mean() * 100

    def get_weights(self):
        return (self.layer1.detach().numpy(), self.layer2.detach().numpy()) if self.layer1 is not None else None


class DigitDrawingApp:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("Распознавание рукописных цифр")
        self.window.geometry("400x500")

        self.canvas = Canvas(self.window, width=280, height=280, bg='white', highlightthickness=1,
                             highlightbackground="black")
        self.canvas.pack(pady=10)

        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        control_frame = Frame(self.window)
        control_frame.pack(pady=10)

        self.predict_btn = Button(control_frame, text="Распознать", command=self.predict_digit, bg='lightblue',
                                  font=('Arial', 12))
        self.predict_btn.pack(side='left', padx=5)

        self.clear_btn = Button(control_frame, text="Очистить", command=self.clear_canvas, bg='lightcoral',
                                font=('Arial', 12))
        self.clear_btn.pack(side='left', padx=5)

        self.result_label = Label(self.window, text="Нарисуйте цифру и нажмите 'Распознать'", font=('Arial', 14),
                                  fg='blue')
        self.result_label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)

        self.last_x = self.last_y = None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=25, fill='black', capstyle=tk.ROUND,
                                    smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=25)
        else:
            self.canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill='black')
            self.draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=0)
        self.last_x, self.last_y = x, y

    def reset_position(self, event):
        self.last_x = self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Нарисуйте цифру и нажмите 'Распознать'", fg='blue')

    def predict_digit(self):
        try:
            processed_img = self.preprocess_image()
            prediction, confidence = self.model.predict_single(processed_img)

            self.result_label.config(text=f"Распознано: {prediction}", fg='green')
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

    def preprocess_image(self):
        img_processed = self.image.copy().filter(ImageFilter.GaussianBlur(1))
        bbox = img_processed.getbbox()

        if not bbox:
            return np.zeros(64, dtype=np.float32)

        cropped = img_processed.crop(bbox)
        width, height = cropped.size
        max_dim = max(width, height)

        squared = Image.new('L', (max_dim + 40, max_dim + 40), 255)
        squared.paste(cropped, (20, 20))

        img_small = squared.resize((8, 8), Image.LANCZOS)
        img_array = np.array(ImageOps.invert(img_small), dtype=np.float32)

        if img_array.max() > 0:
            img_array = (img_array / 255.0) * 16
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0.8)
            img_array = (img_array / img_array.max()) * 16

        return img_array.flatten()

    def run(self):
        self.window.mainloop()


def train_model():
    print("Загрузка и улучшение данных...")
    digits = load_digits()
    X, y = digits.data, digits.target

    X_augmented, y_augmented = [], []
    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])

        img = X[i].reshape(8, 8)
        if np.random.random() > 0.7:
            shifted = np.roll(img, -1, axis=1)
            shifted[:, -1] = 0
            X_augmented.append(shifted.flatten())
            y_augmented.append(y[i])
        if np.random.random() > 0.7:
            shifted = np.roll(img, 1, axis=1)
            shifted[:, 0] = 0
            X_augmented.append(shifted.flatten())
            y_augmented.append(y[i])

    X, y = np.array(X_augmented), np.array(y_augmented)

    print("Обучение модели...")
    model = DigitRecognizer(learning_rate=0.008)
    history = model.fit(X, y, epochs=150, verbose=True)

    accuracy = model.score(X, y)
    print(f"Точность модели: {accuracy:.2f}%")

    return model

def main():
    model = train_model()
    print("\nЗапуск приложения для рисования...")
    app = DigitDrawingApp(model)
    app.run()

if __name__ == "__main__":
    main()
