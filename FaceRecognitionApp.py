import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import mysql.connector
import numpy as np
import face_recognition
import logging

logging.basicConfig(level=logging.INFO)

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Reconhecimento Facial')

        self.capture_button = tk.Button(root, text="Capturar imagem", command=self.capture_image)
        self.capture_button.pack()

        self.train_button = tk.Button(root, text="Treinar modelo", command=self.train_model)
        self.train_button.pack()

        self.recognize_button = tk.Button(root, text="Reconhecimento", command=self.recognize_face)
        self.recognize_button.pack()

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.known_face_names = []

        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.show_frame)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            name = self.prompt_for_name()
            ra = self.prompt_for_ra()
            if name and ra:
                self.save_image_to_db(name, frame, ra)
                messagebox.showinfo("Success", "A imagem foi capturada!")

    def prompt_for_name(self):
        name = simpledialog.askstring("Input", "Digite seu nome:")
        return name
    
    def prompt_for_ra(self):
        while True:
            try:
                ra = simpledialog.askstring("Input", "Digite seu RA:")
                if ra.isdigit():
                    return int(ra)
                else:
                    messagebox.showerror("Erro", "O RA deve ser um número inteiro.")
            except ValueError:
                messagebox.showerror("Erro", "Entrada inválida. Tente novamente.")

    def save_image_to_db(self, name, frame, ra):
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = buffer.tobytes()
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="face_recognition_db"
        )
        cursor = conn.cursor()
        cursor.execute("INSERT INTO faces (nome, imagem, ra) VALUES (%s, %s, %s)", (name, image_data, ra))
        conn.commit()
        cursor.close()
        conn.close()

    def recognize_face(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            recognized = False  # Flag para indicar se um rosto foi reconhecido

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Desconhecido"

                if any(matches):
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        recognized = True  # Um rosto foi reconhecido

                        # Recuperar RA do banco de dados e inserir na tabela de presença
                        self.mark_attendance(name)

                color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            if recognized:
                logging.info("Rosto reconhecido.")
            else:
                logging.info("Nenhum rosto reconhecido.")

            cv2.imshow("Recognize Face", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def mark_attendance(self, name):
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="face_recognition_db"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT ra FROM faces WHERE nome = %s", (name,))
        ra = cursor.fetchone()
        if ra:
            cursor.execute("INSERT INTO presenca (nome, ra) VALUES (%s, %s)", (name, ra[0]))
            conn.commit()
        cursor.close()
        conn.close()

    def load_faces_from_db(self):
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="face_recognition_db"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT nome, imagem FROM faces")
        rows = cursor.fetchall()
        for name, image_data in rows:
            np_image = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            rgb_img = np.ascontiguousarray(img[:, :, ::-1])
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
        cursor.close()
        conn.close()

    def train_model(self):
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.load_faces_from_db()
        messagebox.showinfo("Success", "O modelo foi treinado com sucesso!")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
