import tkinter as tk
from tkinter import messagebox
from customtkinter import CTk, CTkFrame, CTkCanvas, CTkLabel, CTkEntry, CTkButton
from PIL import Image, ImageTk
import cv2
import numpy as np
from centroidtracker import CentroidTracker
import winsound

class CrowdDensityMonitor:
    def __init__(self, window, video_source='test_video1.mp4'):
        self.window = window
        self.window.title("Crowd Density Monitoring System")
        self.video_source = video_source
        self.users = {'Bijil': 'bijil123', 'Aakash': 'akj','Navtej':'kunji234','Gajju' : 'Aanaumfi'}  # Predefined users and passwords

        self.username = None
        self.password = None

        self.threshold = None
        self.video = None
        self.tracker = None
        self.detector = None
        self.current_frame = None
        self.cap = None
        self.frame_count = 0
        self.start_time = 0
        self.total_correct_detections = 0
        self.total_frames_processed = 0
        self.objects = {}  # Initialize objects dictionary to store tracked objects
        self.detected_object_ids = set()  # Initialize set to store unique detected object IDs

        # Paths to prototxt and caffeModel files
        self.protopath = "MobileNetSSD_deploy.prototxt"
        self.modelpath = "MobileNetSSD_deploy.caffemodel"

        self.create_login_page()

    def create_login_page(self):
        self.login_frame = CTkFrame(self.window)
        self.login_frame.pack(fill='both', expand=True)
        
        self.username_label = CTkLabel(self.login_frame, text="Username:")
        self.username_entry = CTkEntry(self.login_frame)
        self.password_label = CTkLabel(self.login_frame, text="Password:")
        self.password_entry = CTkEntry(self.login_frame, show="*")
        self.login_button = CTkButton(self.login_frame, text="Login", command=self.authenticate)

        self.username_label.pack(pady=10)
        self.username_entry.pack(pady=10)
        self.password_label.pack(pady=10)
        self.password_entry.pack(pady=10)
        self.login_button.pack(pady=10)

        # Center the login frame
        self.login_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Make the login frame full screen
        x = self.window.winfo_screenwidth()
        y = self.window.winfo_screenheight()
        self.window.geometry('{}x{}+0+0'.format(x, y))

    def authenticate(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if username in self.users and self.users[username] == password:
            self.username = username
            self.password = password
            self.login_frame.destroy()
            self.create_threshold_page()
        else:
            messagebox.showerror("Error", "Invalid username or password")

    def create_threshold_page(self):
        self.threshold_frame = CTkFrame(self.window)
        self.threshold_frame.pack(fill='both', expand=True)
        
        self.threshold_label = CTkLabel(self.threshold_frame, text="Enter Threshold:")
        self.threshold_entry = CTkEntry(self.threshold_frame)
        self.threshold_button = CTkButton(self.threshold_frame, text="Submit", command=self.start_monitoring)
        
        self.threshold_label.pack(pady=10)
        self.threshold_entry.pack(pady=10)
        self.threshold_button.pack(pady=10)

        # Center the threshold frame
        self.threshold_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Make the threshold frame full screen
        x = self.window.winfo_screenwidth()
        y = self.window.winfo_screenheight()
        self.window.geometry('{}x{}+0+0'.format(x, y))

    def create_output_page(self):
        # Destroy the previous frame if it exists
        if hasattr(self, 'threshold_frame'):
            self.threshold_frame.destroy()

        # Create the output frame
        self.output_frame = CTkFrame(self.window)
        self.output_frame.pack(fill='both', expand=True)

        # Create labels for displaying information
        self.label_fps = CTkLabel(self.output_frame, text="FPS: ", font=('Helvetica', 20,'bold'),text_color="gray26")
        self.label_lpc = CTkLabel(self.output_frame, text="LPC: ", font=('Helvetica' ,20,'bold' ),text_color="gray26")
        self.label_opc = CTkLabel(self.output_frame, text="OPC: ", font=('Helvetica',20,'bold'),text_color="gray26")
        # Position the labels
        self.label_fps.place(x=700,y=50)
        self.label_lpc.place(x=700,y=100)
        self.label_opc.place(x=700,y=150)

      
        # Create a canvas for displaying the video
        # Create a frame to hold the canvas
        frame = CTkFrame(self.output_frame)
        frame.pack(fill='both', expand=True)

        # Calculate the center coordinates of the screen
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        center_x = screen_width // 2
        center_y = screen_height // 2

        # Calculate the position of the canvas
        canvas_width = 600
        canvas_height = 400
        canvas_x = center_x - (canvas_width // 2)
        canvas_y = center_y - (canvas_height // 2)

        # Create the canvas and position it in the frame
        self.canvas = CTkCanvas(frame, width=canvas_width, height=canvas_height)
        self.canvas.pack(padx=10, pady=10)

        # Position the frame in the output frame
        frame.place(x=canvas_x, y=canvas_y)

        # Make the output frame full screen
        x = self.window.winfo_screenwidth()
        y = self.window.winfo_screenheight()
        self.window.geometry('{}x{}+0+0'.format(x, y))

        # Start video processing
        self.process_video()

    def start_monitoring(self):
        threshold_value = self.threshold_entry.get()
        if threshold_value.isdigit():
            self.threshold = int(threshold_value)
            self.threshold_frame.destroy()
            self.create_output_page()
        else:
            messagebox.showerror("Error", "Please enter a valid threshold value (integer)")

    def process_video(self):
        # Open the video source
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video source")
            return

        # Initialize the tracker and detector
        self.tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
        self.detector = cv2.dnn.readNetFromCaffe(prototxt=self.protopath, caffeModel=self.modelpath)

        # Start time for calculating FPS
        self.start_time = cv2.getTickCount()

        # Continuously update the canvas with video frames
        self.update_canvas()

    def update_canvas(self):
        # Read a frame from the video source
        ret, frame = self.cap.read()
        if ret:
            # Increment frame count
            self.frame_count += 1

            # Resize the frame
            frame = cv2.resize(frame, (600, 400))

            # Perform object detection
            rects = self.detect_objects(frame)

            # Apply non-maxima suppression
            rects = self.non_max_suppression_fast(rects, 0.3)

            # Update the tracker with detected objects
            self.objects = self.tracker.update(rects)
            lpc_count = len(self.objects)
            opc_count = len(self.detected_object_ids)  # OPC count based on total unique object IDs

            # Calculate FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.start_time)

            # Update labels with information
            self.label_fps.configure(text="FPS: {:.2f}".format(fps))
            self.label_lpc.configure(text="LPC: {}".format(lpc_count))
            self.label_opc.configure(text="OPC: {}".format(opc_count))

            # Reset start time
            self.start_time = cv2.getTickCount()

            # Check if crowd density exceeds threshold
            if lpc_count >= self.threshold:
                print("ALERT: Crowd density exceeds threshold!")
                winsound.Beep(1000, 240)

            # Evaluate detections (for basic accuracy calculation)
            self.evaluate_detections(rects)

            # Update detected object IDs set
            self.detected_object_ids.update(self.objects.keys())

            # Draw bounding boxes on the frame
            for (objectId, bbox) in self.objects.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "ID: {}".format(objectId), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a Tkinter PhotoImage
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

            # Update the canvas with the new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Schedule the next update after 10 milliseconds
        self.window.after(10, self.update_canvas)

    def detect_objects(self, frame):
        # Preprocess the frame for object detection
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        # Set the blob as input to the detector
        self.detector.setInput(blob)

        # Perform object detection
        person_detections = self.detector.forward()
        rects = []

        # Loop over the detections
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.1:
                idx = int(person_detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(person_box)

        return np.array(rects)

    def non_max_suppression_fast(self, boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    def evaluate_detections(self, detected_boxes):
        # Simulate ground truth data (assuming the same number of people as detected by the model)
        ground_truth_boxes = np.array([bbox for (_, bbox) in self.objects.items()])

        # Calculate Intersection over Union (IoU) for each detected box
        correct_detections = 0
        for det_box in detected_boxes:
            ious = self.calculate_iou(det_box, ground_truth_boxes)
            max_iou = np.max(ious)
            if max_iou >= 0.5:  # Consider a detection as correct if IoU is greater than or equal to 0.5
                correct_detections += 1

        # Increment total frames processed
        self.total_frames_processed += 1

        # Calculate and print accuracy
        accuracy = (correct_detections / len(detected_boxes)) * 100 if len(detected_boxes) > 0 else 0
        print("Accuracy: {:.2f}%".format(accuracy))

        # Update total correct detections
        self.total_correct_detections += correct_detections

    def calculate_iou(self, box1, box2):
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])

        intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

        iou = intersection / (box1_area + box2_area - intersection)

        return iou

if __name__ == "__main__":
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    root = CTk()
    app = CrowdDensityMonitor(root)
    root.mainloop()
