# pip install opencv-python
from deepface import DeepFace
# pip install deepface
# pip install tf-keras

from Database import db, collection

# facenet모델 사용
model = DeepFace.build_model("Facenet512")

# face-recognition standalone script
import cv2
import queue
import threading

class FaceRecognizer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        # self.target_img_path = target_img_path

    def recognize_faces(self):
        while not self.stop_event.is_set():
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                # Perform face recognition on the frame
                target_embedding = self.get_target_embedding(target_img_path=frame)
                if target_embedding:
                    results = self.search_similar_images(target_embedding)
                    for result in results:
                        final_result = self.verify(target_embedding, result["embedding"])
                        print(result["img_path"], final_result)
                        if final_result:
                            print("Face recognized: ", result["img_path"])
                            self.stop_event.set()
                            break
                        else:
                            print("Face not recognized: ", result["img_path"])
                            break

    def start_recognition(self):
        recognize_thread = threading.Thread(target=self.recognize_faces)
        recognize_thread.start()

        while True:
            ret, frame = self.cap.read()
            cv2.imshow("Webcam", frame)

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            if recognize_thread.is_alive():
                if self.stop_event.is_set():
                    break
            else:   # if thread is not alive
                break

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.stop_event.set()
        self.cap.release()
        cv2.destroyAllWindows()

    def get_target_embedding(self, target_img_path):
        try:
            facial_representations = DeepFace.represent(
                img_path=target_img_path, 
                model_name="Facenet512", 
                enforce_detection=True,
                detector_backend="fastmtcnn"
            )
            
            if len(facial_representations) == 1:
                target_embedding = facial_representations[0]["embedding"]
            else:
                print("Multiple faces detected in the target image.")
                print("Selecting the biggest face.")
                biggest_face = facial_representations[0]
                for face in facial_representations:
                    face_area = face["facial_area"]['w'] * face["facial_area"]['h']
                    if face_area > (biggest_face["facial_area"]['w'] * biggest_face["facial_area"]['h']):
                        biggest_face = face
                target_embedding = biggest_face["embedding"]
                
            return target_embedding
        except:
            print("No face detected in the target image.")
            return False

    def search_similar_images(self, target_embedding):

        results = db.deepface.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": target_embedding,
                    "numCandidates": 50,
                    "limit": 3
                }
            }
        ])

        return results

    def verify(self, target, candidate):
        result = DeepFace.verify(
            img1_path=target,
            img2_path=candidate,
            model_name="Facenet512",
            detector_backend="fastmtcnn",
            distance_metric="cosine",
            enforce_detection=False,
            align=True,
            expand_percentage=0,
            normalization="base",
            silent=False,
        )
        return result['verified']

    def insert_embedding(self, target_img_path):
        target_embedding = self.get_target_embedding(target_img_path)
        db.deepface.insert_one({"img_path": target_img_path, "embedding": target_embedding})
        
# Usage example:
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    
    # Start the face recognition process
    recognizer.start_recognition()
