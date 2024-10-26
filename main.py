import face_recognition
import cv2
import os


known_face_encodings = []
known_face_names = []


def load_known_faces(directory):
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)


                    if encodings:
                        for encoding in encodings:
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")


load_known_faces("img_dataset")


face_locations = []
face_encodings = []
face_names = []


video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()


    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unauthorized"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)


    for (top, right, bottom, left), name in zip(face_locations, face_names):

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
video_capture.release()
cv2.destroyAllWindows()
