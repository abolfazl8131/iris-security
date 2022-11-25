import cv2
import numpy 
import matplotlib.pyplot as plt
import mediapipe




class IrisDetection():

    cap = cv2.VideoCapture(0)

    _face_mesh = mediapipe.solutions.face_mesh

    LEFT_EYE = [362, 382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

    LEFT_IRIS = [474 , 475,476,477]

    RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

    RIGHT_IRIS = [469,470,471,472]

    array_of_iris = []

    def video_capture(self) -> None:

        i = 0
        while True:
            ret , frame = self.cap.read()
            frame = cv2.flip(frame,1)
            rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            img_h , img_w = frame.shape[:2]
            results = self.mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                #print(results.multi_face_landmarks[0].landmark)
                mesh_points = numpy.array([numpy.multiply([p.x , p.y] , [img_w , img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                #print(mesh_points)
                # cv2.polylines(frame , [mesh_points[self.LEFT_IRIS]] , True , (0,255,0) , 1 , cv2.LINE_AA)
                # cv2.polylines(frame , [mesh_points[self.RIGHT_IRIS]] , True , (0,255,0) , 1 , cv2.LINE_AA)

                (l_cx , l_cy) , l_radius = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
                (r_cx , r_cy) , r_radius = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])

                center_left = numpy.array([l_cx,l_cy] , dtype=numpy.int32)
                center_right = numpy.array([r_cx,r_cy] , dtype=numpy.int32)

                self.array_of_iris.append('hhhhhhhhhhh')
                cv2.circle(frame , center_right , int(r_radius) , (255,0,255) , 1 , cv2.LINE_AA)

                cv2.circle(frame , center_left , int(l_radius) , (255,0,255) , 1 , cv2.LINE_AA)
            i+=1

            cv2.imshow('img' , frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    

    def face_mesh(self):
        with self._face_mesh.FaceMesh(max_num_faces = 1 , 
            refine_landmarks=True,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5) as self.mesh:
                self.video_capture()

    def return_iris(self):
        return self.array_of_iris

    def run(self):
        self.face_mesh()
        
        

