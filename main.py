import os
import numpy as np
import cv2
from ultralytics import YOLO
import imutils


model_n = YOLO('yolov8n.pt')
model_x = YOLO('yolov8x.pt')
model_x_seg = YOLO('yolov8x-seg.pt')

video_path = r'videos/HD720_SN23341651_15-34-38_br9338_Left.avi' # arkasına fazla ışık vuran
video_path = r'videos/HD720_SN23341651_15-51-57_dgv731_Left.avi' # kırımızı arkası olan (boss)
video_path = r'videos/HD720_SN23341651_15-03-56_1116cm_Left.avi' # en başarılı su tankı beyaz olan 
cap = cv2.VideoCapture(video_path)

VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
OBJECT_CLOSER_RATIO_HEIGHT = 0.45 # How much of the entire frame should be covered by the proximity of the truck
OBJECT_CLOSER_RATIO_WIDTH = 0.80 
POSITIVE_RESULTS_FOR_VERIFICATION = 5 # Number of confirmations to avoid chance occurrence for frame capture
YOLO_DETECTION_CONFIDENT = 0.60
TRACK_START_FRAME_RATE = 0.60 * VIDEO_WIDTH # Variable indicating at which position of the truck we'll start tracking
TRACKING_FRAME_SKIP = 1 # Number of frames to be skipped in each loop in tracking session
WHEEL_POSITION_RANGE_START = int(0.65 * VIDEO_HEIGHT)
WHEEL_POSITION_RANGE_END = int(0.90 * VIDEO_HEIGHT)
VALIDATION_TREMOR_THRESHOLD = 20  # Value for what's considered as a tremor
IRREGULAR_THRESHOLD = 40 # While tracking comfirmed circles, to detect circles that are too far away
ACCEPT_THAT_THE_WHEEL_MOVES_PIXEL = 15 # In how many pixels will we accept that the wheel moves?
PHOTO_CAPTURED_IN_PIXEL_CHANGE = 75 # After a change of how many pixels, the photo was captured again !!! Should be higher than ACCEPT_THAT_THE_WHEEL_MOVES_PIXEL
VALID_DURATION = 10  # Define the duration for a valid circle (in frames)
WHEEL_OUT_OF_FRAME_RATE = 0.95 * VIDEO_WIDTH # The threshold ratio for tracked wheels to be considered out of frame
TRUCK_OUT_OF_FRAME_VERIFICATION_THRESHOLD = 100 # frame
TRUCK_LEFT_THRESHOLD = 100 # How many frames to wait if no confirmed wheel is found

# HoughCircles Adjusts
HOUGHCIRCLES_DP = 1
HOUGHCIRCLES_MINDIST = 200
HOUGHCIRCLES_PARAM1 = 100
HOUGHCIRCLES_PARAM2 = 30
HOUGHCIRCLES_MINRADIUS = 20
HOUGHCIRCLES_MAXRADIUS = 80


frame_skip = 6 # Number of frames to be skipped in each loop (for high-performance operation during non-truck times)
frame_count = 0

captured_frames = [] # Images of the truck we'll be processing last
truck_picture_counter = 0 # Variable indicating the stage of photo collection
verification_counter = 0 
model = model_n # Assignment to change the YOLOv8 model according to the situation
prev_circle_tracker = {} # Information of circles in the previous frame for comparison
valid_circles = [] # Circles that have been confirmed and put under tracking
val_lenght = 0 # Variable for circle validation
validation_counter = 0 # Variable for circle validation
tracking_session = False
truck_left_counter = 0
irregular_distance_counter = 0


while cap.isOpened():
    success, frame = cap.read()
    frame_count = frame_count + 1

    if success:
        if frame_count % frame_skip == 0:

            # Check if there are no trucks nearby with YOLO
            if not tracking_session:
                results = model(
                    frame,
                    conf= YOLO_DETECTION_CONFIDENT, 
                    iou= 0.75,
                    show= True,
                    stream_buffer= False,  # buffer all streaming frames (True) or return the most recent frame (False)
                    classes=[7]  # truck in YOLO coco.names
                    )
                
                for r in results:
                    for box in r.boxes:
                        if box.cls.item() == 7.0: # if a box represent truck

                            # position data of the truck square
                            truck_x1 = box.xyxy.tolist()[0][0]
                            truck_y1 = box.xyxy.tolist()[0][1]
                            truck_x2 = box.xyxy.tolist()[0][2]
                            truck_y2 = box.xyxy.tolist()[0][3]
                            truck_h = truck_y2 - truck_y1
                            truck_w = truck_x2 - truck_x1


                            if truck_h > VIDEO_HEIGHT * OBJECT_CLOSER_RATIO_HEIGHT: # eğer kamyon yeterince yakındaysa
                                
                                if truck_x2 > TRACK_START_FRAME_RATE and truck_picture_counter == 0:
                                    verification_counter += 1
                                    if verification_counter == POSITIVE_RESULTS_FOR_VERIFICATION: 
                                        verification_counter = 0
                                        truck_picture_counter += 1
                                        captured_frames.append(frame)
                                        frame_skip = TRACKING_FRAME_SKIP # 1
                                        tracking_session = True


            # WHEEL TRACKER
            if tracking_session:
                # Get circle datas in frame

                # Preprocess the frame for better edge detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # blurred = cv2.GaussianBlur(gray, (9, 9), 2, 2);


                # Apply Hough Circle Transform for wheel detection
                # and take circle datas in frame
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=HOUGHCIRCLES_DP, minDist=HOUGHCIRCLES_MINDIST, param1=HOUGHCIRCLES_PARAM1, param2=HOUGHCIRCLES_PARAM2, minRadius=HOUGHCIRCLES_MINRADIUS, maxRadius=HOUGHCIRCLES_MAXRADIUS)

                # Process detected circles or wheels
                circle_tracker = []
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        if y > WHEEL_POSITION_RANGE_START and y < WHEEL_POSITION_RANGE_END:
                            circle_tracker.append((x, y, r))

                if not prev_circle_tracker: 
                    prev_circle_tracker = circle_tracker


                # Detecting confirmed wheels to be tracked
                if not valid_circles:
                    temp_valid_circles = []

                    for current_circle in circle_tracker:
                        valid_circle_counter = 0
                        valid_counter = 0
                        min_displacement = VIDEO_WIDTH
                        
                        # Get the current position of the circle in the current frame

                        for prev_circle in prev_circle_tracker:
                            # Calculate the displacement between frames
                            displacement = np.sqrt((current_circle[0] - prev_circle[0])**2 + (current_circle[1] - prev_circle[1])**2)

                            # Get the closest circle
                            if displacement < min_displacement:
                                min_displacement = displacement

                        displacement = min_displacement

                        # Check for tremors
                        if displacement < VALIDATION_TREMOR_THRESHOLD:
                            temp_valid_circles.append(current_circle)


                    if val_lenght == 0: val_lenght = len(temp_valid_circles)
                    if val_lenght == len(temp_valid_circles):
                        validation_counter += 1
                    else:
                        val_lenght = len(temp_valid_circles)
                        validation_counter = 0


                    # If the circle has maintained within the threshold for a certain duration, consider it as a valid circle
                    if validation_counter >= VALID_DURATION:
                        valid_circles = temp_valid_circles


                # Find same circles and detect displacement
                if valid_circles:
                    temp_circle_tracker = []
                    for current_circle in circle_tracker:
                        min_displacement = VIDEO_WIDTH
                        for v_circles in valid_circles:
                            displacement = np.sqrt((current_circle[0] - v_circles[0])**2 + (current_circle[1] - v_circles[1])**2)

                            if displacement < min_displacement:
                                min_displacement = displacement
                        
                        if min_displacement < IRREGULAR_THRESHOLD:
                            temp_circle_tracker.append(current_circle)
                            irregular_distance_counter = 0
                        else:
                            irregular_distance_counter += 1
                        

                    circle_tracker = temp_circle_tracker
                    valid_circles = sorted(valid_circles, key=lambda elem: elem[0])
                    circle_tracker = sorted(circle_tracker, key=lambda elem: elem[0])

                    # Swapping positions by averaging over all wheels
                    truck_displacement = 0
                    if len(circle_tracker) == len(valid_circles):
                        for wheel_id in range(len(valid_circles)):
                            truck_displacement += circle_tracker[wheel_id][0] - valid_circles[wheel_id][0]
                        avarage_truck_displacement = truck_displacement / len(valid_circles)
                        
                        if avarage_truck_displacement > ACCEPT_THAT_THE_WHEEL_MOVES_PIXEL:
                            valid_circles = circle_tracker
                            truck_picture_counter += 1
                            if truck_picture_counter % int(PHOTO_CAPTURED_IN_PIXEL_CHANGE / ACCEPT_THAT_THE_WHEEL_MOVES_PIXEL) == 0:
                                captured_frames.append(frame)

                    # If the wheel is missed or on the wrong circle
                    if irregular_distance_counter >= 50:
                        valid_circles = []

                    # If wheel goes out of frame
                    temp_len_val_circles = len(valid_circles)
                    valid_circles = [circle for circle in valid_circles if circle[0] <= WHEEL_OUT_OF_FRAME_RATE]
                    if temp_len_val_circles != len(valid_circles):
                        # Start tracking if there's a newly entered wheel in the frame
                        valid_circles = []

                    # If the truck is now out of the frame, there won't be any wheels found in 100 frames
                    if not valid_circles:
                        truck_left_counter += 1
                        if truck_left_counter >= TRUCK_LEFT_THRESHOLD:
                            truck_left_counter = 0
                            model = model_x_seg

                            # PANORAMA
                            imgs = []
                            print('len(captured_frames): ', len(captured_frames))

                            for i in range(len(captured_frames)):
                                imgs.append(captured_frames[i]) 
                                imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.6,fy=0.6) 
                                # this is optional if your input images isn't too large 
                                # you don't need to scale down the image 
                                # in my case the input images are of dimensions 3000x1200 
                                # and due to this the resultant image won't fit the screen 
                                # scaling down the images  

                            imgs_reverse = imgs[::-1] # Reversing the order as we collected front images of the truck first followed by the back images


                            # STITCHING
                            stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                            (status, panaroma) = stitcher.stitch(imgs_reverse)

                            if status != cv2.STITCHER_OK: 
                            # checking if the stitching procedure is successful 
                            # .stitch() function returns a true value if stitching is  
                            # done successfully 
                                print("Stitching ain't successful") 
                            else:  
                                print('Your Panorama is ready!!!') 
                                # final output 
                                cv2.imshow('final result',panaroma) 
                                cv2.imwrite('final_result.jpg',panaroma)
                                cv2.waitKey(1)


                            # MASKING
                            results = model(panaroma,
                                                conf= 0.40,
                                                iou= 0.75,
                                                show= False,
                                                show_labels= True,
                                                show_conf= True,
                                                stream_buffer= False,  # buffer all streaming frames (True) or return the most recent frame (False)
                                                boxes= True,
                                                save_crop= False,
                                                visualize= False,
                                                retina_masks= True,
                                                classes=[7])  

                            for r in results:
                                mask_data = r.masks.data.tolist()[0]


                            mask = np.zeros_like(panaroma)
                            mask_array = np.array(mask_data)
                            print(panaroma.shape, mask_array.shape)

                            mask_array_uint8 = mask_array.astype(np.uint8)
                            masked_image = panaroma * mask_array_uint8[:, :, np.newaxis]

                            cv2.imwrite('masked_image.jpg', masked_image)
                            cv2.imshow('masked_image.jpg', masked_image)
                            cv2.waitKey(1)  # This will display the frame for 1 millisecond before moving to the next frame
                            tracking_session = False


                prev_circle_tracker = circle_tracker




                for (xx, yy, zz) in valid_circles:
                    cv2.circle(frame, (xx, yy), ACCEPT_THAT_THE_WHEEL_MOVES_PIXEL, (0, 0, 255), 3)  # Draw the valid circle perimeter
                    cv2.rectangle(frame, (xx - 2, yy - 2), (xx + 2, yy + 2), (0, 0, 255), -1)  # Draw the valid circle center
                cv2.imshow('wheel tracker', frame)
                cv2.waitKey(1)  # This will display the frame for 1 millisecond before moving to the next frame
                


            if 0xFF == ord('q'):
                break
    else:
        break



