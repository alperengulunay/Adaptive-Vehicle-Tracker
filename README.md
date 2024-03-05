# Adaptive-Narrow-Angle-Truck-Tracker

We have developed custom software to optimize the abstraction process of visual data for vehicles at oblique angles.

In this project, the YOLOv8 model based on the YOLO (You Only Look Once) architecture from the ultralytics library is used as the fundamental building block for object detection.

Within the scope of the project, the Hough Transformation method has been successfully applied for the detection and tracking of truck wheels, utilizing OpenCV and imutils libraries. This method is an algorithm used to detect circular objects in an image. With this algorithm, precise adjustment of truck wheels and displacement calculation has been achieved.

To comply with low system requirements, the weights of the YOLOv8 model have been dynamically adjusted with special tuning in our project. These customized weights are optimized to allow the project to operate more efficiently on low-end systems.

![tt](https://github.com/alperengulunay/Adaptive-Vehicle-Tracker/assets/68849018/ed62e1a3-969c-4869-9ee2-3ad313e45eb1)
