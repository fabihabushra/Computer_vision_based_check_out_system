Computer Vision-Based Batch Billing of Supermarket Products
========
This study introduces a computer vision-based checkout system aimed at enhancing retail transaction efficiency by enabling batch processing of items at the point of sale. Traditional barcode-based checkout methods are limited by their sequential scanning process, leading to increased transaction times and reliance on cashier proficiency. To address these limitations, we propose a real-time detection system that utilizes a YOLO-based framework, integrated within a GUI checkout application, to facilitate the simultaneous billing of multiple products. Our methodology involves training the YOLO model with a dataset comprising both authentic and synthetically generated images of 25 local retail items and 5 weight-variable products across various categories. Since conventional packaging uses QR codes that cannot be detected optimally from long distances, our system incorporates Hybrid ArUco Markers as an innovative solution for improved detection accuracy of weight-dependent products. This research highlights the potential of a fully automated billing framework in retail environments, demonstrating a scalable solution through the integration of deep learning algorithms with deterministic pattern recognition.
#### Working Demo of Billing
<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/static.gif?raw=true">

#### Detection on a Conveyor Belt

<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/dynamic.gif?raw=true">

# Dataset:
We created a dataset with 26 detectable locally available products (excluding different categories for weight dependent products) in Bangladesh. The dataset was build by taking photos of the products and web scrapping review images from Daraz.  Our dataset consists of 3056 real images and a total of 5734 images including synthetic generations. They contain 13383 object instances.
```
https://drive.google.com/drive/folders/184pc4umU4QH08CDoZILHrqakbQw9Zug4?usp=sharing
```

![alt text](https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/Products.jpg?raw=true)
```
Products:
1. ACI Pure Salt 1kg 
2. Parachute Coconut Oil 50ml 
3. Energy Biscuit 
4. Dettol Original Soap 
5. Mr. Twist 
6. Teer Sugar 1kg 
7. Rupchanda Cooking Oil 1L 
8. Pran Hot Sauce 
9. Cocola Chicken Masala Noodles 
10. Frutika Grape 250ml 
11. Sepnil HandSanitizer 40ml 
12. Snickers 
13. Sprite 250ml Bottle 
14. Vaseline Original 
15. Coca-cola 250ml bottle 
16. Lux Soft Glow 
17. Dettol Soap Aloe Vera 
18. Neem Soap Pure Neem 
19. Neem Soap Olive and Aloe Vera 
20. Pears Soap with Lemon Flower Extracts
21. Pears Soap with Natural Oils
22. Maggi Soup Chicken 
23. Maggi Soup Vegetable 
24. Fresh Coriander Powder 
25. Fresh Turmeric Powder 
26. Miniket Rice (Weight Dependent) 
27. Local Beef (Weight Dependent)
28. Local Onion (Weight Dependent) 
29. Local Potato (Weight Dependent) 
30. Local Garlic (Weight Dependent)
```
#### Example of Synthetic images used for augmentation
<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/Synthetic image.png?raw=true">

# System Description
#### A Flowchart of the billing system
<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/flowchart.png?raw=true">

## Model Training

We trained both Yolov5m and Yolov5l models on our datasets. The models were initialized with pretrained weights up to 300 epochs on the coco dataset before training on our product dataset.
```
# Training Settings
Epoch  =  300 (Yolov5m), 100 (Yolov5l)
Batch Size = 32
Image Size = 416x416
Workers = 1
Optimizer = SGD
```
![alt text](https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/ArucoLnM.png?raw=true)


## Hardware setup
A table mounted webcam is used to detect the products on the checkout corner.
![alt text](https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/hardware%20setup.png?raw=true)


## Weight Dependent Product Detection
We used a hybrid of 2 classes of ArUco markers to detect the id and weight of products. These labels are presumed to be these labels are assumed to applied on the product by sales assitants before billing.

#### Hybrid ArUco Marker

<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/arucoh.png?raw=true" width="300">

#### Detection
![alt text](https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/arucodet.png?raw=true)


## GUI Implementation
A GUI was created using PyQt5 framework for ease of use of the cashier. Here is how to use it:
1. Press Start Billing
2. Put a batch of products under the camera
3. Press the lock button
4. Remove the products and bring in new products if any were left out
5. And repeat from step 2
6. After all the products are billed, press Stop Billing

<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/lock mechanism.png?raw=true">
The green area represents products that have already been billed, the non-green area represents products that are currently in view of the camera.


# Result And Analysis
#### Confusion Matrix for the Final Ensemble model
<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/ensemble_test3.png?raw=true">

We have used mAP scores for as the evaluation metric in all of our results.
#### Performance on different lighting conditions and front and back sides of the products
<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/shot_220926_173800.png?raw=true">

#### Performance improvement after augmentation with synthetic images
<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/aug.png?raw=true" width="400">

#### Comparison between weight-dependent product detection using QR code and ArUco marker
<img src="https://github.com/Rusab/Supermall-Checkout-system-yolov5/blob/qr-implementation/images/qraruco.png?raw=true">

*Stability = Correct Detections / Number of Frames Scanned**
