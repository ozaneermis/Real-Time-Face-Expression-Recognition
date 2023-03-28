# Real-Time-Face-Expression-Recognition
A pytorch application using ResNet18 for facial expression recognition (FER2013), 67.98% in FER2013.

## Dependies ##
- OpenCV -> 4.7.0
- Pytorch -> 2.0.0
- Pandas -> 1.5.3
- NumPy -> 1.24.2
- Pillow -> 9.4.0


## 68_acc.pth ##
- It's a ResNet18 model trained with FER2013 dataset

## Visualize for a test image by a pre-trained model ##
- Firstly download as a zip and extract it.

- Open 1.0_version.py

- model.load_state_dict(torch.load('./68_acc.pth', map_location=torch.device('cpu'))) If you have GPU you can delete map_location=torch.device('cpu') part.

- image = Image.open('./archive/My data/happy.png') give path to your image.


## Visualize for a real time by a pre-trained model ##
- Firstly download as a zip and extract it.

- Open 1.1_version.py

- Check the paths for
- face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
- model.load_state_dict(torch.load('./68_acc.pth', map_location=torch.device('cpu'))) ( As I mentioned before if you have GPU you delete map_location=torch.device('cpu') part.
- Run it
