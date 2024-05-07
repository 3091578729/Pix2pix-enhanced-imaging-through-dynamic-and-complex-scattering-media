# Pix2pix-enhanced imaging through dynamic and complex scattering media: Seeing through smoke

Python implementation of paper:Pix2pix-enhanced imaging through dynamic and complex scattering media: Seeing through smoke. 

We provide model, pre-trained weights(download link available below), test data and a quick demo.

1.The prepared dataset was uploaded to Google Cloud Drive.
  https://drive.google.com/file/d/1LDup4JZ_vLMhmTCuaa8bv7Byur9hZ09H/view?usp=drive_link
  
2.Upload the trained weights to Google Cloud
  https://drive.google.com/file/d/1jK-42izQt5IxNO1Yk6ogYW8HyKD0W6rM/view?usp=drive_link

 # Abstract
 Scattering media, as optical obstacles, exert a significant impact on image formation and information transmission. Existing imaging methods through scattering media are mostly targeted towards static or quasi-dynamic scattering media, yet in practical applications, the challenges posed by dynamic scattering media are more pronounced. To address this, we developed a novel framework using the pix2pix generative adversarial network in a specific dynamic scattering medium — smoke. Utilizing deep learning, our model overcomes the difficulties of decorrelation caused by the continuous motion of dynamic scattering media, achieving accurate  prediction of speckle patterns. We design five groups of smoke with different particle concentrations as dynamic scattering media and collect corresponding datasets of handwritten digit speckle images for network training. Based on the testing results, we have validated the accuracy of the model in target prediction, providing valuable insights for further research in the field of dynamic scattering medium imaging.

# Pix2pix network structure
![image](https://github.com/3091578729/Dynamic-scattering-medium-imaging-based-on-pix2pix/assets/75689416/513fefd9-3c4b-4036-83f0-983c24284f33)


# Requirements
torch==1.12.0+cu113
torchvision==0.13.0+cu113
Pillow==10.1.0
cv2==4.4.0
numpy==1.26.4


  
