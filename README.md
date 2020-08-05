# Image-Steganography-in-CUDA  
This program is to do image Steganography in CUDA with a blindhid algorithm.  
GPU_encoding_result.png is for GPU encoding for text1.txt and landscape image.  
GPU_decoding_result.png is for GPU decoding for text1.txt and landscape image.  
### requirement environment###   
1. Install opencv (my opencv version is 2.4.9.1)  
you can check the open cv version by   
$ pkg-config --modversion opencv  
install openCV on the Mac:  
https://medium.com/@jaskaranvirdi/setting-up-opencv-and-c-development-environment-in-xcode-b6027728003  

2. Install cuda (my cuda version is V8.0.61)  
you can check the cuda version by   
$ nvcc --version  

### compilation command ###  

Encode message to the image:  
nvcc encode_final.cu `pkg-config --cflags opencv` -o encode_final.out `pkg-config --libs opencv`  

Decode message from the image:  
nvcc decode_final.cu `pkg-config --cflags opencv` -o decode_final.out `pkg-config --libs opencv`  


### execution command ###  
   
  Encode message to the image:  
  $ ./encode_final.out input_image_path message_file_path output_image_path  

  $ ./encode_final.out Images/dennis_ritchie.png TextFiles/text1.txt Images/output_image_profile.png  
  $ ./encode_final.out Images/dennis_ritchie.png TextFiles/text2.txt Images/output_image_profile.png  
  $ ./encode_final.out Images/landscape.jpg TextFiles/text1.txt Images/output_image_landscape.png   
  $ ./encode_final.out Images/landscape.jpg TextFiles/text2.txt Images/output_image_landscape.png  
   
  Decode message from the image:  
  $ ./decode_final.out Images/output_image_profile.png  
  $ ./decode_final.out Images/output_image_landscape.png  
 

### expected output ###  
---GPU version---  
 When you encode the message, the output are an image and Encode Kernel execution time.  
 When you decode the message, the output are decode Kernel execution time and the hidden message that displayed on your console.  

