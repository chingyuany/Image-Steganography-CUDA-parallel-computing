/*
This program referred to the hitanshu-dhawan on the
https://github.com/hitanshu-dhawan/ImageSteganography
and Ghazanfar Abbas on the
http://programmerfish.com/how-to-write-a-custom-cuda-kernel-with-opencv-as-host-library/
*/

#include <fstream>
#include <highgui.h>
#include <iostream>
#include <sstream> //std::stringstream
#include <string>
//#include <cv.h>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

__global__ void LSB(unsigned char *input, char *message, int message_size) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = xIndex + yIndex * blockDim.x * gridDim.x;

  // 8 threads process one char of 8 bits
  int charno = offset / 8;
  if (charno >= message_size) {
    return;
  }
  //  process start from the first bit on the left
  int bit_count = 7 - (offset % 8);
  char ch = message[charno] >> bit_count;
  // if this bit is 1, then put 1 to the image RGB value, if bit == 0, put 0
  if (ch & 1) {
    input[offset] |= 1;
  } else {
    input[offset] &= ~1;
  }
}

int main(int argc, char **argv) {

  /*
  ./encode image.png textfile.txt output_image.png
  argv[0] = ./encode
  argv[1] = image.png
  argv[2] = textfile.txt
  argv[3] = output_image.png
  */

  // Checks if proper number of arguments are passed
  if (argc != 4) {
    cout << "Number of Arguments Error"
         << "\n";
    exit(-1);
  }

  // Stores original image
  Mat image = imread(argv[1]);
  if (image.empty()) {
    cout << "Load Image Error\n";
    exit(-1);
  }

  // print original pixel rgb value
  // Vec3b pixel = image.at<Vec3b>(Point(0, 0));
  // printf("\n0 =%d 1= %d 2 =%d\n", pixel.val[0], pixel.val[1], pixel.val[2]);
  // pixel = image.at<Vec3b>(Point(1, 0));
  // printf("\n3 =%d 4= %d 5 =%d\n", pixel.val[0], pixel.val[1], pixel.val[2]);
  // pixel = image.at<Vec3b>(Point(2, 0));
  // printf("\n6 =%d 7= %d \n", pixel.val[0], pixel.val[1]);

  // Open file for text information
  ifstream file;
  file.open(argv[2]); // open the input file
  if (!file.is_open()) {
    cout << "File Error\n";
    exit(-1);
  }

  stringstream strStream;
  strStream << file.rdbuf();    // read the file
  string str = strStream.str(); // str holds the content of the file
  // +1 is space for end of string '\0'
  char arr[str.length() + 1];
  // below include null characters and newline characters.
  cout << "load text file size is " << str.size() << "\n";
  strcpy(arr, str.c_str());
  // for (int i = 0; i < str.length(); i++)
  //   cout << arr[i];

  // check if text's bit of size larger than image  bit of RGB
  const int ImageSize = image.step * image.rows;
  int message_size = str.size() + 1;
  if ((message_size)*8 > ImageSize * 3) {
    printf("The input text file is too big, choose a larger image");
  }

  cv::Mat output(image.rows, image.cols, CV_8UC3);
  unsigned char *d_input;
  char *message;
  cudaMalloc<unsigned char>(&d_input, ImageSize);
  cudaMalloc((void **)&message, message_size * sizeof(char));

  cudaMemcpy(d_input, image.ptr(), ImageSize, cudaMemcpyHostToDevice);
  cudaMemcpy(message, arr, message_size * sizeof(char), cudaMemcpyHostToDevice);

  const dim3 block(16, 16);
  // Calculate grid size to cover the whole image
  const dim3 grid((image.cols + block.x - 1) / block.x,
                  (image.rows + block.y - 1) / block.y);

  // capture the start time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  LSB<<<grid, block>>>(d_input, message, message_size);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Encode Kernel execution time is:  %3.10f sec\n", elapsedTime / 1000);

  cudaMemcpy(output.ptr(), d_input, ImageSize, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(message);

  // Writes the stegnographic image
  imwrite(argv[3], output);

  // print output pixel rgb value
  // pixel = output.at<Vec3b>(Point(0, 0));
  // printf("\n0 =%d 1= %d 2 =%d\n", pixel.val[0], pixel.val[1],
  // pixel.val[2]); pixel = output.at<Vec3b>(Point(1, 0)); printf("\n3 =%d 4=
  // %d 5 =%d\n", pixel.val[0], pixel.val[1], pixel.val[2]); pixel =
  // output.at<Vec3b>(Point(2, 0)); printf("\n6 =%d 7= %d \n", pixel.val[0],
  // pixel.val[1]);
  return 0;
}
