/*
This program referred to the hitanshu-dhawan on the
https://github.com/hitanshu-dhawan/ImageSteganography
and Ghazanfar Abbas on the
http://programmerfish.com/how-to-write-a-custom-cuda-kernel-with-opencv-as-host-library/
*/

#include <bits/stdc++.h>
#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
//#include <cv.h>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// this function BinaryStringToText referred to the Mirolimjon Majidov on the
// https://stackoverflow.com/questions/23344257/convert-a-string-of-binary-into-an-ascii-string-c/23344876
string BinaryStringToText(string binaryString) {
  string text = "";
  stringstream sstream(binaryString);
  while (sstream.good()) {
    bitset<8> bits;
    sstream >> bits;
    text += char(bits.to_ulong());
  }
  return text;
}

__global__ void LSB(unsigned char *input, char *message, int image_size) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = xIndex + yIndex * blockDim.x * gridDim.x;

  if (offset >= image_size) {
    return;
  }

  message[offset] = 0;
  //   if RGB value's last bit is 1, then we put 1 to the message, otherwise,
  //   use default 0
  if (input[offset] & 1) {
    message[offset] |= 1;
  }
}

int main(int argc, char **argv) {
  /*
  ./decode output_image.png
  argv[0] = ./decode
  argv[1] = output_image.png
  */

  // Checks if proper number of arguments are passed
  if (argc != 2) {
    cout << "Arguments Error"
         << "\n";
    exit(-1);
  }

  // Stores original image
  Mat image = imread(argv[1]);
  if (image.empty()) {
    cout << "Image Error\n";
    exit(-1);
  }
  // imageByte = image_size * 3 ; RGB
  const int imageByte = image.step * image.rows;
  int image_size = image.cols * image.rows;
  // cout << "image byte =" << imageByte << "\n";
  // cout << "image size =" << image_size << "\n";
  unsigned char *d_input;
  char *message_d, *message_h;

  message_h = (char *)malloc(imageByte * sizeof(char));
  cudaMalloc((void **)&message_d, imageByte * sizeof(char));
  cudaMalloc<unsigned char>(&d_input, imageByte);

  cudaMemcpy(d_input, image.ptr(), imageByte, cudaMemcpyHostToDevice);

  const dim3 block(16, 16);
  const dim3 grid((image.cols + block.x - 1) / block.x,
                  (image.rows + block.y - 1) / block.y);

  // capture the start time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  LSB<<<grid, block>>>(d_input, message_d, image_size);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Decode Kernel execution time is:  %3.10f sec\n", elapsedTime / 1000);

  cudaMemcpy(message_h, message_d, imageByte * sizeof(char),
             cudaMemcpyDeviceToHost);
  // every 8 bits convert to one char
  int i = 0, j = 0;
  while (i < imageByte - 8) {
    string oneChar = "";
    // add 8 bit to onechar, However, 0 cannot add to string, so convert to int
    // first.
    for (j = 0; j < 8; j++) {
      int index = i + j;
      int num = (int)message_h[index];
      char temp[1];
      sprintf(temp, "%d", num);
      string s(temp);
      oneChar += s;
    }

    if (oneChar == "00000000") {
      break;
    }

    String ch = BinaryStringToText(oneChar);
    cout << ch;
    i += 8;
  }
  cout << "\n";
  return 0;
}
