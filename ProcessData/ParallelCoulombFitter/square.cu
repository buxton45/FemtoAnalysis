///////////////////////////////////////////////////////////////////////////
// Square:                                                               //
//////////////////////////////////////////////////////////////////////////

#include "square.h"

//Illegal for CUDA __global__ function (i.e. kernel) to be defined as class member function
//You can call a kernel in a struct or class member function, but kernal cannot be member function itself
__global__ void square(float * d_out, float * d_in)
{
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx]  = f*f;
}


Square::Square()
{

}

Square::~Square()
{

}

void Square::RunSquare(float * h_out, float * h_in, const int ARRAY_SIZE)
{
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  //declare GPU memory pointers
  float * d_in;
  float * d_out;

  //allocate GPU memory
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, ARRAY_BYTES);

  //transfer the array to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  //launch the kernel
  GpuTimer timer;
  timer.Start();

  square<<<1, ARRAY_SIZE>>>(d_out, d_in);

  timer.Stop();
  std::cout << "Done: " << timer.Elapsed() << "ms" << std::endl;

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //copy back the result array to the CPU
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  //print out the resulting array
  for(int i=0; i<ARRAY_SIZE; i++)
  {
    printf("%f", h_out[i]);
    printf(((i%4) != 3) ? "\t" : "\n");
  }

  //free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_out);
}
