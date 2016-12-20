///////////////////////////////////////////////////////////////////////////
// InterpolateGPU:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "InterpolateGPU.h"


//________________________________________________________________________________________________________________
__device__ int GetBinNumber(double aBinSize, int aNbins, double aValue)
{
//TODO check the accuracy of this
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<aNbins; i++)
  {
    tBinKStarMin = i*aBinSize;
    tBinKStarMax = (i+1)*aBinSize;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
__device__ int GetBinNumber(int aNbins, double aMin, double aMax, double aValue)
{
//TODO check the accuracy of this
  double tBinSize = (aMax-aMin)/aNbins;
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<aNbins; i++)
  {
    tBinKStarMin = i*tBinSize + aMin;
    tBinKStarMax = (i+1)*tBinSize + aMin;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
__device__ int GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue)
{
//TODO check the accuracy of this
  int tNbins = (aMax-aMin)/aBinWidth;
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<tNbins; i++)
  {
    tBinKStarMin = i*aBinWidth + aMin;
    tBinKStarMax = (i+1)*aBinWidth + aMin;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}





//________________________________________________________________________________________________________________
__global__ void BilinearInterpolateVector(double* d_out, double* d_PairsIn, double* d_2dVecIn)
{
  //NOTE: THIS IS SLOWER THAN BilinearInterpolate, but a method like this may be necessary for parallelization

//  int tIdx = threadIdx.x;
  int tIdx = threadIdx.x + blockIdx.x*blockDim.x;

  double aX = d_PairsIn[2*tIdx];
  double aY = d_PairsIn[2*tIdx+1];

  double tF = 0.;
  double tX1=0., tX2=0., tY1=0., tY2=0.;
  double tdX, tdY;

  int aNbinsX = 160;
  double aMinX = 0.0;
  double aMaxX = 0.4;

  int aNbinsY = 100;
  double aMinY = 0.0;
  double aMaxY = 10.0;

  int tXbin = GetBinNumber(aNbinsX,aMinX,aMaxX,aX);
  int tYbin = GetBinNumber(aNbinsY,aMinY,aMaxY,aY);

  double tBinWidthX = (aMaxX-aMinX)/aNbinsX;
  double tBinMinX = aMinX + tXbin*tBinWidthX;
  double tBinMaxX = aMinX + (tXbin+1)*tBinWidthX;

  double tBinWidthY = (aMaxY-aMinY)/aNbinsY;
  double tBinMinY = aMinY + tYbin*tBinWidthY;
  double tBinMaxY = aMinY + (tYbin+1)*tBinWidthY;

  //---------------------------------
/*
  if(tXbin<0 || tYbin<0) 
  {
    cout << "Error in CoulombFitter::BilinearInterpolateVector, cannot interpolate outside histogram domain" << endl;
  }
  assert(tXbin >= 0);
  assert(tYbin >= 0);
*/
  //---------------------------------

  int tQuadrant = 0; //CCW from UR 1,2,3,4
  // which quadrant of the bin (bin_P) are we in?
  tdX = tBinMaxX - aX;
  tdY = tBinMaxY - aY;

  int tBinX1, tBinX2, tBinY1, tBinY2;

  if(tdX<=tBinWidthX/2 && tdY<=tBinWidthY/2) tQuadrant = 1; //upper right
  else if(tdX>tBinWidthX/2 && tdY<=tBinWidthY/2) tQuadrant = 2; //upper left
  else if(tdX>tBinWidthX/2 && tdY>tBinWidthY/2) tQuadrant = 3; //lower left
  else if(tdX<=tBinWidthX/2 && tdY>tBinWidthY/2) tQuadrant = 4; //lower right
//  else cout << "ERROR IN BilinearInterpolateVector!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;


  switch(tQuadrant)
  {
    case 1:
      tX1 = tBinMinX + tBinWidthX/2;
      tY1 = tBinMinY + tBinWidthY/2;
      tX2 = tBinMaxX + tBinWidthX/2;
      tY2 = tBinMaxY + tBinWidthY/2;

      tBinX1 = tXbin;
      tBinX2 = tXbin+1;
      tBinY1 = tYbin;
      tBinY2 = tYbin+1;

      break;
    case 2:
      tX1 = tBinMinX - tBinWidthX/2;
      tY1 = tBinMinY + tBinWidthY/2;
      tX2 = tBinMinX + tBinWidthX/2;
      tY2 = tBinMaxY + tBinWidthY/2;

      tBinX1 = tXbin-1;
      tBinX2 = tXbin;
      tBinY1 = tYbin;
      tBinY2 = tYbin+1;

      break;
    case 3:
      tX1 = tBinMinX - tBinWidthX/2;
      tY1 = tBinMinY - tBinWidthY/2;
      tX2 = tBinMinX + tBinWidthX/2;
      tY2 = tBinMinY + tBinWidthY/2;

      tBinX1 = tXbin-1;
      tBinX2 = tXbin;
      tBinY1 = tYbin-1;
      tBinY2 = tYbin;

      break;
    case 4:
      tX1 = tBinMinX + tBinWidthX/2;
      tY1 = tBinMinY - tBinWidthY/2;
      tX2 = tBinMaxX + tBinWidthX/2;
      tY2 = tBinMinY + tBinWidthY/2;

      tBinX1 = tXbin;
      tBinX2 = tXbin+1;
      tBinY1 = tYbin-1;
      tBinY2 = tYbin;

      break;
  }

  if(tBinX1<1) tBinX1 = 1;
  if(tBinX2>aNbinsX) tBinX2=aNbinsX;
  if(tBinY1<1) tBinY1 = 1;
  if(tBinY2>aNbinsY) tBinY2=aNbinsY;
/*
  double tQ11 = d_2dVecIn[tBinX1][tBinY1];
  double tQ12 = d_2dVecIn[tBinX1][tBinY2];
  double tQ21 = d_2dVecIn[tBinX2][tBinY1];
  double tQ22 = d_2dVecIn[tBinX2][tBinY2];
*/

  double tQ11 = d_2dVecIn[tBinY1 + tBinX1*100];
  double tQ12 = d_2dVecIn[tBinY2 + tBinX1*100];
  double tQ21 = d_2dVecIn[tBinY1 + tBinX2*100];
  double tQ22 = d_2dVecIn[tBinY2 + tBinX2*100];

  double tD = 1.0*(tX2-tX1)*(tY2-tY1);

  tF = (1.0/tD)*(tQ11*(tX2-aX)*(tY2-aY) + tQ21*(aX-tX1)*(tY2-aY) + tQ12*(tX2-aX)*(aY-tY1) + tQ22*(aX-tX1)*(aY-tY1));

  d_out[tIdx] = tF;
}




InterpolateGPU::InterpolateGPU()
{
}


InterpolateGPU::~InterpolateGPU()
{
}


vector<double> InterpolateGPU::RunBilinearInterpolate(vector<vector<double> > &aPairsIn, vector<vector<double> > &a2dVecIn)
{
  cudaSetDeviceFlags(cudaDeviceMapHost);

  int tNThreadsPerBlock = 1000;
  int tNBlocks = 10;
  int tNbinsK = 160;
  int tNbinsR = 100;
  int tElementsPerPair = 2;

  //Host arrays
  double * host_out;
  double * host_PairsIn;
  double * host_2dVecIn;

  double * device_out;
  double * device_PairsIn;
  double * device_2dVecIn;

  int sizeOut = tNThreadsPerBlock*tNBlocks*sizeof(double);
  int sizePairs = tNThreadsPerBlock*tNBlocks*tElementsPerPair*sizeof(double);
  int size2dVec = tNbinsK*tNbinsR*sizeof(double);

  checkCudaErrors(cudaHostAlloc((void**) &host_out, sizeOut, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &host_PairsIn, sizePairs, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &host_2dVecIn, size2dVec, cudaHostAllocMapped));

  for(int i=0; i<tNbinsK; i++)
  {
    for(int j=0; j<tNbinsR; j++)
    {
      host_2dVecIn[j+tNbinsR*i] = a2dVecIn[i][j];
    }
  }

  for(int i=0; i<tNThreadsPerBlock*tNBlocks; i++)
  {
    for(int j=0; j<tElementsPerPair; j++)
    {
      host_PairsIn[j+tElementsPerPair*i] = aPairsIn[i][j];
    }
  }


  checkCudaErrors(cudaHostGetDevicePointer(&device_out, host_out, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&device_PairsIn, host_PairsIn, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&device_2dVecIn, host_2dVecIn, 0));


  //---------------------
  GpuTimer timer;
  timer.Start();
  BilinearInterpolateVector<<<tNBlocks,tNThreadsPerBlock>>>(device_out,device_PairsIn,device_2dVecIn);
  timer.Stop();
  std::cout << "Kernel finished in " << timer.Elapsed() << "ms" << std::endl;

  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());

  //-------------------------------------
  vector<double> ReturnVector(tNThreadsPerBlock*tNBlocks);
  for(int i=0; i<tNThreadsPerBlock*tNBlocks; i++)
  {
    ReturnVector[i] = host_out[i];
  }

  checkCudaErrors(cudaFreeHost(host_out));
  checkCudaErrors(cudaFreeHost(host_PairsIn));
  checkCudaErrors(cudaFreeHost(host_2dVecIn));


  return ReturnVector;
}


double* InterpolateGPU::RunBilinearInterpolate(double* host_out, double *aPairsIn, double *a2dVecIn)
{
  cudaSetDeviceFlags(cudaDeviceMapHost);

  int tNThreadsPerBlock = 1000;
  int tNBlocks = 10;

  double * device_out;
  double * device_PairsIn;
  double * device_2dVecIn;

  GpuTimer timerCopy;
  timerCopy.Start();
  checkCudaErrors(cudaHostGetDevicePointer(&device_out, host_out, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&device_PairsIn, aPairsIn, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&device_2dVecIn, a2dVecIn, 0));
  timerCopy.Stop();
  cout << "Time to copy: " << timerCopy.Elapsed() << "ms" << endl;

  //---------------------
  GpuTimer timer;
  timer.Start();
  BilinearInterpolateVector<<<tNBlocks,tNThreadsPerBlock>>>(device_out,device_PairsIn,device_2dVecIn);
  timer.Stop();
  std::cout << "Kernel finished in " << timer.Elapsed() << "ms" << std::endl;

  //The following is necessary for the host to be able to "see" the changes that have been done
  GpuTimer timerSync;
  timerSync.Start();
  checkCudaErrors(cudaDeviceSynchronize());
  timerSync.Stop();
  cout << "Time to sync: " << timerSync.Elapsed() << "ms" << endl;

  //-------------------------------------
/*
  vector<double> ReturnVector(tNThreadsPerBlock);
  for(int i=0; i<tNThreadsPerBlock; i++)
  {
    ReturnVector[i] = host_out[i];
  }
*/

//  checkCudaErrors(cudaFreeHost(host_out));
  checkCudaErrors(cudaFreeHost(aPairsIn));
  checkCudaErrors(cudaFreeHost(a2dVecIn));


  return host_out;
}




