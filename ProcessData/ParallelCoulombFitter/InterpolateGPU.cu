///////////////////////////////////////////////////////////////////////////
// InterpolateGPU:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "InterpolateGPU.h"


//________________________________________________________________________________________________________________
__device__ int GetBinNumberTest(double aBinSize, int aNbins, double aValue)
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
__device__ int GetBinNumberTest(int aNbins, double aMin, double aMax, double aValue)
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
__device__ int GetBinNumberTest(double aBinWidth, double aMin, double aMax, double aValue)
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
__device__ int GetInterpLowBinTest(InterpType aInterpType, InterpAxisType aAxisType, double aVal)
{
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  bool tErrorFlag = false;

  switch(aInterpType)
  {
    case kGTilde:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = d_fGTildeInfo->nBinsK;
          tBinWidth = d_fGTildeInfo->binWidthK;
          tMin = d_fGTildeInfo->minK;
          tMax = d_fGTildeInfo->maxK;
          break;

        case kRaxis:
          tNbins = d_fGTildeInfo->nBinsR;
          tBinWidth = d_fGTildeInfo->binWidthR;
          tMin = d_fGTildeInfo->minR;
          tMax = d_fGTildeInfo->maxR;
          break;

        //Invalid axis selection
        case kThetaaxis:
          tErrorFlag = true;
          break;
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kHyperGeo1F1:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = d_fHyperGeo1F1Info->nBinsK;
          tBinWidth = d_fHyperGeo1F1Info->binWidthK;
          tMin = d_fHyperGeo1F1Info->minK;
          tMax = d_fHyperGeo1F1Info->maxK;
          break;

        case kRaxis:
          tNbins = d_fHyperGeo1F1Info->nBinsR;
          tBinWidth = d_fHyperGeo1F1Info->binWidthR;
          tMin = d_fHyperGeo1F1Info->minR;
          tMax = d_fHyperGeo1F1Info->maxR;
          break;

        case kThetaaxis:
          tNbins = d_fHyperGeo1F1Info->nBinsTheta;
          tBinWidth = d_fHyperGeo1F1Info->binWidthTheta;
          tMin = d_fHyperGeo1F1Info->minTheta;
          tMax = d_fHyperGeo1F1Info->maxTheta;
          break;

        //Invalid axis selection
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kScattLen:
      switch(aAxisType)
      {
        case kReF0axis:
          tNbins = d_fScattLenInfo->nBinsReF0;
          tBinWidth = d_fScattLenInfo->binWidthReF0;
          tMin = d_fScattLenInfo->minReF0;
          tMax = d_fScattLenInfo->maxReF0;
          break;

        case kImF0axis:
          tNbins = d_fScattLenInfo->nBinsImF0;
          tBinWidth = d_fScattLenInfo->binWidthImF0;
          tMin = d_fScattLenInfo->minImF0;
          tMax = d_fScattLenInfo->maxImF0;
          break;

        case kD0axis:
          tNbins = d_fScattLenInfo->nBinsD0;
          tBinWidth = d_fScattLenInfo->binWidthD0;
          tMin = d_fScattLenInfo->minD0;
          tMax = d_fScattLenInfo->maxD0;
          break;

        case kKaxis:
          tNbins = d_fScattLenInfo->nBinsK;
          tBinWidth = d_fScattLenInfo->binWidthK;
          tMin = d_fScattLenInfo->minK;
          tMax = d_fScattLenInfo->maxK;
          break;


        //Invalid axis selection
        case kRaxis:
          tErrorFlag = true;
          break;
        case kThetaaxis:
          tErrorFlag = true;
          break;

      }
      break;
  }

  //Check error
  if(tErrorFlag) return -3;

  //---------------------------------
  tBin = GetBinNumberTest(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0) return -2;
  if(tReturnBin>=tNbins) return -1;

//  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;
  else return tReturnBin;

}



//________________________________________________________________________________________________________________
__device__ int GetInterpLowBinTestKaxisTest(double aVal)
{
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  tNbins = 160;
  tBinWidth = 0.0025;
  tMin = 0.0;
  tMax = 0.40;

  //---------------------------------
  tBin = GetBinNumberTest(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;
  else return tReturnBin;

}

//________________________________________________________________________________________________________________
__device__ int GetInterpLowBinTestRaxisTest(double aVal)
{
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  tNbins = 100;
  tBinWidth = 0.1;
  tMin = 0.0;
  tMax = 10.0;

  //---------------------------------
  tBin = GetBinNumberTest(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;
  else return tReturnBin;

}


//________________________________________________________________________________________________________________
__device__ double GetInterpLowBinTestCenter(InterpType aInterpType, InterpAxisType aAxisType, double aVal)
{
  double tReturnValue;
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  bool tErrorFlag = false;

  switch(aInterpType)
  {
    case kGTilde:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = d_fGTildeInfo->nBinsK;
          tBinWidth = d_fGTildeInfo->binWidthK;
          tMin = d_fGTildeInfo->minK;
          tMax = d_fGTildeInfo->maxK;
          break;

        case kRaxis:
          tNbins = d_fGTildeInfo->nBinsR;
          tBinWidth = d_fGTildeInfo->binWidthR;
          tMin = d_fGTildeInfo->minR;
          tMax = d_fGTildeInfo->maxR;
          break;

        //Invalid axis selection
        case kThetaaxis:
          tErrorFlag = true;
          break;
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kHyperGeo1F1:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = d_fHyperGeo1F1Info->nBinsK;
          tBinWidth = d_fHyperGeo1F1Info->binWidthK;
          tMin = d_fHyperGeo1F1Info->minK;
          tMax = d_fHyperGeo1F1Info->maxK;
          break;

        case kRaxis:
          tNbins = d_fHyperGeo1F1Info->nBinsR;
          tBinWidth = d_fHyperGeo1F1Info->binWidthR;
          tMin = d_fHyperGeo1F1Info->minR;
          tMax = d_fHyperGeo1F1Info->maxR;
          break;

        case kThetaaxis:
          tNbins = d_fHyperGeo1F1Info->nBinsTheta;
          tBinWidth = d_fHyperGeo1F1Info->binWidthTheta;
          tMin = d_fHyperGeo1F1Info->minTheta;
          tMax = d_fHyperGeo1F1Info->maxTheta;
          break;

        //Invalid axis selection
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kScattLen:
      switch(aAxisType)
      {
        case kReF0axis:
          tNbins = d_fScattLenInfo->nBinsReF0;
          tBinWidth = d_fScattLenInfo->binWidthReF0;
          tMin = d_fScattLenInfo->minReF0;
          tMax = d_fScattLenInfo->maxReF0;
          break;

        case kImF0axis:
          tNbins = d_fScattLenInfo->nBinsImF0;
          tBinWidth = d_fScattLenInfo->binWidthImF0;
          tMin = d_fScattLenInfo->minImF0;
          tMax = d_fScattLenInfo->maxImF0;
          break;

        case kD0axis:
          tNbins = d_fScattLenInfo->nBinsD0;
          tBinWidth = d_fScattLenInfo->binWidthD0;
          tMin = d_fScattLenInfo->minD0;
          tMax = d_fScattLenInfo->maxD0;
          break;

        case kKaxis:
          tNbins = d_fScattLenInfo->nBinsK;
          tBinWidth = d_fScattLenInfo->binWidthK;
          tMin = d_fScattLenInfo->minK;
          tMax = d_fScattLenInfo->maxK;
          break;


        //Invalid axis selection
        case kRaxis:
          tErrorFlag = true;
          break;
        case kThetaaxis:
          tErrorFlag = true;
          break;

      }
      break;
  }

  //Check error
  if(tErrorFlag) return -2;

  //---------------------------------
  tBin = GetBinNumberTest(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;

  tReturnValue = tMin + (tReturnBin+0.5)*tBinWidth;
  return tReturnValue;
}


//________________________________________________________________________________________________________________
__device__ double GetInterpLowBinTestCenterKaxisTest(double aVal)
{
  double tReturnValue;
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  tNbins = 160;
  tBinWidth = 0.0025;
  tMin = 0.0;
  tMax = 0.40;


  //---------------------------------
  tBin = GetBinNumberTest(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;

  tReturnValue = tMin + (tReturnBin+0.5)*tBinWidth;
  return tReturnValue;
}

//________________________________________________________________________________________________________________
__device__ double GetInterpLowBinTestCenterRaxisTest(double aVal)
{
  double tReturnValue;
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  tNbins = 100;
  tBinWidth = 0.1;
  tMin = 0.0;
  tMax = 10.0;


  //---------------------------------
  tBin = GetBinNumberTest(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;

  tReturnValue = tMin + (tReturnBin+0.5)*tBinWidth;
  return tReturnValue;
}


//________________________________________________________________________________________________________________
__global__ void GTildeInterpolate(double* aKStar, double* aRStar, double* aGTildeReal)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  double tResultReal = 0.;
//  double tResultImag = 0.;
  //----------------------------

  int tNbinsR = d_fGTildeInfo->nBinsR;
  int tNbinsK = d_fGTildeInfo->nBinsK;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBinTestCenter does not return the error -2
  double tBinWidthK = d_fGTildeInfo->binWidthK;
//  double tBinWidthK = 0.0025;
  int tBinLowK = GetInterpLowBinTest(kGTilde,kKaxis,aKStar[idx]);
//  int tBinLowK = GetInterpLowBinTestKaxisTest(aKStar[idx]);
  int tBinHighK = tBinLowK+1;
  double tBinLowCenterK = GetInterpLowBinTestCenter(kGTilde,kKaxis,aKStar[idx]);
//  double tBinLowCenterK = GetInterpLowBinTestCenterKaxisTest(aKStar[idx]);
  double tBinHighCenterK = tBinLowCenterK+tBinWidthK;

  double tBinWidthR = d_fGTildeInfo->binWidthR;
//  double tBinWidthR = 0.1;
  int tBinLowR = GetInterpLowBinTest(kGTilde,kRaxis,aRStar[idx]);
//  int tBinLowR = GetInterpLowBinTestRaxisTest(aRStar[idx]);
  int tBinHighR = tBinLowR+1;
  double tBinLowCenterR = GetInterpLowBinTestCenter(kGTilde,kRaxis,aRStar[idx]);
//  double tBinLowCenterR = GetInterpLowBinTestCenterRaxisTest(aRStar[idx]);
  double tBinHighCenterR = tBinLowCenterR+tBinWidthR;

  //--------------------------
  assert(tBinLowK>=0);
  assert(tBinHighK<tNbinsK);
  assert(tBinLowCenterK>0);
  assert(tBinHighCenterK>0);


  assert(tBinLowR>-3);
  assert(tBinLowR>-2);
  assert(tBinLowR>-1);

  assert(tBinLowR>=0);
  assert(tBinHighR<tNbinsR);
  assert(tBinLowCenterR>0);
  assert(tBinHighCenterR>0);


  double tQ11Real = d_fGTildeReal[tBinLowR + tBinLowK*tNbinsR];
  double tQ12Real = d_fGTildeReal[tBinHighR + tBinLowK*tNbinsR];
  double tQ21Real = d_fGTildeReal[tBinLowR + tBinHighK*tNbinsR];
  double tQ22Real = d_fGTildeReal[tBinHighR + tBinHighK*tNbinsR];
/*
  double tQ11Imag = d_fGTildeImag[tBinLowR + tBinLowK*tNbinsR];
  double tQ12Imag = d_fGTildeImag[tBinHighR + tBinLowK*tNbinsR];
  double tQ21Imag = d_fGTildeImag[tBinLowR + tBinHighK*tNbinsR];
  double tQ22Imag = d_fGTildeImag[tBinHighR + tBinHighK*tNbinsR];
*/
//--------------------------

  double tD = 1.0*tBinWidthK*tBinWidthR;

  tResultReal = (1.0/tD)*(tQ11Real*(tBinHighCenterK-aKStar[idx])*(tBinHighCenterR-aRStar[idx]) + tQ21Real*(aKStar[idx]-tBinLowCenterK)*(tBinHighCenterR-aRStar[idx]) + tQ12Real*(tBinHighCenterK-aKStar[idx])*(aRStar[idx]-tBinLowCenterR) + tQ22Real*(aKStar[idx]-tBinLowCenterK)*(aRStar[idx]-tBinLowCenterR));

//  tResultImag = (1.0/tD)*(tQ11Imag*(tBinHighCenterK-aKStar)*(tBinHighCenterR-aRStar) + tQ21Imag*(aKStar-tBinLowCenterK)*(tBinHighCenterR-aRStar) + tQ12Imag*(tBinHighCenterK-aKStar)*(aRStar-tBinLowCenterR) + tQ22Imag*(aKStar-tBinLowCenterK)*(aRStar-tBinLowCenterR));


//--------------------------
  aGTildeReal[idx] = tResultReal;

}






















//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

InterpolateGPU::InterpolateGPU(int aNThreadsPerBlock, int aNBlocks) :
  fNThreadsPerBlock(aNThreadsPerBlock),
  fNBlocks(aNBlocks)
{
  cudaSetDeviceFlags(cudaDeviceMapHost);
}

//________________________________________________________________________________________________________________
InterpolateGPU::~InterpolateGPU()
{
}

//________________________________________________________________________________________________________________
void InterpolateGPU::LoadGTildeReal(td2dVec &aGTildeReal)
{
  int tNbinsK = aGTildeReal.size();
  int tNbinsR = aGTildeReal[0].size();

  int tSize = tNbinsK*tNbinsR*sizeof(double);

//  checkCudaErrors(cudaHostAlloc((void**) &fGTildeReal, tSize, cudaHostAllocMapped));
  checkCudaErrors(cudaMallocManaged(&d_fGTildeReal, tSize));

  int tIndex;
  for(int iK=0; iK<tNbinsK; iK++)
  {
    for(int iR=0; iR<tNbinsR; iR++)
    {
      tIndex = iR + iK*tNbinsR;
      d_fGTildeReal[tIndex] = aGTildeReal[iK][iR];
    }
  }

//  checkCudaErrors(cudaHostGetDevicePointer(&d_fGTildeReal, fGTildeReal, 0));

}


//________________________________________________________________________________________________________________
void InterpolateGPU::LoadGTildeImag(td2dVec &aGTildeImag)
{
  int tNbinsK = aGTildeImag.size();
  int tNbinsR = aGTildeImag[0].size();

  int tSize = tNbinsK*tNbinsR*sizeof(double);

//  checkCudaErrors(cudaHostAlloc((void**) &fGTildeImag, tSize, cudaHostAllocMapped));
  checkCudaErrors(cudaMallocManaged(&d_fGTildeImag, tSize));

  int tIndex;
  for(int iK=0; iK<tNbinsK; iK++)
  {
    for(int iR=0; iR<tNbinsR; iR++)
    {
      tIndex = iR + iK*tNbinsR;
      d_fGTildeImag[tIndex] = aGTildeImag[iK][iR];
    }
  }

//  checkCudaErrors(cudaHostGetDevicePointer(&d_fGTildeImag, fGTildeImag, 0));

}


//________________________________________________________________________________________________________________
void InterpolateGPU::LoadGTildeInfo(BinInfoGTilde &aBinInfo)
{
//  checkCudaErrors(cudaHostAlloc((void**) &fGTildeInfo, sizeof(BinInfoGTilde), cudaHostAllocMapped));
  checkCudaErrors(cudaMallocManaged(&d_fGTildeInfo, sizeof(BinInfoGTilde)));

  d_fGTildeInfo->nBinsK = aBinInfo.nBinsK; 
  d_fGTildeInfo->nBinsR = aBinInfo.nBinsR; 

  d_fGTildeInfo->binWidthK = aBinInfo.binWidthK; 
  d_fGTildeInfo->binWidthR = aBinInfo.binWidthR; 

  d_fGTildeInfo->minK = aBinInfo.minK; 
  d_fGTildeInfo->maxK = aBinInfo.maxK; 
  d_fGTildeInfo->minR = aBinInfo.minR; 
  d_fGTildeInfo->maxR = aBinInfo.maxR; 

  d_fGTildeInfo->minInterpK = aBinInfo.minInterpK; 
  d_fGTildeInfo->maxInterpK = aBinInfo.maxInterpK; 
  d_fGTildeInfo->minInterpR = aBinInfo.minInterpR; 
  d_fGTildeInfo->maxInterpR = aBinInfo.maxInterpR; 

//  checkCudaErrors(cudaHostGetDevicePointer(&d_fGTildeInfo, fGTildeInfo, 0));
}

//________________________________________________________________________________________________________________
vector<double> InterpolateGPU::RunBilinearInterpolate(vector<vector<double> > &aPairsIn)
{
//  cudaSetDeviceFlags(cudaDeviceMapHost);

  int tNPairs = aPairsIn.size();
  int tSize = tNPairs*sizeof(double);


  //---Host array allocations
  double * h_KStarMag;
  double * h_RStarMag;
  double * h_GTildeReal;

  checkCudaErrors(cudaHostAlloc((void**) &h_KStarMag, tSize, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &h_RStarMag, tSize, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &h_GTildeReal, tSize, cudaHostAllocMapped));

  for(int i=0; i<tNPairs; i++)
  {
    h_KStarMag[i] = aPairsIn[i][0];
    h_RStarMag[i] = aPairsIn[i][1];
  }


  //---Device array allocations
  //---Device arrays and allocations
  double * d_KStarMag;
  double * d_RStarMag;
  double * d_GTildeReal;

  checkCudaErrors(cudaHostGetDevicePointer(&d_KStarMag, h_KStarMag, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&d_RStarMag, h_RStarMag, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&d_GTildeReal, h_GTildeReal, 0));


  //----------Run the kernel-----------------------------------------------
  GpuTimer timer;
  timer.Start();
  GTildeInterpolate<<<fNBlocks,fNThreadsPerBlock>>>(d_KStarMag,d_RStarMag,d_GTildeReal);
  timer.Stop();
  std::cout << "GTildeInterpolate kernel finished in " << timer.Elapsed() << " ms" << std::endl;

  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFreeHost(h_KStarMag));
  checkCudaErrors(cudaFreeHost(h_RStarMag));

  vector<double> tReturnVec(tNPairs);
  for(int i=0; i<tNPairs; i++)
  {
    tReturnVec[i] = h_GTildeReal[i];
//    cout << "i = " << i << endl;
//    cout << "h_GTildeReal[i] = " << h_GTildeReal[i] << endl;
//    cout << "tReturnVec[i] = " << tReturnVec[i] << endl << endl;
  }

  checkCudaErrors(cudaFreeHost(h_GTildeReal));
  return tReturnVec;
}
/*
//________________________________________________________________________________________________________________
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

//  vector<double> ReturnVector(tNThreadsPerBlock);
//  for(int i=0; i<tNThreadsPerBlock; i++)
//  {
//    ReturnVector[i] = host_out[i];
//  }


//  checkCudaErrors(cudaFreeHost(host_out));
  checkCudaErrors(cudaFreeHost(aPairsIn));
  checkCudaErrors(cudaFreeHost(a2dVecIn));


  return host_out;
}
*/



