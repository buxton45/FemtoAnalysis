///////////////////////////////////////////////////////////////////////////
// ParallelWaveFunction:                                                 //
///////////////////////////////////////////////////////////////////////////

#include "ParallelWaveFunction.h"

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
__device__ int GetInterpLowBin(InterpType aInterpType, InterpAxisType aAxisType, double aVal)
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
  if(tErrorFlag) return -2;

  //---------------------------------
  tBin = GetBinNumber(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;
  else return tReturnBin;

}

//________________________________________________________________________________________________________________
__device__ double GetInterpLowBinCenter(InterpType aInterpType, InterpAxisType aAxisType, double aVal)
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
  tBin = GetBinNumber(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;

  tReturnValue = tMin + (tReturnBin+0.5)*tBinWidth;
  return tReturnValue;
}



//________________________________________________________________________________________________________________
__device__ cuDoubleComplex GTildeInterpolate(double aKStar, double aRStar)
{
  double tResultReal = 0.;
  double tResultImag = 0.;
  //----------------------------

  int tNbinsR = d_fGTildeInfo->nBinsR;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBinCenter does not return the error -2
  double tBinWidthK = d_fGTildeInfo->binWidthK;
  int tBinLowK = GetInterpLowBin(kGTilde,kKaxis,aKStar);
  int tBinHighK = tBinLowK+1;
  double tBinLowCenterK = GetInterpLowBinCenter(kGTilde,kKaxis,aKStar);
  double tBinHighCenterK = tBinLowCenterK+tBinWidthK;

  double tBinWidthR = d_fGTildeInfo->binWidthR;
  int tBinLowR = GetInterpLowBin(kGTilde,kRaxis,aRStar);
  int tBinHighR = tBinLowR+1;
  double tBinLowCenterR = GetInterpLowBinCenter(kGTilde,kRaxis,aRStar);
  double tBinHighCenterR = tBinLowCenterR+tBinWidthR;

  //--------------------------

  double tQ11Real = d_fGTildeReal[tBinLowR + tBinLowK*tNbinsR];
  double tQ12Real = d_fGTildeReal[tBinHighR + tBinLowK*tNbinsR];
  double tQ21Real = d_fGTildeReal[tBinLowR + tBinHighK*tNbinsR];
  double tQ22Real = d_fGTildeReal[tBinHighR + tBinHighK*tNbinsR];

  double tQ11Imag = d_fGTildeImag[tBinLowR + tBinLowK*tNbinsR];
  double tQ12Imag = d_fGTildeImag[tBinHighR + tBinLowK*tNbinsR];
  double tQ21Imag = d_fGTildeImag[tBinLowR + tBinHighK*tNbinsR];
  double tQ22Imag = d_fGTildeImag[tBinHighR + tBinHighK*tNbinsR];

//--------------------------

  double tD = 1.0*tBinWidthK*tBinWidthR;

  tResultReal = (1.0/tD)*(tQ11Real*(tBinHighCenterK-aKStar)*(tBinHighCenterR-aRStar) + tQ21Real*(aKStar-tBinLowCenterK)*(tBinHighCenterR-aRStar) + tQ12Real*(tBinHighCenterK-aKStar)*(aRStar-tBinLowCenterR) + tQ22Real*(aKStar-tBinLowCenterK)*(aRStar-tBinLowCenterR));

  tResultImag = (1.0/tD)*(tQ11Imag*(tBinHighCenterK-aKStar)*(tBinHighCenterR-aRStar) + tQ21Imag*(aKStar-tBinLowCenterK)*(tBinHighCenterR-aRStar) + tQ12Imag*(tBinHighCenterK-aKStar)*(aRStar-tBinLowCenterR) + tQ22Imag*(aKStar-tBinLowCenterK)*(aRStar-tBinLowCenterR));


//--------------------------
  cuDoubleComplex tReturnValue = make_cuDoubleComplex(tResultReal,tResultImag);
  return tReturnValue;
}


//________________________________________________________________________________________________________________
__device__ cuDoubleComplex HyperGeo1F1Interpolate(double aKStar, double aRStar, double aTheta)
{
  double tResultReal = 0.;
  double tResultImag = 0.;
  //----------------------------

  int tNbinsTheta = d_fHyperGeo1F1Info->nBinsTheta;
  int tNbinsR = d_fHyperGeo1F1Info->nBinsR;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBinCenter does not return the error -2
  double tBinWidthK = d_fHyperGeo1F1Info->binWidthK;
  int tBin0K = GetInterpLowBin(kHyperGeo1F1,kKaxis,aKStar);
  int tBin1K = tBin0K+1;
  double tBin0CenterK = GetInterpLowBinCenter(kHyperGeo1F1,kKaxis,aKStar);
//  double tBin1CenterK = tBin0CenterK+tBinWidthK;

  double tBinWidthR = d_fHyperGeo1F1Info->binWidthR;
  int tBin0R = GetInterpLowBin(kHyperGeo1F1,kRaxis,aRStar);
  int tBin1R = tBin0R+1;
  double tBin0CenterR = GetInterpLowBinCenter(kHyperGeo1F1,kRaxis,aRStar);
//  double tBin1CenterR = tBin0CenterR+tBinWidthR;

  double tBinWidthTheta = d_fHyperGeo1F1Info->binWidthTheta;
  int tBin0Theta = GetInterpLowBin(kHyperGeo1F1,kThetaaxis,aTheta);
  int tBin1Theta = tBin0Theta+1;
  double tBin0CenterTheta = GetInterpLowBinCenter(kHyperGeo1F1,kThetaaxis,aTheta);
//  double tBin1CenterTheta = tBin0CenterTheta+tBinWidthTheta;

  //--------------------------

  double tDiffK = (aKStar - tBin0CenterK)/tBinWidthK;
  double tDiffR = (aRStar - tBin0CenterR)/tBinWidthR;
  double tDiffTheta = (aTheta - tBin0CenterTheta)/tBinWidthTheta;

  //-----------REAL---------------
  //interpolate along z (i.e. theta)
  double tC000Real = d_fHyperGeo1F1Real[tBin0K*tNbinsTheta*tNbinsR + tBin0R*tNbinsTheta + tBin0Theta];
  double tC001Real = d_fHyperGeo1F1Real[tBin0K*tNbinsTheta*tNbinsR + tBin0R*tNbinsTheta + tBin1Theta];

  double tC010Real = d_fHyperGeo1F1Real[tBin0K*tNbinsTheta*tNbinsR + tBin1R*tNbinsTheta + tBin0Theta];
  double tC011Real = d_fHyperGeo1F1Real[tBin0K*tNbinsTheta*tNbinsR + tBin1R*tNbinsTheta + tBin1Theta];

  double tC100Real = d_fHyperGeo1F1Real[tBin1K*tNbinsTheta*tNbinsR + tBin0R*tNbinsTheta + tBin0Theta];
  double tC101Real = d_fHyperGeo1F1Real[tBin1K*tNbinsTheta*tNbinsR + tBin0R*tNbinsTheta + tBin1Theta];

  double tC110Real = d_fHyperGeo1F1Real[tBin1K*tNbinsTheta*tNbinsR + tBin1R*tNbinsTheta + tBin0Theta];
  double tC111Real = d_fHyperGeo1F1Real[tBin1K*tNbinsTheta*tNbinsR + tBin1R*tNbinsTheta + tBin1Theta];

  double tC00Real = tC000Real*(1.0-tDiffTheta) + tC001Real*tDiffTheta;
  double tC01Real = tC010Real*(1.0-tDiffTheta) + tC011Real*tDiffTheta;
  double tC10Real = tC100Real*(1.0-tDiffTheta) + tC101Real*tDiffTheta;
  double tC11Real = tC110Real*(1.0-tDiffTheta) + tC111Real*tDiffTheta;

  //interpolate along y (i.e. r)
  double tC0Real = tC00Real*(1.0-tDiffR) + tC01Real*tDiffR;
  double tC1Real = tC10Real*(1.0-tDiffR) + tC11Real*tDiffR;

  //interpolate along x (i.e. k)
  tResultReal = tC0Real*(1.0-tDiffK) + tC1Real*tDiffK;

  //-----------IMAG---------------
  //interpolate along z (i.e. theta)
  double tC000Imag = d_fHyperGeo1F1Imag[tBin0K*tNbinsTheta*tNbinsR + tBin0R*tNbinsTheta + tBin0Theta];
  double tC001Imag = d_fHyperGeo1F1Imag[tBin0K*tNbinsTheta*tNbinsR + tBin0R*tNbinsTheta + tBin1Theta];

  double tC010Imag = d_fHyperGeo1F1Imag[tBin0K*tNbinsTheta*tNbinsR + tBin1R*tNbinsTheta + tBin0Theta];
  double tC011Imag = d_fHyperGeo1F1Imag[tBin0K*tNbinsTheta*tNbinsR + tBin1R*tNbinsTheta + tBin1Theta];

  double tC100Imag = d_fHyperGeo1F1Imag[tBin1K*tNbinsTheta*tNbinsR + tBin0R*tNbinsTheta + tBin0Theta];
  double tC101Imag = d_fHyperGeo1F1Imag[tBin1K*tNbinsTheta*tNbinsR + tBin0R*tNbinsTheta + tBin1Theta];

  double tC110Imag = d_fHyperGeo1F1Imag[tBin1K*tNbinsTheta*tNbinsR + tBin1R*tNbinsTheta + tBin0Theta];
  double tC111Imag = d_fHyperGeo1F1Imag[tBin1K*tNbinsTheta*tNbinsR + tBin1R*tNbinsTheta + tBin1Theta];

  double tC00Imag = tC000Imag*(1.0-tDiffTheta) + tC001Imag*tDiffTheta;
  double tC01Imag = tC010Imag*(1.0-tDiffTheta) + tC011Imag*tDiffTheta;
  double tC10Imag = tC100Imag*(1.0-tDiffTheta) + tC101Imag*tDiffTheta;
  double tC11Imag = tC110Imag*(1.0-tDiffTheta) + tC111Imag*tDiffTheta;

  //interpolate along y (i.e. r)
  double tC0Imag = tC00Imag*(1.0-tDiffR) + tC01Imag*tDiffR;
  double tC1Imag = tC10Imag*(1.0-tDiffR) + tC11Imag*tDiffR;

  //interpolate along x (i.e. k)
  tResultImag = tC0Imag*(1.0-tDiffK) + tC1Imag*tDiffK;

  //--------------------------------
  cuDoubleComplex tReturnValue = make_cuDoubleComplex(tResultReal,tResultImag);
  return tReturnValue;
}

/*
//________________________________________________________________________________________________________________
__device__ cuDoubleComplex ScattLenInterpolateFull(double aReF0, double aImF0, double aD0, double aKStar)
{
//This doesn't work because d_fCoulombScatteringLengthReal and d_fCoulombScatteringLengthImag are
// too big to fit onto the GPU memory. I am keeping it in case I figure out how to resolve the memory issue
// i.e. figure out how to let the device directly access host memory

  double tResultReal = 0.;
  double tResultImag = 0.;
  //----------------------------

  int tNbinsK = d_fScattLenInfo->nBinsK;
  int tNbinsD0 = d_fScattLenInfo->nBinsD0;
  int tNbinsImF0 = d_fScattLenInfo->nBinsImF0;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBinCenter does not return the error -2
  double tBinWidthReF0 = d_fScattLenInfo->binWidthReF0;
  int tBin0ReF0 = GetInterpLowBin(kScattLen,kReF0axis,aReF0);
  int tBin1ReF0 = tBin0ReF0+1;
  double tBin0CenterReF0 = GetInterpLowBinCenter(kScattLen,kReF0axis,aReF0);
//  double tBin1CenterReF0 = tBin0CenterReF0+tBinWidthReF0;

  double tBinWidthImF0 = d_fScattLenInfo->binWidthImF0;
  int tBin0ImF0 = GetInterpLowBin(kScattLen,kImF0axis,aImF0);
  int tBin1ImF0 = tBin0ImF0+1;
  double tBin0CenterImF0 = GetInterpLowBinCenter(kScattLen,kImF0axis,aImF0);
//  double tBin1CenterImF0 = tBin0CenterImF0+tBinWidthImF0;

  double tBinWidthD0 = d_fScattLenInfo->binWidthD0;
  int tBin0D0 = GetInterpLowBin(kScattLen,kD0axis,aD0);
  int tBin1D0 = tBin0D0+1;
  double tBin0CenterD0 = GetInterpLowBinCenter(kScattLen,kD0axis,aD0);
//  double tBin1CenterD0 = tBin0CenterD0+tBinWidthD0;

  double tBinWidthK = d_fScattLenInfo->binWidthK;
  int tBin0K = GetInterpLowBin(kScattLen,kKaxis,aKStar);
  int tBin1K = tBin0K+1;
  double tBin0CenterK = GetInterpLowBinCenter(kScattLen,kKaxis,aKStar);
//  double tBin1CenterK = tBin0CenterK+tBinWidthK;

  //--------------------------

  double tDiffReF0 = (aReF0 - tBin0CenterReF0)/tBinWidthReF0;
  double tDiffImF0 = (aImF0 - tBin0CenterImF0)/tBinWidthImF0;
  double tDiffD0 = (aD0 - tBin0CenterD0)/tBinWidthD0;
  double tDiffK = (aKStar - tBin0CenterK)/tBinWidthK;

  //--------------------------
  //Assuming f(t,x,y,z) = f(ReF0,ImF0,D0,KStar).  Ordering for memory access reasons

  //---------------REAL----------------------------------
  //interpolate along z (i.e. KStar)
  double tC0000Real = d_fCoulombScatteringLengthReal[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC0001Real = d_fCoulombScatteringLengthReal[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC0010Real = d_fCoulombScatteringLengthReal[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC0011Real = d_fCoulombScatteringLengthReal[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC0100Real = d_fCoulombScatteringLengthReal[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC0101Real = d_fCoulombScatteringLengthReal[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC0110Real = d_fCoulombScatteringLengthReal[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC0111Real = d_fCoulombScatteringLengthReal[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC1000Real = d_fCoulombScatteringLengthReal[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC1001Real = d_fCoulombScatteringLengthReal[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC1010Real = d_fCoulombScatteringLengthReal[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC1011Real = d_fCoulombScatteringLengthReal[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC1100Real = d_fCoulombScatteringLengthReal[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC1101Real = d_fCoulombScatteringLengthReal[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC1110Real = d_fCoulombScatteringLengthReal[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC1111Real = d_fCoulombScatteringLengthReal[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  //---
  double tC000Real = tC0000Real*(1.0-tDiffK) + tC0001Real*tDiffK;
  double tC001Real = tC0010Real*(1.0-tDiffK) + tC0011Real*tDiffK;

  double tC010Real = tC0100Real*(1.0-tDiffK) + tC0101Real*tDiffK;
  double tC011Real = tC0110Real*(1.0-tDiffK) + tC0111Real*tDiffK;

  double tC100Real = tC1000Real*(1.0-tDiffK) + tC1001Real*tDiffK;
  double tC101Real = tC1010Real*(1.0-tDiffK) + tC1011Real*tDiffK;

  double tC110Real = tC1100Real*(1.0-tDiffK) + tC1101Real*tDiffK;
  double tC111Real = tC1110Real*(1.0-tDiffK) + tC1111Real*tDiffK;

  //interpolate along y (i.e. D0)
  double tC00Real = tC000Real*(1.0-tDiffD0) + tC001Real*tDiffD0;
  double tC01Real = tC010Real*(1.0-tDiffD0) + tC011Real*tDiffD0;

  double tC10Real = tC100Real*(1.0-tDiffD0) + tC101Real*tDiffD0;
  double tC11Real = tC110Real*(1.0-tDiffD0) + tC111Real*tDiffD0;

  //interpolate along x (i.e. ImF0)
  double tC0Real = tC00Real*(1.0-tDiffImF0) + tC01Real*tDiffImF0;
  double tC1Real = tC10Real*(1.0-tDiffImF0) + tC11Real*tDiffImF0;

  //interpolate along t (i.e. ReF0)
  tResultReal = tC0Real*(1.0-tDiffReF0) + tC1Real*tDiffReF0;


  //---------------Imag----------------------------------
  //interpolate along z (i.e. KStar)
  double tC0000Imag = d_fCoulombScatteringLengthImag[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC0001Imag = d_fCoulombScatteringLengthImag[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC0010Imag = d_fCoulombScatteringLengthImag[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC0011Imag = d_fCoulombScatteringLengthImag[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC0100Imag = d_fCoulombScatteringLengthImag[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC0101Imag = d_fCoulombScatteringLengthImag[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC0110Imag = d_fCoulombScatteringLengthImag[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC0111Imag = d_fCoulombScatteringLengthImag[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC1000Imag = d_fCoulombScatteringLengthImag[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC1001Imag = d_fCoulombScatteringLengthImag[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC1010Imag = d_fCoulombScatteringLengthImag[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC1011Imag = d_fCoulombScatteringLengthImag[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC1100Imag = d_fCoulombScatteringLengthImag[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC1101Imag = d_fCoulombScatteringLengthImag[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC1110Imag = d_fCoulombScatteringLengthImag[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC1111Imag = d_fCoulombScatteringLengthImag[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  //---
  double tC000Imag = tC0000Imag*(1.0-tDiffK) + tC0001Imag*tDiffK;
  double tC001Imag = tC0010Imag*(1.0-tDiffK) + tC0011Imag*tDiffK;

  double tC010Imag = tC0100Imag*(1.0-tDiffK) + tC0101Imag*tDiffK;
  double tC011Imag = tC0110Imag*(1.0-tDiffK) + tC0111Imag*tDiffK;

  double tC100Imag = tC1000Imag*(1.0-tDiffK) + tC1001Imag*tDiffK;
  double tC101Imag = tC1010Imag*(1.0-tDiffK) + tC1011Imag*tDiffK;

  double tC110Imag = tC1100Imag*(1.0-tDiffK) + tC1101Imag*tDiffK;
  double tC111Imag = tC1110Imag*(1.0-tDiffK) + tC1111Imag*tDiffK;

  //interpolate along y (i.e. D0)
  double tC00Imag = tC000Imag*(1.0-tDiffD0) + tC001Imag*tDiffD0;
  double tC01Imag = tC010Imag*(1.0-tDiffD0) + tC011Imag*tDiffD0;

  double tC10Imag = tC100Imag*(1.0-tDiffD0) + tC101Imag*tDiffD0;
  double tC11Imag = tC110Imag*(1.0-tDiffD0) + tC111Imag*tDiffD0;

  //interpolate along x (i.e. ImF0)
  double tC0Imag = tC00Imag*(1.0-tDiffImF0) + tC01Imag*tDiffImF0;
  double tC1Imag = tC10Imag*(1.0-tDiffImF0) + tC11Imag*tDiffImF0;

  //interpolate along t (i.e. ReF0)
  tResultImag = tC0Imag*(1.0-tDiffReF0) + tC1Imag*tDiffReF0;


  //--------------------------------
  cuDoubleComplex tReturnValue = make_cuDoubleComplex(tResultReal,tResultImag);
  return tReturnValue;

}
*/

//________________________________________________________________________________________________________________
__device__ cuDoubleComplex ScattLenInterpolate(double aReF0, double aImF0, double aD0, double aKStar)
{
  double tResultReal = 0.;
  double tResultImag = 0.;
  //----------------------------

  int tNbinsK = d_fScattLenInfo->nBinsK;
//  int tNbinsD0 = d_fScattLenInfo->nBinsD0;
//  int tNbinsImF0 = d_fScattLenInfo->nBinsImF0;
  int tNbinsD0 = 2;
  int tNbinsImF0 = 2;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBinCenter does not return the error -2
  double tBinWidthReF0 = d_fScattLenInfo->binWidthReF0;
  int tBin0ReF0 = 0;
  int tBin1ReF0 = tBin0ReF0+1;
  double tBin0CenterReF0 = GetInterpLowBinCenter(kScattLen,kReF0axis,aReF0);
//  double tBin1CenterReF0 = tBin0CenterReF0+tBinWidthReF0;

  double tBinWidthImF0 = d_fScattLenInfo->binWidthImF0;
  int tBin0ImF0 = 0;
  int tBin1ImF0 = tBin0ImF0+1;
  double tBin0CenterImF0 = GetInterpLowBinCenter(kScattLen,kImF0axis,aImF0);
//  double tBin1CenterImF0 = tBin0CenterImF0+tBinWidthImF0;

  double tBinWidthD0 = d_fScattLenInfo->binWidthD0;
  int tBin0D0 = 0;
  int tBin1D0 = tBin0D0+1;
  double tBin0CenterD0 = GetInterpLowBinCenter(kScattLen,kD0axis,aD0);
//  double tBin1CenterD0 = tBin0CenterD0+tBinWidthD0;

  double tBinWidthK = d_fScattLenInfo->binWidthK;
  int tBin0K = GetInterpLowBin(kScattLen,kKaxis,aKStar);
  int tBin1K = tBin0K+1;
  double tBin0CenterK = GetInterpLowBinCenter(kScattLen,kKaxis,aKStar);
//  double tBin1CenterK = tBin0CenterK+tBinWidthK;

  //--------------------------
  assert(tBin0K>=0);
  assert(tBin0CenterK>0);

  double tDiffReF0 = (aReF0 - tBin0CenterReF0)/tBinWidthReF0;
  double tDiffImF0 = (aImF0 - tBin0CenterImF0)/tBinWidthImF0;
  double tDiffD0 = (aD0 - tBin0CenterD0)/tBinWidthD0;
  double tDiffK = (aKStar - tBin0CenterK)/tBinWidthK;

  //--------------------------
  //Assuming f(t,x,y,z) = f(ReF0,ImF0,D0,KStar).  Ordering for memory access reasons

  //---------------REAL----------------------------------
  //interpolate along z (i.e. KStar)
  double tC0000Real = d_fScattLenRealSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC0001Real = d_fScattLenRealSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC0010Real = d_fScattLenRealSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC0011Real = d_fScattLenRealSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC0100Real = d_fScattLenRealSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC0101Real = d_fScattLenRealSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC0110Real = d_fScattLenRealSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC0111Real = d_fScattLenRealSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC1000Real = d_fScattLenRealSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC1001Real = d_fScattLenRealSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC1010Real = d_fScattLenRealSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC1011Real = d_fScattLenRealSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC1100Real = d_fScattLenRealSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC1101Real = d_fScattLenRealSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC1110Real = d_fScattLenRealSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC1111Real = d_fScattLenRealSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  //---
  double tC000Real = tC0000Real*(1.0-tDiffK) + tC0001Real*tDiffK;
  double tC001Real = tC0010Real*(1.0-tDiffK) + tC0011Real*tDiffK;

  double tC010Real = tC0100Real*(1.0-tDiffK) + tC0101Real*tDiffK;
  double tC011Real = tC0110Real*(1.0-tDiffK) + tC0111Real*tDiffK;

  double tC100Real = tC1000Real*(1.0-tDiffK) + tC1001Real*tDiffK;
  double tC101Real = tC1010Real*(1.0-tDiffK) + tC1011Real*tDiffK;

  double tC110Real = tC1100Real*(1.0-tDiffK) + tC1101Real*tDiffK;
  double tC111Real = tC1110Real*(1.0-tDiffK) + tC1111Real*tDiffK;

  //interpolate along y (i.e. D0)
  double tC00Real = tC000Real*(1.0-tDiffD0) + tC001Real*tDiffD0;
  double tC01Real = tC010Real*(1.0-tDiffD0) + tC011Real*tDiffD0;

  double tC10Real = tC100Real*(1.0-tDiffD0) + tC101Real*tDiffD0;
  double tC11Real = tC110Real*(1.0-tDiffD0) + tC111Real*tDiffD0;

  //interpolate along x (i.e. ImF0)
  double tC0Real = tC00Real*(1.0-tDiffImF0) + tC01Real*tDiffImF0;
  double tC1Real = tC10Real*(1.0-tDiffImF0) + tC11Real*tDiffImF0;

  //interpolate along t (i.e. ReF0)
  tResultReal = tC0Real*(1.0-tDiffReF0) + tC1Real*tDiffReF0;


  //---------------Imag----------------------------------
  //interpolate along z (i.e. KStar)
  double tC0000Imag = d_fScattLenImagSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC0001Imag = d_fScattLenImagSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC0010Imag = d_fScattLenImagSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC0011Imag = d_fScattLenImagSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC0100Imag = d_fScattLenImagSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC0101Imag = d_fScattLenImagSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC0110Imag = d_fScattLenImagSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC0111Imag = d_fScattLenImagSubVec[tBin0ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC1000Imag = d_fScattLenImagSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC1001Imag = d_fScattLenImagSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC1010Imag = d_fScattLenImagSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC1011Imag = d_fScattLenImagSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin0ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  double tC1100Imag = d_fScattLenImagSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin0K];
  double tC1101Imag = d_fScattLenImagSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin0D0*tNbinsK + tBin1K];

  double tC1110Imag = d_fScattLenImagSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin0K];
  double tC1111Imag = d_fScattLenImagSubVec[tBin1ReF0*tNbinsImF0*tNbinsD0*tNbinsK + tBin1ImF0*tNbinsD0*tNbinsK + tBin1D0*tNbinsK + tBin1K];

  //---
  double tC000Imag = tC0000Imag*(1.0-tDiffK) + tC0001Imag*tDiffK;
  double tC001Imag = tC0010Imag*(1.0-tDiffK) + tC0011Imag*tDiffK;

  double tC010Imag = tC0100Imag*(1.0-tDiffK) + tC0101Imag*tDiffK;
  double tC011Imag = tC0110Imag*(1.0-tDiffK) + tC0111Imag*tDiffK;

  double tC100Imag = tC1000Imag*(1.0-tDiffK) + tC1001Imag*tDiffK;
  double tC101Imag = tC1010Imag*(1.0-tDiffK) + tC1011Imag*tDiffK;

  double tC110Imag = tC1100Imag*(1.0-tDiffK) + tC1101Imag*tDiffK;
  double tC111Imag = tC1110Imag*(1.0-tDiffK) + tC1111Imag*tDiffK;

  //interpolate along y (i.e. D0)
  double tC00Imag = tC000Imag*(1.0-tDiffD0) + tC001Imag*tDiffD0;
  double tC01Imag = tC010Imag*(1.0-tDiffD0) + tC011Imag*tDiffD0;

  double tC10Imag = tC100Imag*(1.0-tDiffD0) + tC101Imag*tDiffD0;
  double tC11Imag = tC110Imag*(1.0-tDiffD0) + tC111Imag*tDiffD0;

  //interpolate along x (i.e. ImF0)
  double tC0Imag = tC00Imag*(1.0-tDiffImF0) + tC01Imag*tDiffImF0;
  double tC1Imag = tC10Imag*(1.0-tDiffImF0) + tC11Imag*tDiffImF0;

  //interpolate along t (i.e. ReF0)
  tResultImag = tC0Imag*(1.0-tDiffReF0) + tC1Imag*tDiffReF0;


  //--------------------------------
  cuDoubleComplex tReturnValue = make_cuDoubleComplex(tResultReal,tResultImag);
  return tReturnValue;

}


//________________________________________________________________________________________________________________
__device__ double GetGamowFactor(double aKStar)
{
  double d_hbarc = 0.197327;
  double d_gBohrRadius = 75.23349845;

  //TODO figure out how to load hbarc and gBohrRadius into GPU
  //TODO figure out how to use Pi here
  //TODO figure out how to make bohr radius negative when needed

  double tEta = pow(((aKStar/d_hbarc)*d_gBohrRadius),-1);
  tEta *= 6.28318530718;  //eta always comes with 2Pi here
  double tGamow = tEta*pow((exp(tEta)-1),-1);

  return tGamow;
}

//________________________________________________________________________________________________________________
__device__ cuDoubleComplex GetExpTerm(double aKStar, double aRStar, double aTheta)
{
  //TODO figure out how to load hbarc and gBohrRadius into GPU
  double d_hbarc = 0.197327;

  double tReal = cos((aKStar/d_hbarc)*aRStar*cos(aTheta));
  double tImag = -sin((aKStar/d_hbarc)*aRStar*cos(aTheta));

  cuDoubleComplex tExpTermCmplx = make_cuDoubleComplex(tReal,tImag);
  return tExpTermCmplx;
}

//________________________________________________________________________________________________________________
__device__ double AssembleWfSquared(double aRStarMag, double aGamowFactor, cuDoubleComplex aExpTermCmplx, cuDoubleComplex aGTildeCmplx, cuDoubleComplex aHyperGeo1F1Cmplx, cuDoubleComplex aScattLenCmplx)
{
  cuDoubleComplex tGTildeCmplxConj = cuConj(aGTildeCmplx);
  cuDoubleComplex tScattLenCmplxConj = cuConj(aScattLenCmplx);
//  cuDoubleComplex tGamowFactor = make_cuDoubleComplex(aGamowFactor,0.);  //cuda doesn't want to multiply double*double2


  //-------------Stupid cuda can only multiple/divide two at once
  //TODO test to see if there is an easier way to accomplish this
  double tMagSq_HyperGeo1F1 = cuCabs(aHyperGeo1F1Cmplx)*cuCabs(aHyperGeo1F1Cmplx);
  double tMagSq_ScattLen = cuCabs(tScattLenCmplxConj)*cuCabs(tScattLenCmplxConj);
  double tMagSq_GTilde = cuCabs(tGTildeCmplxConj)*cuCabs(tGTildeCmplxConj);

  cuDoubleComplex tTerm1 = cuCmul(aExpTermCmplx,aHyperGeo1F1Cmplx);
  cuDoubleComplex tTerm2 = cuCmul(tScattLenCmplxConj,tGTildeCmplxConj);
  cuDoubleComplex tTerm12 = cuCmul(tTerm1,tTerm2);
  double tTerm12Real = cuCreal(tTerm12);
  double tTermFinal = tTerm12Real/aRStarMag;
/*
  cuDoubleComplex tRStarMagCmplx = make_cuDoubleComplex(aRStarMag,0.);
  cuDoubleComplex tTermFinalCmplx = cuCdiv(tTerm12,tRStarMagCmplx);
  double tTermFinal = cuCreal(tTermFinalCmplx);
*/


  double tResult = aGamowFactor*(tMagSq_HyperGeo1F1 + tMagSq_ScattLen*tMagSq_GTilde/(aRStarMag*aRStarMag) + 2.0*tTermFinal);
  return tResult;
/*
  cuDoubleComplex tResultComplex = tGamowFactor*( cuCabs(aHyperGeo1F1Cmplx)*cuCabs(aHyperGeo1F1Cmplx) + cuCabs(tScattLenCmplxConj)*cuCabs(tScattLenCmplxConj)*cuCabs(tGTildeCmplxConj)*cuCabs(tGTildeCmplxConj)/(aRStarMag*aRStarMag) + 2.*cuCreal(aExpTermCmplx*aHyperGeo1F1Cmplx*tScattLenCmplxConj*tGTildeCmplxConj/aRStarMag) );


  //TODO put in check to make sure there is no imaginary part
//  if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in ParellelWaveFunction::InterpolateWfSquared !!!!!" << endl;
//  assert(imag(tResultComplex) < std::numeric_limits< double >::min());

  return cuCreal(tResultComplex);
*/
}

//________________________________________________________________________________________________________________
__global__ void InterpolateWfSquared(double *aKStarMag, double *aRStarMag, double *aTheta, double aReF0, double aImF0, double aD0, double *aWfSquared)
{
  //TODO figure out which thread
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  double tGamow = GetGamowFactor(aKStarMag[idx]);
  cuDoubleComplex tExpTermCmplx = GetExpTerm(aKStarMag[idx],aRStarMag[idx],aTheta[idx]);

  cuDoubleComplex tGTildeCmplx, tHyperGeo1F1Cmplx, tScattLenCmplx;

  tGTildeCmplx = GTildeInterpolate(aKStarMag[idx],aRStarMag[idx]);
  tHyperGeo1F1Cmplx = HyperGeo1F1Interpolate(aKStarMag[idx],aRStarMag[idx],aTheta[idx]);
  tScattLenCmplx = ScattLenInterpolate(aReF0,aImF0,aD0,aKStarMag[idx]);

  double tResult = AssembleWfSquared(aRStarMag[idx],tGamow,tExpTermCmplx,tGTildeCmplx,tHyperGeo1F1Cmplx,tScattLenCmplx);

  aWfSquared[idx] = tResult;

}




//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


ParallelWaveFunction::ParallelWaveFunction(int aNThreadsPerBlock, int aNBlocks):
  fNThreadsPerBlock(aNThreadsPerBlock),
  fNBlocks(aNBlocks)
{
  cudaSetDeviceFlags(cudaDeviceMapHost);

}

//________________________________________________________________________________________________________________
ParallelWaveFunction::~ParallelWaveFunction()
{
  checkCudaErrors(cudaFree(d_fGTildeReal));
  checkCudaErrors(cudaFree(d_fGTildeImag));
  checkCudaErrors(cudaFree(d_fGTildeInfo));

  checkCudaErrors(cudaFree(d_fHyperGeo1F1Real));
  checkCudaErrors(cudaFree(d_fHyperGeo1F1Imag));
  checkCudaErrors(cudaFree(d_fHyperGeo1F1Info));

//  checkCudaErrors(cudaFree(d_fCoulombScatteringLengthReal));
//  checkCudaErrors(cudaFree(d_fCoulombScatteringLengthImag));
  checkCudaErrors(cudaFree(d_fScattLenInfo));
}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadGTildeReal(td2dVec &aGTildeReal)
{
  int tNbinsK = aGTildeReal.size();
  int tNbinsR = aGTildeReal[0].size();

  int tSize = tNbinsK*tNbinsR*sizeof(double);

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

}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadGTildeImag(td2dVec &aGTildeImag)
{
  int tNbinsK = aGTildeImag.size();
  int tNbinsR = aGTildeImag[0].size();

  int tSize = tNbinsK*tNbinsR*sizeof(double);

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

}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadHyperGeo1F1Real(td3dVec &aHyperGeo1F1Real)
{
  int tNbinsK = aHyperGeo1F1Real.size();
  int tNbinsR = aHyperGeo1F1Real[0].size();
  int tNbinsTheta = aHyperGeo1F1Real[0][0].size();

  int tSize = tNbinsK*tNbinsR*tNbinsTheta*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fHyperGeo1F1Real, tSize));

  int tIndex;
  for(int iK=0; iK<tNbinsK; iK++)
  {
    for(int iR=0; iR<tNbinsR; iR++)
    {
      for(int iTheta=0; iTheta<tNbinsTheta; iTheta++)
      {
        tIndex = iTheta + iR*tNbinsTheta + iK*tNbinsTheta*tNbinsR;
        d_fHyperGeo1F1Real[tIndex] = aHyperGeo1F1Real[iK][iR][iTheta];
      }
    }
  }

}

//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadHyperGeo1F1Imag(td3dVec &aHyperGeo1F1Imag)
{
  int tNbinsK = aHyperGeo1F1Imag.size();
  int tNbinsR = aHyperGeo1F1Imag[0].size();
  int tNbinsTheta = aHyperGeo1F1Imag[0][0].size();

  int tSize = tNbinsK*tNbinsR*tNbinsTheta*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fHyperGeo1F1Imag, tSize));

  int tIndex;
  for(int iK=0; iK<tNbinsK; iK++)
  {
    for(int iR=0; iR<tNbinsR; iR++)
    {
      for(int iTheta=0; iTheta<tNbinsTheta; iTheta++)
      {
        tIndex = iTheta + iR*tNbinsTheta + iK*tNbinsTheta*tNbinsR;
        d_fHyperGeo1F1Imag[tIndex] = aHyperGeo1F1Imag[iK][iR][iTheta];
      }
    }
  }

}



//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadScattLenReal(td4dVec &aScattLenReal)
{
  int tNbinsReF0 = aScattLenReal.size();
  int tNbinsImF0 = aScattLenReal[0].size();
  int tNbinsD0 = aScattLenReal[0][0].size();
  int tNbinsK = aScattLenReal[0][0][0].size();


  int tSize = tNbinsReF0*tNbinsImF0*tNbinsD0*tNbinsK*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fCoulombScatteringLengthReal, tSize));

  int tIndex;
  for(int iReF0=0; iReF0<tNbinsReF0; iReF0++)
  {
    for(int iImF0=0; iImF0<tNbinsImF0; iImF0++)
    {
      for(int iD0=0; iD0<tNbinsD0; iD0++)
      {
        for(int iK=0; iK<tNbinsK; iK++)
        {
          tIndex = iK + iD0*tNbinsK + iImF0*tNbinsK*tNbinsD0 + iReF0*tNbinsK*tNbinsD0*tNbinsImF0;
          d_fCoulombScatteringLengthReal[tIndex] = aScattLenReal[iReF0][iImF0][iD0][iK];
        }
      }
    }
  }

}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadScattLenImag(td4dVec &aScattLenImag)
{
  int tNbinsReF0 = aScattLenImag.size();
  int tNbinsImF0 = aScattLenImag[0].size();
  int tNbinsD0 = aScattLenImag[0][0].size();
  int tNbinsK = aScattLenImag[0][0][0].size();


  int tSize = tNbinsReF0*tNbinsImF0*tNbinsD0*tNbinsK*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fCoulombScatteringLengthImag, tSize));

  int tIndex;
  for(int iReF0=0; iReF0<tNbinsReF0; iReF0++)
  {
    for(int iImF0=0; iImF0<tNbinsImF0; iImF0++)
    {
      for(int iD0=0; iD0<tNbinsD0; iD0++)
      {
        for(int iK=0; iK<tNbinsK; iK++)
        {
          tIndex = iK + iD0*tNbinsK + iImF0*tNbinsK*tNbinsD0 + iReF0*tNbinsK*tNbinsD0*tNbinsImF0;
          d_fCoulombScatteringLengthImag[tIndex] = aScattLenImag[iReF0][iImF0][iD0][iK];
        }
      }
    }
  }

}

//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadScattLenRealSub(td4dVec &aScattLenReal)
{
  int tNbinsReF0 = aScattLenReal.size();
  int tNbinsImF0 = aScattLenReal[0].size();
  int tNbinsD0 = aScattLenReal[0][0].size();
  int tNbinsK = aScattLenReal[0][0][0].size();


  int tSize = tNbinsReF0*tNbinsImF0*tNbinsD0*tNbinsK*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fScattLenRealSubVec, tSize));

  int tIndex;
  for(int iReF0=0; iReF0<tNbinsReF0; iReF0++)
  {
    for(int iImF0=0; iImF0<tNbinsImF0; iImF0++)
    {
      for(int iD0=0; iD0<tNbinsD0; iD0++)
      {
        for(int iK=0; iK<tNbinsK; iK++)
        {
          tIndex = iK + iD0*tNbinsK + iImF0*tNbinsK*tNbinsD0 + iReF0*tNbinsK*tNbinsD0*tNbinsImF0;
          d_fScattLenRealSubVec[tIndex] = aScattLenReal[iReF0][iImF0][iD0][iK];
        }
      }
    }
  }

}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadScattLenImagSub(td4dVec &aScattLenImag)
{
  int tNbinsReF0 = aScattLenImag.size();
  int tNbinsImF0 = aScattLenImag[0].size();
  int tNbinsD0 = aScattLenImag[0][0].size();
  int tNbinsK = aScattLenImag[0][0][0].size();


  int tSize = tNbinsReF0*tNbinsImF0*tNbinsD0*tNbinsK*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fScattLenImagSubVec, tSize));

  int tIndex;
  for(int iReF0=0; iReF0<tNbinsReF0; iReF0++)
  {
    for(int iImF0=0; iImF0<tNbinsImF0; iImF0++)
    {
      for(int iD0=0; iD0<tNbinsD0; iD0++)
      {
        for(int iK=0; iK<tNbinsK; iK++)
        {
          tIndex = iK + iD0*tNbinsK + iImF0*tNbinsK*tNbinsD0 + iReF0*tNbinsK*tNbinsD0*tNbinsImF0;
          d_fScattLenImagSubVec[tIndex] = aScattLenImag[iReF0][iImF0][iD0][iK];
        }
      }
    }
  }

}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::UnLoadScattLenRealSub()
{
  checkCudaErrors(cudaFree(d_fScattLenRealSubVec));
}

//________________________________________________________________________________________________________________
void ParallelWaveFunction::UnLoadScattLenImagSub()
{
  checkCudaErrors(cudaFree(d_fScattLenImagSubVec));
}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadGTildeInfo(BinInfoGTilde &aBinInfo)
{
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

}

//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadHyperGeo1F1Info(BinInfoHyperGeo1F1 &aBinInfo)
{
  checkCudaErrors(cudaMallocManaged(&d_fHyperGeo1F1Info, sizeof(BinInfoHyperGeo1F1)));

  d_fHyperGeo1F1Info->nBinsK = aBinInfo.nBinsK; 
  d_fHyperGeo1F1Info->nBinsR = aBinInfo.nBinsR; 
  d_fHyperGeo1F1Info->nBinsTheta = aBinInfo.nBinsTheta; 

  d_fHyperGeo1F1Info->binWidthK = aBinInfo.binWidthK; 
  d_fHyperGeo1F1Info->binWidthR = aBinInfo.binWidthR; 
  d_fHyperGeo1F1Info->binWidthTheta = aBinInfo.binWidthTheta; 

  d_fHyperGeo1F1Info->minK = aBinInfo.minK; 
  d_fHyperGeo1F1Info->maxK = aBinInfo.maxK; 
  d_fHyperGeo1F1Info->minR = aBinInfo.minR; 
  d_fHyperGeo1F1Info->maxR = aBinInfo.maxR; 
  d_fHyperGeo1F1Info->minTheta = aBinInfo.minTheta; 
  d_fHyperGeo1F1Info->maxTheta = aBinInfo.maxTheta; 

  d_fHyperGeo1F1Info->minInterpK = aBinInfo.minInterpK; 
  d_fHyperGeo1F1Info->maxInterpK = aBinInfo.maxInterpK; 
  d_fHyperGeo1F1Info->minInterpR = aBinInfo.minInterpR; 
  d_fHyperGeo1F1Info->maxInterpR = aBinInfo.maxInterpR; 
  d_fHyperGeo1F1Info->minInterpTheta = aBinInfo.minInterpTheta; 
  d_fHyperGeo1F1Info->maxInterpTheta = aBinInfo.maxInterpTheta; 

}

//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadScattLenInfo(BinInfoScattLen &aBinInfo)
{
  checkCudaErrors(cudaMallocManaged(&d_fScattLenInfo, sizeof(BinInfoScattLen)));

  d_fScattLenInfo->nBinsReF0 = aBinInfo.nBinsReF0;
  d_fScattLenInfo->nBinsImF0 = aBinInfo.nBinsImF0;
  d_fScattLenInfo->nBinsD0 = aBinInfo.nBinsD0;
  d_fScattLenInfo->nBinsK = aBinInfo.nBinsK;

  d_fScattLenInfo->binWidthReF0 = aBinInfo.binWidthReF0;
  d_fScattLenInfo->binWidthImF0 = aBinInfo.binWidthImF0;
  d_fScattLenInfo->binWidthD0 = aBinInfo.binWidthD0;
  d_fScattLenInfo->binWidthK = aBinInfo.binWidthK;

  d_fScattLenInfo->minReF0 = aBinInfo.minReF0;
  d_fScattLenInfo->maxReF0 = aBinInfo.maxReF0;
  d_fScattLenInfo->minImF0 = aBinInfo.minImF0;
  d_fScattLenInfo->maxImF0 = aBinInfo.maxImF0;
  d_fScattLenInfo->minD0 = aBinInfo.minD0;
  d_fScattLenInfo->maxD0 = aBinInfo.maxD0;
  d_fScattLenInfo->minK = aBinInfo.minK;
  d_fScattLenInfo->maxK = aBinInfo.maxK;



  d_fScattLenInfo->minInterpReF0 = aBinInfo.minInterpReF0;
  d_fScattLenInfo->maxInterpReF0 = aBinInfo.maxInterpReF0;
  d_fScattLenInfo->minInterpImF0 = aBinInfo.minInterpImF0;
  d_fScattLenInfo->maxInterpImF0 = aBinInfo.maxInterpImF0;
  d_fScattLenInfo->minInterpD0 = aBinInfo.minInterpD0;
  d_fScattLenInfo->maxInterpD0 = aBinInfo.maxInterpD0;
  d_fScattLenInfo->minInterpK = aBinInfo.minInterpK;
  d_fScattLenInfo->maxInterpK = aBinInfo.maxInterpK;

}

//________________________________________________________________________________________________________________
//double* ParallelWaveFunction::RunInterpolateWfSquared(td2dVec &aPairs, double aReF0, double aImF0, double aD0)
vector<double> ParallelWaveFunction::RunInterpolateWfSquared(td2dVec &aPairs, double aReF0, double aImF0, double aD0)
{
  int tNPairs = aPairs.size();
  int tSize = tNPairs*sizeof(double);

  //---Host arrays and allocations
  double * h_KStarMag;
  double * h_RStarMag;
  double * h_Theta;
  double * h_WfSquared;
/*
  checkCudaErrors(cudaHostAlloc((void**) &h_KStarMag, tSize, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &h_RStarMag, tSize, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &h_Theta, tSize, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &h_WfSquared, tSize, cudaHostAllocMapped));
*/

  checkCudaErrors(cudaMallocManaged(&h_KStarMag, tSize));
  checkCudaErrors(cudaMallocManaged(&h_RStarMag, tSize));
  checkCudaErrors(cudaMallocManaged(&h_Theta, tSize));
  checkCudaErrors(cudaMallocManaged(&h_WfSquared, tSize));

  for(int i=0; i<tNPairs; i++)
  {
    h_KStarMag[i] = aPairs[i][0];
    h_RStarMag[i] = aPairs[i][1];
    h_Theta[i] = aPairs[i][2];
  }

/*
  //---Device arrays and allocations
  double * d_KStarMag;
  double * d_RStarMag;
  double * d_Theta;
  double * d_WfSquared;

  checkCudaErrors(cudaHostGetDevicePointer(&d_KStarMag, h_KStarMag, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&d_RStarMag, h_RStarMag, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&d_Theta, h_Theta, 0));
  checkCudaErrors(cudaHostGetDevicePointer(&d_WfSquared, h_WfSquared, 0));
*/
  //----------Run the kernel-----------------------------------------------
  GpuTimer timer;
  timer.Start();
  InterpolateWfSquared<<<fNBlocks,fNThreadsPerBlock>>>(h_KStarMag,h_RStarMag,h_Theta,aReF0,aImF0,aD0,h_WfSquared);
  timer.Stop();
  std::cout << "InterpolateWfSquared kernel finished in " << timer.Elapsed() << " ms" << std::endl;

  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());

/*
  checkCudaErrors(cudaFreeHost(h_KStarMag));
  checkCudaErrors(cudaFreeHost(h_RStarMag));
  checkCudaErrors(cudaFreeHost(h_Theta));
*/
  checkCudaErrors(cudaFree(h_KStarMag));
  checkCudaErrors(cudaFree(h_RStarMag));
  checkCudaErrors(cudaFree(h_Theta));

//  return h_WfSquared;
  vector<double> tReturnVec(tNPairs);
  for(int i=0; i<tNPairs; i++) 
  {
    tReturnVec[i] = h_WfSquared[i];
//    cout << "i = " << i << endl;
//    cout << "h_WfSquared[i] = " << h_WfSquared[i] << endl;
//    cout << "tReturnVec[i] = " << tReturnVec[i] << endl << endl;
  }

//  checkCudaErrors(cudaFreeHost(h_WfSquared));
  checkCudaErrors(cudaFree(h_WfSquared));

  return tReturnVec;
}

