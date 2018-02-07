///////////////////////////////////////////////////////////////////////////
// ParallelWaveFunction:                                                 //
///////////////////////////////////////////////////////////////////////////

#include "ParallelWaveFunction.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> texRefA;
const textureReference* texRefAPtr;

texture<float, cudaTextureType2D, cudaReadModeElementType> texRefB;
const textureReference* texRefBPtr;

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
  }

  //Check error
  if(tErrorFlag) assert(0);

  //---------------------------------
  tBin = GetBinNumber(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) assert(0);

  return tReturnBin;
}

//________________________________________________________________________________________________________________
__device__ double LednickyHFunctionInterpolate(double aKStar)
{
  double tResult = 0.0;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBin does not return the error -2
  //TODO make HFunctionInfo objects instead of using GTilde
  //TODO check accuracy

  double tBinWidthK = d_fGTildeInfo->binWidthK;
  int tBinLowK = GetInterpLowBin(kGTilde,kKaxis,aKStar);
  int tBinHighK = tBinLowK+1;
  double tBinLowCenterK = d_fGTildeInfo->minK + (tBinLowK+0.5)*d_fGTildeInfo->binWidthK;
  double tBinHighCenterK = tBinLowCenterK+tBinWidthK;

  double tX0 = tBinLowCenterK;
  double tX1 = tBinHighCenterK;
  double tY0 = d_fLednickyHFunction[tBinLowK];
  double tY1 = d_fLednickyHFunction[tBinHighK];

  tResult = tY0 + (aKStar-tX0)*((tY1-tY0)/(tX1-tX0));
  return tResult;
}



//________________________________________________________________________________________________________________
__device__ cuDoubleComplex GTildeInterpolate(double aKStar, double aRStar)
{
  double tResultReal = 0.;
  double tResultImag = 0.;
  //----------------------------

  int tNbinsR = d_fGTildeInfo->nBinsR;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBin does not return the error -2
  double tBinWidthK = d_fGTildeInfo->binWidthK;
  int tBinLowK = GetInterpLowBin(kGTilde,kKaxis,aKStar);
  int tBinHighK = tBinLowK+1;
  double tBinLowCenterK = d_fGTildeInfo->minK + (tBinLowK+0.5)*d_fGTildeInfo->binWidthK;
  double tBinHighCenterK = tBinLowCenterK+tBinWidthK;

  double tBinWidthR = d_fGTildeInfo->binWidthR;
  int tBinLowR = GetInterpLowBin(kGTilde,kRaxis,aRStar);
  int tBinHighR = tBinLowR+1;
  double tBinLowCenterR = d_fGTildeInfo->minR + (tBinLowR+0.5)*d_fGTildeInfo->binWidthR;
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

/*
//________________________________________________________________________________________________________________
__device__ cuDoubleComplex GTildeInterpolate(double aKStar, double aRStar)
{
  double tResultReal = 0.;
  double tResultImag = 0.;
  //----------------------------

  int tNbinsR = d_fGTildeInfo->nBinsR;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBin does not return the error -2
  double tBinWidthK = d_fGTildeInfo->binWidthK;
  int tBinLowK = GetInterpLowBin(kGTilde,kKaxis,aKStar);
  int tBinHighK = tBinLowK+1;
  double tBinLowCenterK = d_fGTildeInfo->minK + (tBinLowK+0.5)*d_fGTildeInfo->binWidthK;
  double tBinHighCenterK = tBinLowCenterK+tBinWidthK;

  double tBinWidthR = d_fGTildeInfo->binWidthR;
  int tBinLowR = GetInterpLowBin(kGTilde,kRaxis,aRStar);
  int tBinHighR = tBinLowR+1;
  double tBinLowCenterR = d_fGTildeInfo->minR + (tBinLowR+0.5)*d_fGTildeInfo->binWidthR;
  double tBinHighCenterR = tBinLowCenterR+tBinWidthR;

  //--------------------------

  double tQ11Real = d_fGTildeReal[tBinLowR + tBinLowK*tNbinsR];
  double tQ12Real = d_fGTildeReal[tBinHighR + tBinLowK*tNbinsR];
  double tQ21Real = d_fGTildeReal[tBinLowR + tBinHighK*tNbinsR];
  double tQ22Real = d_fGTildeReal[tBinHighR + tBinHighK*tNbinsR];

//  float tQ11Realv2 = tex2D(texRefA, tBinLowR, tBinLowK);
//  float tQ12Realv2 = tex2D(texRefA, tBinHighR, tBinLowK);
//  float tQ21Realv2 = tex2D(texRefA, tBinLowR, tBinHighK);
//  float tQ22Realv2 = tex2D(texRefA, tBinHighR, tBinHighK);

  double tQ11Imag = d_fGTildeImag[tBinLowR + tBinLowK*tNbinsR];
  double tQ12Imag = d_fGTildeImag[tBinHighR + tBinLowK*tNbinsR];
  double tQ21Imag = d_fGTildeImag[tBinLowR + tBinHighK*tNbinsR];
  double tQ22Imag = d_fGTildeImag[tBinHighR + tBinHighK*tNbinsR];

//--------------------------

  double tD = 1.0*tBinWidthK*tBinWidthR;

  tResultReal = (1.0/tD)*(tQ11Real*(tBinHighCenterK-aKStar)*(tBinHighCenterR-aRStar) + tQ21Real*(aKStar-tBinLowCenterK)*(tBinHighCenterR-aRStar) + tQ12Real*(tBinHighCenterK-aKStar)*(aRStar-tBinLowCenterR) + tQ22Real*(aKStar-tBinLowCenterK)*(aRStar-tBinLowCenterR));

//float tResultReal2 = (1.0/tD)*(tQ11Realv2*(tBinHighCenterK-aKStar)*(tBinHighCenterR-aRStar) + tQ21Realv2*(aKStar-tBinLowCenterK)*(tBinHighCenterR-aRStar) + tQ12Realv2*(tBinHighCenterK-aKStar)*(aRStar-tBinLowCenterR) + tQ22Realv2*(aKStar-tBinLowCenterK)*(aRStar-tBinLowCenterR));
//printf("tResultReal = %0.4f \t tResultReal2 = %0.4f \n", tResultReal, tResultReal2);


  float tDiffK2 = aKStar - (d_fGTildeInfo->minK + tBinLowK*d_fGTildeInfo->binWidthK);
  float tNormDiffK2 = tBinLowK + tDiffK2/tBinLowK*d_fGTildeInfo->binWidthK;

  float tDiffR2 = aRStar - (d_fGTildeInfo->minR + tBinLowR*d_fGTildeInfo->binWidthR);
  float tNormDiffR2 = tBinLowR + tDiffR2/tBinLowR*d_fGTildeInfo->binWidthR;

  float tResultReal2 = tex2D(texRefA, tNormDiffR2, tNormDiffK2);
  //-------------------
  float tDiffK3 = aKStar - (d_fGTildeInfo->minK + (tBinLowK+0.5)*d_fGTildeInfo->binWidthK);
  float tNormDiffK3 = tBinLowK + tDiffK3/tBinLowK*d_fGTildeInfo->binWidthK;

  float tDiffR3 = aRStar - (d_fGTildeInfo->minR + (tBinLowR+0.5)*d_fGTildeInfo->binWidthR);
  float tNormDiffR3 = tBinLowR + tDiffR3/tBinLowR*d_fGTildeInfo->binWidthR;

  float tResultReal3 = tex2D(texRefA, tNormDiffR3+0.5, tNormDiffK3+0.5);
  //----------------------
printf("tResultReal = %0.4f \t tResultReal2 = %0.4f \t tResultReal3 = %0.4f \n", tResultReal, tResultReal2, tResultReal3);

  tResultImag = (1.0/tD)*(tQ11Imag*(tBinHighCenterK-aKStar)*(tBinHighCenterR-aRStar) + tQ21Imag*(aKStar-tBinLowCenterK)*(tBinHighCenterR-aRStar) + tQ12Imag*(tBinHighCenterK-aKStar)*(aRStar-tBinLowCenterR) + tQ22Imag*(aKStar-tBinLowCenterK)*(aRStar-tBinLowCenterR));


//--------------------------
  cuDoubleComplex tReturnValue = make_cuDoubleComplex(tResultReal,tResultImag);
  return tReturnValue;
}
*/
//________________________________________________________________________________________________________________
__device__ cuDoubleComplex GTildeInterpolateTexture(double aKStar, double aRStar)
{
  float tResultReal = 0.;
  float tResultImag = 0.;
  //----------------------------

  int tNbinsR = d_fGTildeInfo->nBinsR;
  //----------------------------

  //TODO put in check to make sure GetInterpLowBin does not return the error -2
  double tBinWidthK = d_fGTildeInfo->binWidthK;
  int tBinLowK = GetInterpLowBin(kGTilde,kKaxis,aKStar);
  int tBinHighK = tBinLowK+1;
  double tBinLowCenterK = d_fGTildeInfo->minK + (tBinLowK+0.5)*d_fGTildeInfo->binWidthK;
  double tBinHighCenterK = tBinLowCenterK+tBinWidthK;

  double tBinWidthR = d_fGTildeInfo->binWidthR;
  int tBinLowR = GetInterpLowBin(kGTilde,kRaxis,aRStar);
  int tBinHighR = tBinLowR+1;
  double tBinLowCenterR = d_fGTildeInfo->minR + (tBinLowR+0.5)*d_fGTildeInfo->binWidthR;
  double tBinHighCenterR = tBinLowCenterR+tBinWidthR;

  //-------------------

  float tDiffK = aKStar - (d_fGTildeInfo->minK + (tBinLowK+0.5)*d_fGTildeInfo->binWidthK);
  float tNormDiffK = tBinLowK + tDiffK/tBinLowK*d_fGTildeInfo->binWidthK;

  float tDiffR = aRStar - (d_fGTildeInfo->minR + (tBinLowR+0.5)*d_fGTildeInfo->binWidthR);
  float tNormDiffR = tBinLowR + tDiffR/tBinLowR*d_fGTildeInfo->binWidthR;

  tResultReal = tex2D(texRefA, tNormDiffR+0.5, tNormDiffK+0.5);
  tResultImag = tex2D(texRefB, tNormDiffR+0.5, tNormDiffK+0.5);
  //----------------------

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

  //TODO put in check to make sure GetInterpLowBin does not return the error -2
  double tBinWidthK = d_fHyperGeo1F1Info->binWidthK;
  int tBin0K = GetInterpLowBin(kHyperGeo1F1,kKaxis,aKStar);
  int tBin1K = tBin0K+1;
  double tBin0CenterK = d_fHyperGeo1F1Info->minK + (tBin0K+0.5)*d_fHyperGeo1F1Info->binWidthK;
//  double tBin1CenterK = tBin0CenterK+tBinWidthK;

  double tBinWidthR = d_fHyperGeo1F1Info->binWidthR;
  int tBin0R = GetInterpLowBin(kHyperGeo1F1,kRaxis,aRStar);
  int tBin1R = tBin0R+1;
  double tBin0CenterR = d_fHyperGeo1F1Info->minR + (tBin0R+0.5)*d_fHyperGeo1F1Info->binWidthR;
//  double tBin1CenterR = tBin0CenterR+tBinWidthR;

  double tBinWidthTheta = d_fHyperGeo1F1Info->binWidthTheta;
  int tBin0Theta = GetInterpLowBin(kHyperGeo1F1,kThetaaxis,aTheta);
  int tBin1Theta = tBin0Theta+1;
  double tBin0CenterTheta = d_fHyperGeo1F1Info->minTheta + (tBin0Theta+0.5)*d_fHyperGeo1F1Info->binWidthTheta;
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

//________________________________________________________________________________________________________________
__device__ double GetEta(double aKStar)
{
  double d_hbarc = 0.197327;

  //TODO figure out how to use Pi here
  //TODO figure out how to make bohr radius negative when needed

  double tEta = pow(((aKStar/d_hbarc)*d_fBohrRadius),-1);
  return tEta;
}


//________________________________________________________________________________________________________________
__device__ double GetGamowFactor(double aKStar)
{
  double d_hbarc = 0.197327;

  //TODO figure out how to use Pi here
  //TODO figure out how to make bohr radius negative when needed

  double tEta = pow(((aKStar/d_hbarc)*d_fBohrRadius),-1);
  tEta *= 6.28318530718;  //eta always comes with 2Pi here
  double tGamow = tEta*pow((exp(tEta)-1),-1);

  return tGamow;
}

//________________________________________________________________________________________________________________
__device__ cuDoubleComplex GetExpTerm(double aKStar, double aRStar, double aTheta)
{
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
__device__ cuDoubleComplex BuildScatteringLength(double aKStarMag, double aReF0, double aImF0, double aD0)
{
  double d_hbarc = 0.197327;

  cuDoubleComplex tRealUnity = make_cuDoubleComplex(1.0,0);
  cuDoubleComplex tF0 = make_cuDoubleComplex(aReF0, aImF0);
  cuDoubleComplex tInvF0 = cuCdiv(tRealUnity,tF0);

  cuDoubleComplex tScattLenCmplx;
  if(aReF0==0.0 && aImF0==0.0 && aD0==0.0) tScattLenCmplx = make_cuDoubleComplex(0.0,0.0);
//TODO
/*
  else if(fTurnOffCoulomb)
  {
    double tKStar = aKStarMag/d_hbarc;
    double tTerm2 = 0.5*aD0*tKStar*tKStar;
    cuDoubleComplex tTerm2Complex = make_cuDoubleComplex(tTerm2,0);
    cuDoubleComplex tTerm3Complex = make_cuDoubleComplex(0.0, tKStar);

    cuDoubleComplex tTerm12 = cuCadd(tInvF0,tTerm2Complex);
    cuDoubleComplex tInvScattLen = cuCsub(tTerm12,tTerm3Complex);

    tScattLenCmplx = cuCdiv(tRealUnity,tInvScattLen);
  }
*/
  else
  {


    double tGamow = GetGamowFactor(aKStarMag);  
    double tLednickyHFunction = LednickyHFunctionInterpolate(aKStarMag);
    double tImag = tGamow/(2.0*GetEta(aKStarMag));
    cuDoubleComplex tLednickyChi = make_cuDoubleComplex(tLednickyHFunction,tImag);

    double tKStar = aKStarMag/d_hbarc;
    double tTerm2 = 0.5*aD0*tKStar*tKStar;
    cuDoubleComplex tTerm2Complex = make_cuDoubleComplex(tTerm2,0);

    double tStupid = 2.0/d_fBohrRadius;
    cuDoubleComplex tMultFact = make_cuDoubleComplex(tStupid, 0);
    cuDoubleComplex tTerm3Complex = cuCmul(tMultFact,tLednickyChi);

    cuDoubleComplex tTerm12 = cuCadd(tInvF0,tTerm2Complex);
    cuDoubleComplex tInvScattLen = cuCsub(tTerm12,tTerm3Complex);

    tScattLenCmplx = cuCdiv(tRealUnity,tInvScattLen);
  }
  return tScattLenCmplx;
}

//________________________________________________________________________________________________________________
__device__ double InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
{
  double tGamow = GetGamowFactor(aKStarMag);
  cuDoubleComplex tExpTermCmplx = GetExpTerm(aKStarMag,aRStarMag,aTheta);

  cuDoubleComplex tGTildeCmplx, tHyperGeo1F1Cmplx, tScattLenCmplx;

  tGTildeCmplx = GTildeInterpolate(aKStarMag,aRStarMag);
  tHyperGeo1F1Cmplx = HyperGeo1F1Interpolate(aKStarMag,aRStarMag,aTheta);

  //---Build scatt len
  tScattLenCmplx = BuildScatteringLength(aKStarMag, aReF0, aImF0, aD0);

  //--------------------------

  double tResult = AssembleWfSquared(aRStarMag,tGamow,tExpTermCmplx,tGTildeCmplx,tHyperGeo1F1Cmplx,tScattLenCmplx);

  return tResult;

}

//________________________________________________________________________________________________________________
__device__ bool CanInterpPair(double aKStar, double aRStar, double aTheta)
{
  if(aKStar < d_fGTildeInfo->minInterpK || aKStar > d_fGTildeInfo->maxInterpK) return false;
  if(aRStar < d_fGTildeInfo->minInterpR || aRStar > d_fGTildeInfo->maxInterpR) return false;
  if(aTheta < d_fHyperGeo1F1Info->minInterpTheta || aTheta > d_fHyperGeo1F1Info->maxInterpTheta) return false;
  return true;
}
 

//________________________________________________________________________________________________________________
__global__ void GetWfAverage(double *aKStarMag, double *aRStarMag, double *aTheta, double aReF0, double aImF0, double aD0, double *g_odata)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ double sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[tid] = InterpolateWfSquared(aKStarMag[i],aRStarMag[i],aTheta[i],aReF0,aImF0,aD0);
  __syncthreads();

  //do reduction in shared mem
  //strided
  for(unsigned int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*tid;

    if(index < blockDim.x)
    {
      sdata[index] += sdata[index+s];
    }
    __syncthreads();
  }
/*
  //sequential
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) //>>= is bitwise shift, here reducing s in powers of 2
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
*/
  //write result for this block to global mem
  if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}


//________________________________________________________________________________________________________________
__global__ void GetEntireCf(double *aKStarMag, double *aRStarMag, double *aTheta, double aReF0, double aImF0, double aD0, double *g_odata, int aOffsetInput, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ double sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x + aOffsetInput;

  sdata[tid] = InterpolateWfSquared(aKStarMag[i],aRStarMag[i],aTheta[i],aReF0,aImF0,aD0);
  __syncthreads();

  //do reduction in shared mem
  //strided
  for(unsigned int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*tid;

    if(index < blockDim.x)
    {
      sdata[index] += sdata[index+s];
    }
    __syncthreads();
  }
/*
  //sequential
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) //>>= is bitwise shift, here reducing s in powers of 2
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
*/
  //write result for this block to global mem
  if(tid == 0) g_odata[blockIdx.x+aOffsetOutput] = sdata[0];
}

//________________________________________________________________________________________________________________
__global__ void GetEntireCfComplete(double *aKStarMag, double *aRStarMag, double *aTheta, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double *g_odata, int aOffsetInput, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ double sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x + aOffsetInput;

  double tWfSqSinglet, tWfSqTriplet, tWfSq;

  tWfSqSinglet = InterpolateWfSquared(aKStarMag[i],aRStarMag[i],aTheta[i],aReF0s,aImF0s,aD0s);
  tWfSqTriplet = InterpolateWfSquared(aKStarMag[i],aRStarMag[i],aTheta[i],aReF0t,aImF0t,aD0t);

  tWfSq = 0.25*tWfSqSinglet + 0.75*tWfSqTriplet;
  sdata[tid] = tWfSq;

  __syncthreads();

  //do reduction in shared mem
  //strided
  for(unsigned int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*tid;

    if(index < blockDim.x)
    {
      sdata[index] += sdata[index+s];
    }
    __syncthreads();
  }
/*
  //sequential
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) //>>= is bitwise shift, here reducing s in powers of 2
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
*/
  //write result for this block to global mem
  if(tid == 0) g_odata[blockIdx.x+aOffsetOutput] = sdata[0];
}

//________________________________________________________________________________________________________________
__device__ int GetSamplePairOffset(int aAnalysis, int aBinK, int aPair)
{
  int tNBinsK = d_fPairSample4dVecInfo->nBinsK;
  int tNPairsPerBin = d_fPairSample4dVecInfo->nPairsPerBin;
  int tNElementsPerPair = d_fPairSample4dVecInfo->nElementsPerPair;

  int tIndex = aPair*tNElementsPerPair + aBinK*tNPairsPerBin*tNElementsPerPair + aAnalysis*tNBinsK*tNPairsPerBin*tNElementsPerPair;
  return tIndex;
}

//________________________________________________________________________________________________________________
__global__ void GetEntireCfwStaticPairs(double aRadiusScale, double aReF0, double aImF0, double aD0, double *g_odata, double *g_odata2, int aAnalysisNumber, int aBinKNumber, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ double sdata2[][2];

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);

  if(CanInterpPair(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2]))
  {
    double tWfSq = InterpolateWfSquared(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2], aReF0, aImF0, aD0);
    sdata2[tid][0] = tWfSq;
    sdata2[tid][1] = 1.;
  }

  else
  {
    sdata2[tid][0] = 0.;
    sdata2[tid][1] = 0.;
  }

  __syncthreads();

  //do reduction in shared mem
  //strided
  for(unsigned int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*tid;

    if(index < blockDim.x)
    {
      sdata2[index][0] += sdata2[index+s][0];
      sdata2[index][1] += sdata2[index+s][1];
    }
    __syncthreads();
  }
/*
  //sequential
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) //>>= is bitwise shift, here reducing s in powers of 2
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
*/
  //write result for this block to global mem
  if(tid == 0) 
  {
    g_odata[blockIdx.x+aOffsetOutput] = sdata2[0][0];
    g_odata2[blockIdx.x+aOffsetOutput] = sdata2[0][1];
  }
}


//________________________________________________________________________________________________________________
__global__ void GetEntireCfCompletewStaticPairs(double aRadiusScale, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double *g_odata, double *g_odata2, int aAnalysisNumber, int aBinKNumber, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ double sdata2[][2];

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);

  if(CanInterpPair(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2]))
  {
    double tWfSqSinglet = InterpolateWfSquared(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2], aReF0s, aImF0s, aD0s);
    double tWfSqTriplet = InterpolateWfSquared(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2], aReF0t, aImF0t, aD0t);

    double tWfSq = 0.25*tWfSqSinglet + 0.75*tWfSqTriplet;
    sdata2[tid][0] = tWfSq;
    sdata2[tid][1] = 1.;
  }

  else
  {
    sdata2[tid][0] = 0.;
    sdata2[tid][1] = 0.;
  }

  __syncthreads();

  //do reduction in shared mem
  //strided
  for(unsigned int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*tid;

    if(index < blockDim.x)
    {
      sdata2[index][0] += sdata2[index+s][0];
      sdata2[index][1] += sdata2[index+s][1];
    }
    __syncthreads();
  }
/*
  //sequential
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) //>>= is bitwise shift, here reducing s in powers of 2
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
*/
  //write result for this block to global mem
  if(tid == 0) 
  {
    g_odata[blockIdx.x+aOffsetOutput] = sdata2[0][0];
    g_odata2[blockIdx.x+aOffsetOutput] = sdata2[0][1];
  }
}


//________________________________________________________________________________________________________________
__global__ void RandInit(curandState *state, unsigned long seed, int aOffset)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x + aOffset;
  curand_init(seed, idx, 0, &state[idx]);
}


//________________________________________________________________________________________________________________
__global__ void GetEntireCfComplete2(curandState *state1, curandState *state2, curandState *state3, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double *g_odata, int aKbin, int aOffsetInput, int aOffsetOutput, double* aCPUPairs)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ double sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x + aOffsetInput;

  bool tPass = false;
  int tNPairs = d_fPairKStar3dVecInfo->nPairsPerBin[aKbin];
  int tSimPairLocalIndex, tSimPairGlobalIndex;
  double tKStarOut, tKStarSide, tKStarLong, tKStarMagSq, tKStarMag;
  double tRStarOut, tRStarSide, tRStarLong, tRStarMagSq, tRStarMag;
  double tCosTheta, tTheta;

  //TODO need to be able to return all failing pairs back to CPU for mathematica processing

  while(!tPass)
  {
    tSimPairLocalIndex = tNPairs*curand_uniform_double(&state1[i]);
    tSimPairGlobalIndex = d_fPairKStar3dVecInfo->binOffset[aKbin] + 4*tSimPairLocalIndex;
    tKStarOut = d_fPairKStar3dVec[tSimPairGlobalIndex+1]; //note, 0th element is KStarMag
    tKStarSide = d_fPairKStar3dVec[tSimPairGlobalIndex+2]; //note, 0th element is KStarMag
    tKStarLong = d_fPairKStar3dVec[tSimPairGlobalIndex+3]; //note, 0th element is KStarMag
    tKStarMagSq = tKStarOut*tKStarOut + tKStarSide*tKStarSide + tKStarLong*tKStarLong;
    tKStarMag = sqrt(tKStarMagSq);

    tRStarOut = aR*curand_normal_double(&state1[i]);
    tRStarSide = aR*curand_normal_double(&state2[i]);
    tRStarLong = aR*curand_normal_double(&state3[i]);
    tRStarMagSq = tRStarOut*tRStarOut + tRStarSide*tRStarSide + tRStarLong*tRStarLong;
    tRStarMag = sqrt(tRStarMagSq);

    tCosTheta = (tKStarOut*tRStarOut + tKStarSide*tRStarSide + tKStarLong*tRStarLong)/(tKStarMag*tRStarMag);
    tTheta = acos(tCosTheta);

    tPass = CanInterpPair(tKStarMag,tRStarMag,tTheta);
  }

  double tWfSqSinglet, tWfSqTriplet, tWfSq;

  tWfSqSinglet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,aReF0s,aImF0s,aD0s);
  tWfSqTriplet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,aReF0t,aImF0t,aD0t);

  tWfSq = 0.25*tWfSqSinglet + 0.75*tWfSqTriplet;
  sdata[tid] = tWfSq;
  __syncthreads();

  //do reduction in shared mem
  //strided
  for(unsigned int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*tid;

    if(index < blockDim.x)
    {
      sdata[index] += sdata[index+s];
    }
    __syncthreads();
  }
/*
  //sequential
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) //>>= is bitwise shift, here reducing s in powers of 2
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
*/
  //write result for this block to global mem
  if(tid == 0) g_odata[blockIdx.x+aOffsetOutput] = sdata[0];
}














//________________________________________________________________________________________________________________
__global__ void GetAllGamowFactors(double *g_odata, int aAnalysisNumber, int aBinKNumber, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);

  double tGamow = GetGamowFactor(d_fPairSample4dVec[i]);
  g_odata[tPairNumber+aOffsetOutput] = tGamow;
}

//________________________________________________________________________________________________________________
__global__ void GetAllExpTermsCmplx(double aRadiusScale, double *g_odataReal, double *g_odataImag, int aAnalysisNumber, int aBinKNumber, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);

  cuDoubleComplex tExpTermCmplx = GetExpTerm(d_fPairSample4dVec[i],aRadiusScale*d_fPairSample4dVec[i+1],d_fPairSample4dVec[i+2]);

  g_odataReal[tPairNumber+aOffsetOutput] = cuCreal(tExpTermCmplx);
  g_odataImag[tPairNumber+aOffsetOutput] = cuCimag(tExpTermCmplx);
}

//________________________________________________________________________________________________________________
__global__ void GetAllGTildeCmplx(double aRadiusScale, double *g_odataReal, double *g_odataImag, int aAnalysisNumber, int aBinKNumber, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);

  if(CanInterpPair(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2]))
  {
    cuDoubleComplex tGTildeCmplx = GTildeInterpolate(d_fPairSample4dVec[i],aRadiusScale*d_fPairSample4dVec[i+1]);

    g_odataReal[tPairNumber+aOffsetOutput] = cuCreal(tGTildeCmplx);
    g_odataImag[tPairNumber+aOffsetOutput] = cuCimag(tGTildeCmplx);
  }
  else
  {
    g_odataReal[tPairNumber+aOffsetOutput] = 0.;
    g_odataImag[tPairNumber+aOffsetOutput] = 0.;
  }

}

//________________________________________________________________________________________________________________
__global__ void GetAllGTildeCmplxTexture(double aRadiusScale, double *g_odataReal, double *g_odataImag, int aAnalysisNumber, int aBinKNumber, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);

  if(CanInterpPair(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2]))
  {
    cuDoubleComplex tGTildeCmplx = GTildeInterpolateTexture(d_fPairSample4dVec[i],aRadiusScale*d_fPairSample4dVec[i+1]);

    g_odataReal[tPairNumber+aOffsetOutput] = cuCreal(tGTildeCmplx);
    g_odataImag[tPairNumber+aOffsetOutput] = cuCimag(tGTildeCmplx);
  }
  else
  {
    g_odataReal[tPairNumber+aOffsetOutput] = 0.;
    g_odataImag[tPairNumber+aOffsetOutput] = 0.;
  }

}

//________________________________________________________________________________________________________________
__global__ void GetAllHyperGeo1F1Cmplx(double aRadiusScale, double *g_odataReal, double *g_odataImag, int aAnalysisNumber, int aBinKNumber, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);

  if(CanInterpPair(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2]))
  {
    cuDoubleComplex tHyperGeo1F1Cmplx = HyperGeo1F1Interpolate(d_fPairSample4dVec[i],aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2]);

    g_odataReal[tPairNumber+aOffsetOutput] = cuCreal(tHyperGeo1F1Cmplx);
    g_odataImag[tPairNumber+aOffsetOutput] = cuCimag(tHyperGeo1F1Cmplx);
  }
  else
  {
    g_odataReal[tPairNumber+aOffsetOutput] = 0.;
    g_odataImag[tPairNumber+aOffsetOutput] = 0.;
  }
}


//________________________________________________________________________________________________________________
__global__ void GetAllScattLenCmplx(double aRadiusScale, double aReF0, double aImF0, double aD0, double *g_odataReal, double *g_odataImag, int aAnalysisNumber, int aBinKNumber, int aOffsetOutput)
{
//  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);

  if(CanInterpPair(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2]))
  {
    cuDoubleComplex tScattLenCmplx = BuildScatteringLength(d_fPairSample4dVec[i], aReF0, aImF0, aD0);

    g_odataReal[tPairNumber+aOffsetOutput] = cuCreal(tScattLenCmplx);
    g_odataImag[tPairNumber+aOffsetOutput] = cuCimag(tScattLenCmplx);
  }
  else
  {
    g_odataReal[tPairNumber+aOffsetOutput] = 0.;
    g_odataImag[tPairNumber+aOffsetOutput] = 0.;
  }
}

//________________________________________________________________________________________________________________
__global__ void InterpolateAllWfSquared(double aRadiusScale, double* g_idataGamow, double* g_idataExpTermReal, double* g_idataExpTermImag, double* g_idataGTildeReal, double* g_idataGTildeImag, double* g_idataHyperGeo1F1Real, double* g_idataHyperGeo1F1Complex, double* g_idataScattLenReal, double* g_idataScattLenImag, double *g_odata, double *g_odata2, int aAnalysisNumber, int aBinKNumber, int tOffsetInput, int aOffsetOutput)
{
  extern __shared__ double sdata2[][2];

  unsigned int tid = threadIdx.x;
  unsigned int tPairNumber = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int i = GetSamplePairOffset(aAnalysisNumber,aBinKNumber,tPairNumber);
  tPairNumber += tOffsetInput;
  if(CanInterpPair(d_fPairSample4dVec[i], aRadiusScale*d_fPairSample4dVec[i+1], d_fPairSample4dVec[i+2]))
  {
    double tGamow = g_idataGamow[tPairNumber];

    cuDoubleComplex tExpTermCmplx = make_cuDoubleComplex(g_idataExpTermReal[tPairNumber], g_idataExpTermImag[tPairNumber]);

    cuDoubleComplex tGTildeCmplx  = make_cuDoubleComplex(g_idataGTildeReal[tPairNumber], g_idataGTildeImag[tPairNumber]);

    cuDoubleComplex tHyperGeo1F1Cmplx  = make_cuDoubleComplex(g_idataHyperGeo1F1Real[tPairNumber], g_idataHyperGeo1F1Complex[tPairNumber]);
    cuDoubleComplex tScattLenCmplx  = make_cuDoubleComplex(g_idataScattLenReal[tPairNumber], g_idataScattLenImag[tPairNumber]);

    //--------------------------

    double tResult = AssembleWfSquared(aRadiusScale*d_fPairSample4dVec[i+1],tGamow,tExpTermCmplx,tGTildeCmplx,tHyperGeo1F1Cmplx,tScattLenCmplx);

    //--------------------------

    sdata2[tid][0] = tResult;
    sdata2[tid][1] = 1.;
  }

  else
  {
    sdata2[tid][0] = 0.;
    sdata2[tid][1] = 0.;
  }

  __syncthreads();

  //do reduction in shared mem
  //strided
  for(unsigned int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*tid;

    if(index < blockDim.x)
    {
      sdata2[index][0] += sdata2[index+s][0];
      sdata2[index][1] += sdata2[index+s][1];
    }
    __syncthreads();
  }

  if(tid == 0) 
  {
    g_odata[blockIdx.x+aOffsetOutput] = sdata2[0][0];
    g_odata2[blockIdx.x+aOffsetOutput] = sdata2[0][1];
  }
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
  checkCudaErrors(cudaFree(d_fPairKStar3dVec));
  checkCudaErrors(cudaFree(d_fPairKStar3dVecInfo));

  checkCudaErrors(cudaFree(d_fLednickyHFunction));

  checkCudaErrors(cudaFree(d_fGTildeReal));
  checkCudaErrors(cudaFree(d_fGTildeImag));
  checkCudaErrors(cudaFree(d_fGTildeInfo));

  checkCudaErrors(cudaFree(d_fHyperGeo1F1Real));
  checkCudaErrors(cudaFree(d_fHyperGeo1F1Imag));
  checkCudaErrors(cudaFree(d_fHyperGeo1F1Info));
}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadBohrRadius(double aRadius)
{
  d_fBohrRadius = aRadius;
}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadPairSample4dVec(td4dVec &aPairSample4dVec, BinInfoSamplePairs &aBinInfo)
{
  //------ Load bin info first ---------------------------
  checkCudaErrors(cudaMallocManaged(&d_fPairSample4dVecInfo, sizeof(BinInfoSamplePairs)));

  d_fPairSample4dVecInfo->nAnalyses = aBinInfo.nAnalyses;
  d_fPairSample4dVecInfo->nBinsK = aBinInfo.nBinsK;
  d_fPairSample4dVecInfo->nPairsPerBin = aBinInfo.nPairsPerBin;
  d_fPairSample4dVecInfo->minK = aBinInfo.minK;
  d_fPairSample4dVecInfo->maxK = aBinInfo.maxK;
  d_fPairSample4dVecInfo->binWidthK = aBinInfo.binWidthK;
  d_fPairSample4dVecInfo->nElementsPerPair = aBinInfo.nElementsPerPair;
  //------------------------------------------------------
  fSamplePairsBinInfo.nAnalyses = aBinInfo.nAnalyses;
  fSamplePairsBinInfo.nBinsK = aBinInfo.nBinsK;
  fSamplePairsBinInfo.nPairsPerBin = aBinInfo.nPairsPerBin;
  fSamplePairsBinInfo.minK = aBinInfo.minK;
  fSamplePairsBinInfo.maxK = aBinInfo.maxK;
  fSamplePairsBinInfo.binWidthK = aBinInfo.binWidthK;
  fSamplePairsBinInfo.nElementsPerPair = aBinInfo.nElementsPerPair;
  //------------------------------------------------------
  assert((int)aPairSample4dVec.size() == d_fPairSample4dVecInfo->nAnalyses);
  assert((int)aPairSample4dVec[0].size() == d_fPairSample4dVecInfo->nBinsK);
  assert((int)aPairSample4dVec[0][0].size() == d_fPairSample4dVecInfo->nPairsPerBin);
  assert((int)aPairSample4dVec[0][0][0].size() == d_fPairSample4dVecInfo->nElementsPerPair);
  assert(d_fPairSample4dVecInfo->nElementsPerPair == 3);
  //------------------------------------------------------

  int tTotalPairs = 0;
  for(int iAnaly=0; iAnaly<(int)aPairSample4dVec.size(); iAnaly++)
  {
    for(int iK=0; iK<(int)aPairSample4dVec[iAnaly].size(); iK++) tTotalPairs += aPairSample4dVec[iAnaly][iK].size();
  }
  int tSize = tTotalPairs*fSamplePairsBinInfo.nElementsPerPair*sizeof(double);
  checkCudaErrors(cudaMallocManaged(&d_fPairSample4dVec, tSize));

  int tIndex=0;
  for(int iAnaly=0; iAnaly<(int)aPairSample4dVec.size(); iAnaly++)
  {
    for(int iK=0; iK<(int)aPairSample4dVec[iAnaly].size(); iK++)
    {
      for(int iPair=0; iPair<(int)aPairSample4dVec[iAnaly][iK].size(); iPair++)
      {
        d_fPairSample4dVec[tIndex] = aPairSample4dVec[iAnaly][iK][iPair][0];
        d_fPairSample4dVec[tIndex+1] = aPairSample4dVec[iAnaly][iK][iPair][1];
        d_fPairSample4dVec[tIndex+2] = aPairSample4dVec[iAnaly][iK][iPair][2];
        tIndex += d_fPairSample4dVecInfo->nElementsPerPair;
      }
    }
  }

}

//________________________________________________________________________________________________________________
void ParallelWaveFunction::UpdatePairSampleRadii(double aScaleFactor)
{
  //TODO make this more general, and probably is better way to do this
  int tTotalEntries = d_fPairSample4dVecInfo->nAnalyses * d_fPairSample4dVecInfo->nBinsK * d_fPairSample4dVecInfo->nPairsPerBin * d_fPairSample4dVecInfo->nElementsPerPair;
  for(int i=0; i<tTotalEntries; i++)
  {
    if(i%d_fPairSample4dVecInfo->nElementsPerPair == 1) d_fPairSample4dVec[i] *= aScaleFactor;
  }
}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadPairKStar3dVec(td3dVec &aPairKStar3dVec, BinInfoKStar &aBinInfo)
{
  //------ Load bin info first ---------------------------
  checkCudaErrors(cudaMallocManaged(&d_fPairKStar3dVecInfo, sizeof(BinInfoKStar)));

  d_fPairKStar3dVecInfo->nBinsK = aBinInfo.nBinsK;
  d_fPairKStar3dVecInfo->minK = aBinInfo.minK;
  d_fPairKStar3dVecInfo->maxK = aBinInfo.maxK;
  d_fPairKStar3dVecInfo->binWidthK = aBinInfo.binWidthK;
  for(int i=0; i<d_fPairKStar3dVecInfo->nBinsK; i++)
  {
    d_fPairKStar3dVecInfo->nPairsPerBin[i] = aBinInfo.nPairsPerBin[i];
    d_fPairKStar3dVecInfo->binOffset[i] = aBinInfo.binOffset[i];
  }
  //------------------------------------------------------
  int tNbinsK = aPairKStar3dVec.size();
  assert(tNbinsK == d_fPairKStar3dVecInfo->nBinsK);

  int tNPairsTotal=0;
  for(int i=0; i<tNbinsK; i++) tNPairsTotal += aPairKStar3dVec[i].size();
  int tNPairsTotal2=0;
  for(int i=0; i<tNbinsK; i++) tNPairsTotal2 += d_fPairKStar3dVecInfo->nPairsPerBin[i];
  assert(tNPairsTotal == tNPairsTotal2);

  assert(aPairKStar3dVec[0][0].size() == 4); //all should have the same size, but maybe input a more thorough check

  int tSize = tNPairsTotal*4*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fPairKStar3dVec, tSize));

  int tIndex=0;
  int tIndex2=0;
  int tOffset=0;
  for(int iK=0; iK<tNbinsK; iK++)
  {
    for(int iPair=0; iPair<(int)aPairKStar3dVec[iK].size(); iPair++)
    {
      tOffset = d_fPairKStar3dVecInfo->binOffset[iK];
      tIndex2 = tOffset + 4*iPair;
      assert(tIndex2 == tIndex);

      d_fPairKStar3dVec[tIndex] = aPairKStar3dVec[iK][iPair][0];
      d_fPairKStar3dVec[tIndex+1] = aPairKStar3dVec[iK][iPair][1];
      d_fPairKStar3dVec[tIndex+2] = aPairKStar3dVec[iK][iPair][2];
      d_fPairKStar3dVec[tIndex+3] = aPairKStar3dVec[iK][iPair][3];
      tIndex+=4;
    }
  }
}


//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadLednickyHFunction(td1dVec &aHFunc)
{
  int tNbinsK = aHFunc.size();

  int tSize = tNbinsK*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fLednickyHFunction, tSize));

  for(int iK=0; iK<tNbinsK; iK++)
  {
    d_fLednickyHFunction[iK] = aHFunc[iK];
  }
}

/*
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
*/

//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadGTildeReal(td2dVec &aGTildeReal)
{
  int tNbinsK = aGTildeReal.size();
  int tNbinsR = aGTildeReal[0].size();

  int tSize = tNbinsK*tNbinsR*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fGTildeReal, tSize));

  int tIndex;
  float *A = new float[tNbinsK*tNbinsR];
  for(int iK=0; iK<tNbinsK; iK++)
  {
    for(int iR=0; iR<tNbinsR; iR++)
    {
      tIndex = iR + iK*tNbinsR;
      d_fGTildeReal[tIndex] = aGTildeReal[iK][iR];
      A[tIndex] = aGTildeReal[iK][iR];
    }
  }

//----------------------------------

texRefA.addressMode[0] = cudaAddressModeClamp;
texRefA.addressMode[1] = cudaAddressModeClamp;
texRefA.filterMode = cudaFilterModeLinear;
texRefA.normalized = false;

cudaGetTextureReference(&texRefAPtr, &texRefA);
cudaChannelFormatDesc channelDescA = cudaCreateChannelDesc<float>();

size_t pitchA;
checkCudaErrors(cudaMallocPitch(&dA, &pitchA, sizeof(float)*tNbinsR, tNbinsK));
checkCudaErrors(cudaMemcpy2D(dA, pitchA, A, sizeof(float)*tNbinsR, sizeof(float)*tNbinsR, tNbinsK, cudaMemcpyHostToDevice));

size_t offset;
cudaBindTexture2D(&offset, texRefAPtr, dA, &channelDescA, tNbinsR, tNbinsK, pitchA);

delete[] A;
}

/*
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
*/

//________________________________________________________________________________________________________________
void ParallelWaveFunction::LoadGTildeImag(td2dVec &aGTildeImag)
{
  int tNbinsK = aGTildeImag.size();
  int tNbinsR = aGTildeImag[0].size();

  int tSize = tNbinsK*tNbinsR*sizeof(double);

  checkCudaErrors(cudaMallocManaged(&d_fGTildeImag, tSize));

  int tIndex;
  float *B = new float[tNbinsK*tNbinsR];
  for(int iK=0; iK<tNbinsK; iK++)
  {
    for(int iR=0; iR<tNbinsR; iR++)
    {
      tIndex = iR + iK*tNbinsR;
      d_fGTildeImag[tIndex] = aGTildeImag[iK][iR];
      B[tIndex] = aGTildeImag[iK][iR];
    }
  }

//----------------------------------

texRefB.addressMode[0] = cudaAddressModeClamp;
texRefB.addressMode[1] = cudaAddressModeClamp;
texRefB.filterMode = cudaFilterModeLinear;
texRefB.normalized = false;

cudaGetTextureReference(&texRefBPtr, &texRefB);
cudaChannelFormatDesc channelDescB = cudaCreateChannelDesc<float>();

size_t pitchB;
checkCudaErrors(cudaMallocPitch(&dB, &pitchB, sizeof(float)*tNbinsR, tNbinsK));
checkCudaErrors(cudaMemcpy2D(dB, pitchB, B, sizeof(float)*tNbinsR, sizeof(float)*tNbinsR, tNbinsK, cudaMemcpyHostToDevice));

size_t offset;
cudaBindTexture2D(&offset, texRefBPtr, dB, &channelDescB, tNbinsR, tNbinsK, pitchB);

delete[] B;

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
//double* ParallelWaveFunction::RunInterpolateWfSquared(td2dVec &aPairs, double aReF0, double aImF0, double aD0)
vector<double> ParallelWaveFunction::RunInterpolateWfSquared(td2dVec &aPairs, double aReF0, double aImF0, double aD0)
{
  int tNPairs = aPairs.size();
  int tSize = tNPairs*sizeof(double);
  int tSizeShared = fNThreadsPerBlock*sizeof(double);
  int tSizeOut = fNBlocks*sizeof(double);

  //---Host arrays and allocations
  double * h_KStarMag;
  double * h_RStarMag;
  double * h_Theta;
  double * h_WfSquared;


  checkCudaErrors(cudaMallocManaged(&h_KStarMag, tSize));
  checkCudaErrors(cudaMallocManaged(&h_RStarMag, tSize));
  checkCudaErrors(cudaMallocManaged(&h_Theta, tSize));
  checkCudaErrors(cudaMallocManaged(&h_WfSquared, tSizeOut));

  for(int i=0; i<tNPairs; i++)
  {
    h_KStarMag[i] = aPairs[i][0];
    h_RStarMag[i] = aPairs[i][1];
    h_Theta[i] = aPairs[i][2];
  }

  //----------Run the kernel-----------------------------------------------
  GpuTimer timer;
  timer.Start();
  GetWfAverage<<<fNBlocks,fNThreadsPerBlock,tSizeShared>>>(h_KStarMag,h_RStarMag,h_Theta,aReF0,aImF0,aD0,h_WfSquared);
  timer.Stop();
  std::cout << "InterpolateWfSquared kernel finished in " << timer.Elapsed() << " ms" << std::endl;

  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());


  checkCudaErrors(cudaFree(h_KStarMag));
  checkCudaErrors(cudaFree(h_RStarMag));
  checkCudaErrors(cudaFree(h_Theta));

//  return h_WfSquared;
  vector<double> tReturnVec(tNPairs);
  for(int i=0; i<fNBlocks; i++) 
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

//________________________________________________________________________________________________________________
vector<double> ParallelWaveFunction::RunInterpolateEntireCf(td3dVec &aPairs, double aReF0, double aImF0, double aD0)
{
//  GpuTimer timerPre;
//  timerPre.Start();

  int tNBins = aPairs.size();
  int tNPairsPerBin = aPairs[0].size();  //TODO all bins should have equal number of pairs
  int tSizeInput = tNBins*tNPairsPerBin*sizeof(double);
  int tSizeOutput = tNBins*fNBlocks*sizeof(double); //the kernel reduces the values for tNPairs bins down to fNBlocks bins
  int tSizeShared = fNThreadsPerBlock*sizeof(double);

  const int tNStreams = tNBins;

  //---Host arrays and allocations
  double * h_KStarMag;
  double * h_RStarMag;
  double * h_Theta;
  double * h_Cf;

  checkCudaErrors(cudaMallocManaged(&h_KStarMag, tSizeInput));
  checkCudaErrors(cudaMallocManaged(&h_RStarMag, tSizeInput));
  checkCudaErrors(cudaMallocManaged(&h_Theta, tSizeInput));

  checkCudaErrors(cudaMallocManaged(&h_Cf, tSizeOutput));

  cudaStream_t tStreams[tNStreams];

  for(int i=0; i<tNBins; i++)
  {
    cudaStreamCreate(&tStreams[i]);
    for(int j=0; j<tNPairsPerBin; j++)
    {
      h_KStarMag[j+i*tNPairsPerBin] = aPairs[i][j][0];
      h_RStarMag[j+i*tNPairsPerBin] = aPairs[i][j][1];
      h_Theta[j+i*tNPairsPerBin] = aPairs[i][j][2];
    }
  }

//  timerPre.Stop();
//  std::cout << " timerPre: " << timerPre.Elapsed() << " ms" << std::endl;


  //----------Run the kernels-----------------------------------------------
//  GpuTimer timer;
//  timer.Start();

  for(int i=0; i<tNBins; i++)
  {
    int tOffsetInput = i*tNPairsPerBin;
    int tOffsetOutput = i*fNBlocks;
    GetEntireCf<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(h_KStarMag,h_RStarMag,h_Theta,aReF0,aImF0,aD0,h_Cf,tOffsetInput,tOffsetOutput);
  }
//  timer.Stop();
//  std::cout << "GetEntireCf kernel finished in " << timer.Elapsed() << " ms" << std::endl;


//  GpuTimer timerPost;
//  timerPost.Start();

  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(h_KStarMag));
  checkCudaErrors(cudaFree(h_RStarMag));
  checkCudaErrors(cudaFree(h_Theta));


  // return the CF
  vector<double> tReturnVec(tNBins);
  double tSum = 0.0;
  for(int i=0; i<tNBins; i++)
  {
    tSum=0.0;
    for(int j=0; j<fNBlocks; j++)
    {
      tSum += h_Cf[j+i*fNBlocks]; 
    }
    tReturnVec[i] = tSum;
  }

  checkCudaErrors(cudaFree(h_Cf));

  for(int i=0; i<tNStreams; i++) cudaStreamDestroy(tStreams[i]);

//  timerPost.Stop();
//  std::cout << " timerPost: " << timerPost.Elapsed() << " ms" << std::endl;

  return tReturnVec;
}


//________________________________________________________________________________________________________________
vector<double> ParallelWaveFunction::RunInterpolateEntireCfComplete(td3dVec &aPairs, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t)
{
//  GpuTimer timerPre;
//  timerPre.Start();

  int tNBins = aPairs.size();
  int tNPairsPerBin = aPairs[0].size();  //TODO all bins should have equal number of pairs
  int tSizeInput = tNBins*tNPairsPerBin*sizeof(double);
  int tSizeOutput = tNBins*fNBlocks*sizeof(double); //the kernel reduces the values for tNPairs bins down to fNBlocks bins
  int tSizeShared = fNThreadsPerBlock*sizeof(double);

  const int tNStreams = tNBins;

  //---Host arrays and allocations
  double * h_KStarMag;
  double * h_RStarMag;
  double * h_Theta;
  double * h_Cf;

  checkCudaErrors(cudaMallocManaged(&h_KStarMag, tSizeInput));
  checkCudaErrors(cudaMallocManaged(&h_RStarMag, tSizeInput));
  checkCudaErrors(cudaMallocManaged(&h_Theta, tSizeInput));

  checkCudaErrors(cudaMallocManaged(&h_Cf, tSizeOutput));

  cudaStream_t tStreams[tNStreams];

  for(int i=0; i<tNBins; i++)
  {
    cudaStreamCreate(&tStreams[i]);
    for(int j=0; j<tNPairsPerBin; j++)
    {
      h_KStarMag[j+i*tNPairsPerBin] = aPairs[i][j][0];
      h_RStarMag[j+i*tNPairsPerBin] = aPairs[i][j][1];
      h_Theta[j+i*tNPairsPerBin] = aPairs[i][j][2];
    }
  }

//  timerPre.Stop();
//  std::cout << " timerPre: " << timerPre.Elapsed() << " ms" << std::endl;


  //----------Run the kernels-----------------------------------------------
//  GpuTimer timer;
//  timer.Start();

  for(int i=0; i<tNBins; i++)
  {
    int tOffsetInput = i*tNPairsPerBin;
    int tOffsetOutput = i*fNBlocks;
    GetEntireCfComplete<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(h_KStarMag,h_RStarMag,h_Theta,aReF0s,aImF0s,aD0s,aReF0t,aImF0t,aD0t,h_Cf,tOffsetInput,tOffsetOutput);
  }
//  timer.Stop();
//  std::cout << "GetEntireCf kernel finished in " << timer.Elapsed() << " ms" << std::endl;


//  GpuTimer timerPost;
//  timerPost.Start();

  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(h_KStarMag));
  checkCudaErrors(cudaFree(h_RStarMag));
  checkCudaErrors(cudaFree(h_Theta));


  // return the CF
  vector<double> tReturnVec(tNBins);
  double tSum = 0.0;
  for(int i=0; i<tNBins; i++)
  {
    tSum=0.0;
    for(int j=0; j<fNBlocks; j++)
    {
      tSum += h_Cf[j+i*fNBlocks]; 
    }
    tReturnVec[i] = tSum;
  }

  checkCudaErrors(cudaFree(h_Cf));

  for(int i=0; i<tNStreams; i++) cudaStreamDestroy(tStreams[i]);

//  timerPost.Stop();
//  std::cout << " timerPost: " << timerPost.Elapsed() << " ms" << std::endl;

  return tReturnVec;
}

/*
//________________________________________________________________________________________________________________
td2dVec ParallelWaveFunction::RunInterpolateEntireCfwStaticPairs(int aAnalysisNumber, double aRadiusScale, double aReF0, double aImF0, double aD0)
{
  GpuTimer timer;
  timer.Start();

  int tNBins = fSamplePairsBinInfo.nBinsK;
//  int tNPairsPerBin = fSamplePairsBinInfo.nPairsPerBin;
  int tSizeOutput = tNBins*fNBlocks*sizeof(double); //the kernel reduces the values for tNPairs bins down to fNBlocks bins
  int tSizeShared = fNThreadsPerBlock*sizeof(double);
  tSizeShared *= 2; //to account for Cf values and counts

  const int tNStreams = tNBins;

  //---Host arrays and allocations
  double * h_CfSums;
  double * h_CfCounts;

  checkCudaErrors(cudaMallocManaged(&h_CfSums, tSizeOutput));
  checkCudaErrors(cudaMallocManaged(&h_CfCounts, tSizeOutput));

  cudaStream_t tStreams[tNStreams];

  for(int i=0; i<tNBins; i++)
  {
    cudaStreamCreate(&tStreams[i]);
  }

  timer.Stop();
  std::cout << " Setup time: " << timer.Elapsed() << " ms" << std::endl;


  //----------Run the kernels-----------------------------------------------
  timer.Start();

  for(int i=0; i<tNBins; i++)
  {
    int tOffsetOutput = i*fNBlocks;
    GetEntireCfwStaticPairs<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(aRadiusScale, aReF0, aImF0, aD0, h_CfSums, h_CfCounts, aAnalysisNumber, i, tOffsetOutput);
  }
  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());
  timer.Stop();
  std::cout << " GetEntireCfwStaticPairs kernel and cudaDeviceSynchronize() finished in " << timer.Elapsed() << " ms" << std::endl;
  //NOTE: cudaDeviceSynchronize should be included in kernel time calculation because...
  //  The kernel call is asynchronous, meaning it launches the kernel and then immediately returns control to the host thread, allowing the host thread to continue. 
  //  Therefore the overhead in the host thread for a kernel call may be as low as a few microseconds.

  timer.Start();
  // return the CF
  td2dVec tReturnVec;
    tReturnVec.resize(tNBins,td1dVec(2));

  double tSum = 0.0;
  int tCounts = 0;
  for(int i=0; i<tNBins; i++)
  {
    tSum=0.0;
    tCounts = 0;
    for(int j=0; j<fNBlocks; j++)
    {
      tSum += h_CfSums[j+i*fNBlocks]; 
      tCounts += h_CfCounts[j+i*fNBlocks]; 
    }
    tReturnVec[i][0] = tSum;
    tReturnVec[i][1] = tCounts;
  }

  checkCudaErrors(cudaFree(h_CfSums));
  checkCudaErrors(cudaFree(h_CfCounts));

  for(int i=0; i<tNStreams; i++) cudaStreamDestroy(tStreams[i]);

  timer.Stop();
  std::cout << " timerPost: " << timer.Elapsed() << " ms" << std::endl;

  return tReturnVec;
}
*/



//________________________________________________________________________________________________________________
td2dVec ParallelWaveFunction::RunInterpolateEntireCfwStaticPairs(int aAnalysisNumber, double aRadiusScale, double aReF0, double aImF0, double aD0)
{
  bool tOutputTime = false;
  GpuTimer timer;
  if(tOutputTime) timer.Start();

  int tNBins = fSamplePairsBinInfo.nBinsK;
//  int tNPairsPerBin = fSamplePairsBinInfo.nPairsPerBin;
  int tSizeOutput = tNBins*fNBlocks*sizeof(double); //the kernel reduces the values for tNPairs bins down to fNBlocks bins
  int tSizeShared = fNThreadsPerBlock*sizeof(double);
  tSizeShared *= 2; //to account for Cf values and counts

  const int tNStreams = tNBins;

  //---Host arrays and allocations
  double * h_CfSums;
  double * h_CfCounts;

  checkCudaErrors(cudaMallocManaged(&h_CfSums, tSizeOutput));
  checkCudaErrors(cudaMallocManaged(&h_CfCounts, tSizeOutput));

  double * h_GamowFactors;

  double * h_ExpTermsReal;
  double * h_ExpTermsImag;

  double * h_GTildeReal;
  double * h_GTildeImag;

  double * h_GTildeRealTexture;
  double * h_GTildeImagTexture;

  double * h_HyperGeo1F1Real;
  double * h_HyperGeo1F1Imag;

  double * h_ScattLenReal;
  double * h_ScattLenImag;





  int tSizeOutputBig = tNBins*fNThreadsPerBlock*fNBlocks*sizeof(double);
  checkCudaErrors(cudaMallocManaged(&h_GamowFactors, tSizeOutputBig));

  checkCudaErrors(cudaMallocManaged(&h_ExpTermsReal, tSizeOutputBig));
  checkCudaErrors(cudaMallocManaged(&h_ExpTermsImag, tSizeOutputBig));

  checkCudaErrors(cudaMallocManaged(&h_GTildeReal, tSizeOutputBig));
  checkCudaErrors(cudaMallocManaged(&h_GTildeImag, tSizeOutputBig));

  checkCudaErrors(cudaMallocManaged(&h_GTildeRealTexture, tSizeOutputBig));
  checkCudaErrors(cudaMallocManaged(&h_GTildeImagTexture, tSizeOutputBig));

  checkCudaErrors(cudaMallocManaged(&h_HyperGeo1F1Real, tSizeOutputBig));
  checkCudaErrors(cudaMallocManaged(&h_HyperGeo1F1Imag, tSizeOutputBig));

  checkCudaErrors(cudaMallocManaged(&h_ScattLenReal, tSizeOutputBig));
  checkCudaErrors(cudaMallocManaged(&h_ScattLenImag, tSizeOutputBig));

  cudaStream_t tStreams[tNStreams];

  for(int i=0; i<tNStreams; i++)
  {
    cudaStreamCreate(&tStreams[i]);
  }

  if(tOutputTime) 
  {
    timer.Stop();
    std::cout << " Setup time: " << timer.Elapsed() << " ms" << std::endl;
  }

  //----------Run the kernels-----------------------------------------------
  if(tOutputTime) timer.Start();
/*
  for(int i=0; i<tNBins; i++)
  {
    int tOffsetOutput = i*fNBlocks;

    GetAllGamowFactors<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[0]>>>(h_GamowFactors, aAnalysisNumber, i, tOffsetOutput);
    GetAllExpTermsCmplx<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[1]>>>(aRadiusScale, h_ExpTermsReal, h_ExpTermsImag, aAnalysisNumber, i, tOffsetOutput);
    GetAllGTildeCmplx<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[2]>>>(aRadiusScale, h_GTildeReal, h_GTildeImag, aAnalysisNumber, i, tOffsetOutput);
    GetAllHyperGeo1F1Cmplx<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[3]>>>(aRadiusScale, h_HyperGeo1F1Real, h_HyperGeo1F1Imag, aAnalysisNumber, i, tOffsetOutput);
    GetAllScattLenCmplx<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[4]>>>(aRadiusScale, aReF0, aImF0, aD0, h_ScattLenReal, h_ScattLenImag, aAnalysisNumber, i, tOffsetOutput);

    checkCudaErrors(cudaDeviceSynchronize());

    InterpolateAllWfSquared<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[5]>>>(aRadiusScale, h_GamowFactors, h_ExpTermsReal, h_ExpTermsImag, h_GTildeReal, h_GTildeImag, h_HyperGeo1F1Real, h_HyperGeo1F1Imag, h_ScattLenReal, h_ScattLenImag, h_CfSums, h_CfCounts, aAnalysisNumber, i, tOffsetOutput);
  }
*/
  int tOffset = 0;
  for(int i=0; i<tNBins; i++)
  {
    tOffset = i*fNThreadsPerBlock*fNBlocks;
    GetAllGamowFactors<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(h_GamowFactors, aAnalysisNumber, i, tOffset);
  }

  for(int i=0; i<tNBins; i++)
  {
    tOffset = i*fNThreadsPerBlock*fNBlocks;
    GetAllExpTermsCmplx<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(aRadiusScale, h_ExpTermsReal, h_ExpTermsImag, aAnalysisNumber, i, tOffset);
  }

  for(int i=0; i<tNBins; i++)
  {
    tOffset = i*fNThreadsPerBlock*fNBlocks;
    GetAllGTildeCmplx<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(aRadiusScale, h_GTildeReal, h_GTildeImag, aAnalysisNumber, i, tOffset);
  }

  for(int i=0; i<tNBins; i++)
  {
    tOffset = i*fNThreadsPerBlock*fNBlocks;
    GetAllGTildeCmplxTexture<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(aRadiusScale, h_GTildeRealTexture, h_GTildeImagTexture, aAnalysisNumber, i, tOffset);
  }

  for(int i=0; i<tNBins; i++)
  {
    tOffset = i*fNThreadsPerBlock*fNBlocks;
    GetAllHyperGeo1F1Cmplx<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(aRadiusScale, h_HyperGeo1F1Real, h_HyperGeo1F1Imag, aAnalysisNumber, i, tOffset);
  }

  for(int i=0; i<tNBins; i++)
  {
    tOffset = i*fNThreadsPerBlock*fNBlocks;
    GetAllScattLenCmplx<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(aRadiusScale, aReF0, aImF0, aD0, h_ScattLenReal, h_ScattLenImag, aAnalysisNumber, i, tOffset);
  }

  checkCudaErrors(cudaDeviceSynchronize());

  int tOffsetInput=0, tOffsetOutput=0;
  for(int i=0; i<tNBins; i++)
  {
    tOffsetInput = i*fNThreadsPerBlock*fNBlocks;
    tOffsetOutput = i*fNBlocks;
    InterpolateAllWfSquared<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(aRadiusScale, h_GamowFactors, h_ExpTermsReal, h_ExpTermsImag, h_GTildeReal, h_GTildeImag, h_HyperGeo1F1Real, h_HyperGeo1F1Imag, h_ScattLenReal, h_ScattLenImag, h_CfSums, h_CfCounts, aAnalysisNumber, i, tOffsetInput, tOffsetOutput);
  }

  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());
  if(tOutputTime) 
  {
    timer.Stop();
    std::cout << " GetEntireCfwStaticPairs kernel and cudaDeviceSynchronize() finished in " << timer.Elapsed() << " ms" << std::endl;
  }
  //NOTE: cudaDeviceSynchronize should be included in kernel time calculation because...
  //  The kernel call is asynchronous, meaning it launches the kernel and then immediately returns control to the host thread, allowing the host thread to continue. 
  //  Therefore the overhead in the host thread for a kernel call may be as low as a few microseconds.

  if(tOutputTime) timer.Start();
  // return the CF
  td2dVec tReturnVec;
    tReturnVec.resize(tNBins,td1dVec(2));

  double tSum = 0.0;
  int tCounts = 0;
  for(int i=0; i<tNBins; i++)
  {
    tSum=0.0;
    tCounts = 0;
    for(int j=0; j<fNBlocks; j++)
    {
      tSum += h_CfSums[j+i*fNBlocks]; 
      tCounts += h_CfCounts[j+i*fNBlocks]; 
    }
    tReturnVec[i][0] = tSum;
    tReturnVec[i][1] = tCounts;
  }

  checkCudaErrors(cudaFree(h_CfSums));
  checkCudaErrors(cudaFree(h_CfCounts));

  checkCudaErrors(cudaFree(h_GamowFactors));

  checkCudaErrors(cudaFree(h_ExpTermsReal));
  checkCudaErrors(cudaFree(h_ExpTermsImag));

  checkCudaErrors(cudaFree(h_GTildeReal));
  checkCudaErrors(cudaFree(h_GTildeImag));

  checkCudaErrors(cudaFree(h_GTildeRealTexture));
  checkCudaErrors(cudaFree(h_GTildeImagTexture));

  checkCudaErrors(cudaFree(h_HyperGeo1F1Real));
  checkCudaErrors(cudaFree(h_HyperGeo1F1Imag));

  checkCudaErrors(cudaFree(h_ScattLenReal));
  checkCudaErrors(cudaFree(h_ScattLenImag));

  for(int i=0; i<tNStreams; i++) cudaStreamDestroy(tStreams[i]);

  if(tOutputTime) 
  {
    timer.Stop();
    std::cout << " timerPost: " << timer.Elapsed() << " ms" << std::endl;
  }

  return tReturnVec;
}






//________________________________________________________________________________________________________________
td2dVec ParallelWaveFunction::RunInterpolateEntireCfCompletewStaticPairs(int aAnalysisNumber, double aRadiusScale, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t)
{
  GpuTimer timer;
  timer.Start();

  int tNBins = fSamplePairsBinInfo.nBinsK;
//  int tNPairsPerBin = fSamplePairsBinInfo.nPairsPerBin;
  int tSizeOutput = tNBins*fNBlocks*sizeof(double); //the kernel reduces the values for tNPairs bins down to fNBlocks bins
  int tSizeShared = fNThreadsPerBlock*sizeof(double);
  tSizeShared *= 2; //to account for Cf values and counts

  const int tNStreams = tNBins;

  //---Host arrays and allocations
  double * h_CfSums;
  double * h_CfCounts;

  checkCudaErrors(cudaMallocManaged(&h_CfSums, tSizeOutput));
  checkCudaErrors(cudaMallocManaged(&h_CfCounts, tSizeOutput));

  cudaStream_t tStreams[tNStreams];

  for(int i=0; i<tNBins; i++)
  {
    cudaStreamCreate(&tStreams[i]);
  }

  timer.Stop();
  std::cout << " Setup time: " << timer.Elapsed() << " ms" << std::endl;


  //----------Run the kernels-----------------------------------------------
  timer.Start();

  for(int i=0; i<tNBins; i++)
  {
    int tOffsetOutput = i*fNBlocks;
    GetEntireCfCompletewStaticPairs<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(aRadiusScale, aReF0s, aImF0s, aD0s, aReF0t, aImF0t, aD0t, h_CfSums, h_CfCounts, aAnalysisNumber, i, tOffsetOutput);
  }
  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());
  timer.Stop();
  std::cout << " GetEntireCfCompletewStaticPairs kernel and cudaDeviceSynchronize() finished in " << timer.Elapsed() << " ms" << std::endl;
  //NOTE: cudaDeviceSynchronize should be included in kernel time calculation because...
  //  The kernel call is asynchronous, meaning it launches the kernel and then immediately returns control to the host thread, allowing the host thread to continue. 
  //  Therefore the overhead in the host thread for a kernel call may be as low as a few microseconds.

  timer.Start();
  // return the CF
  td2dVec tReturnVec;
    tReturnVec.resize(tNBins,td1dVec(2));

  double tSum = 0.0;
  int tCounts = 0;
  for(int i=0; i<tNBins; i++)
  {
    tSum=0.0;
    tCounts = 0;
    for(int j=0; j<fNBlocks; j++)
    {
      tSum += h_CfSums[j+i*fNBlocks]; 
      tCounts += h_CfCounts[j+i*fNBlocks]; 
    }
    tReturnVec[i][0] = tSum;
    tReturnVec[i][1] = tCounts;
  }

  checkCudaErrors(cudaFree(h_CfSums));
  checkCudaErrors(cudaFree(h_CfCounts));

  for(int i=0; i<tNStreams; i++) cudaStreamDestroy(tStreams[i]);

  timer.Stop();
  std::cout << " timerPost: " << timer.Elapsed() << " ms" << std::endl;

  return tReturnVec;
}





//________________________________________________________________________________________________________________
vector<double> ParallelWaveFunction::RunInterpolateEntireCfComplete2(int aNSimPairsPerBin, double aKStarMin, double aKStarMax, double aNbinsK, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t)
{
//  GpuTimer timerPre;
//  timerPre.Start();

  int tNBins = aNbinsK;
  int tNPairsPerBin = aNSimPairsPerBin;
//  int tSizeInput = tNBins*tNPairsPerBin*sizeof(double);
  int tSizeOutput = tNBins*fNBlocks*sizeof(double); //the kernel reduces the values for tNPairs bins down to fNBlocks bins
  int tSizeShared = fNThreadsPerBlock*sizeof(double);
  int tSizedState = tNBins*tNPairsPerBin*sizeof(curandState);
  int tSizeCPUPairs = tNBins*tNPairsPerBin*3*sizeof(double);

  const int tNStreams = tNBins;

  //---Host arrays and allocations
  double * h_Cf;
  double * h_CPUPairs;

  checkCudaErrors(cudaMallocManaged(&h_Cf, tSizeOutput));
  checkCudaErrors(cudaMallocManaged(&h_CPUPairs, tSizeCPUPairs));

  cudaStream_t tStreams[tNStreams];

  for(int i=0; i<tNBins; i++) cudaStreamCreate(&tStreams[i]);

  curandState *d_state1;
  checkCudaErrors(cudaMallocManaged(&d_state1, tSizedState));

  curandState *d_state2;
  checkCudaErrors(cudaMallocManaged(&d_state2, tSizedState));

  curandState *d_state3;
  checkCudaErrors(cudaMallocManaged(&d_state3, tSizedState));

//  timerPre.Stop();
//  std::cout << " timerPre: " << timerPre.Elapsed() << " ms" << std::endl;


  //----------Run the kernels-----------------------------------------------
//  GpuTimer timer;
//  timer.Start();

  for(int i=0; i<tNBins; i++)
  {
    int tOffsetInput = i*tNPairsPerBin;
    int tOffsetOutput = i*fNBlocks;

    RandInit<<<fNBlocks,fNThreadsPerBlock,0,tStreams[i]>>>(d_state1,std::clock(),tOffsetInput);
    RandInit<<<fNBlocks,fNThreadsPerBlock,0,tStreams[i]>>>(d_state2,std::clock(),tOffsetInput);
    RandInit<<<fNBlocks,fNThreadsPerBlock,0,tStreams[i]>>>(d_state3,std::clock(),tOffsetInput);
    GetEntireCfComplete2<<<fNBlocks,fNThreadsPerBlock,tSizeShared,tStreams[i]>>>(d_state1,d_state2,d_state3, aR, aReF0s,aImF0s,aD0s, aReF0t,aImF0t,aD0t, h_Cf,i,tOffsetInput,tOffsetOutput,h_CPUPairs);
  }
//  timer.Stop();
//  std::cout << "GetEntireCfComplete2 kernel finished in " << timer.Elapsed() << " ms" << std::endl;


//  GpuTimer timerPost;
//  timerPost.Start();

  //The following is necessary for the host to be able to "see" the changes that have been done
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_state1));
  checkCudaErrors(cudaFree(d_state2));
  checkCudaErrors(cudaFree(d_state3));

  // return the CF
  vector<double> tReturnVec(tNBins);
  double tSum = 0.0;
  for(int i=0; i<tNBins; i++)
  {
    tSum=0.0;
    for(int j=0; j<fNBlocks; j++)
    {
      tSum += h_Cf[j+i*fNBlocks]; 
    }
    tReturnVec[i] = tSum;
  }

  checkCudaErrors(cudaFree(h_Cf));
  checkCudaErrors(cudaFree(h_CPUPairs));

  for(int i=0; i<tNStreams; i++) cudaStreamDestroy(tStreams[i]);

//  timerPost.Stop();
//  std::cout << " timerPost: " << timerPost.Elapsed() << " ms" << std::endl;

  return tReturnVec;
}

