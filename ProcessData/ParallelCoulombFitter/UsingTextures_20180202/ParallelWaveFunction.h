///////////////////////////////////////////////////////////////////////////
// ParallelWaveFunction:                                                 //
///////////////////////////////////////////////////////////////////////////

#ifndef PARALLELWAVEFUNCTION_H
#define PARALLELWAVEFUNCTION_H

//includes and any constant variable declarations
#include <stdio.h>
#include <iostream>
#include <vector>

#include "timer.h"
#include "utils.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Types.h"
#include "ParallelTypes.h"
#include "timer.h"

using std::cout;
using std::endl;
using std::vector;

extern __managed__ double* d_fLednickyHFunction;
extern __managed__ double *d_fGTildeReal, *d_fGTildeImag, *d_fHyperGeo1F1Real, *d_fHyperGeo1F1Imag;
extern __managed__ BinInfoGTilde *d_fGTildeInfo;
extern __managed__ BinInfoHyperGeo1F1 *d_fHyperGeo1F1Info;

extern __managed__ double *d_fPairKStar3dVec;
extern __managed__ BinInfoKStar *d_fPairKStar3dVecInfo;

extern __managed__ double *d_fPairSample4dVec;
extern __managed__ BinInfoSamplePairs *d_fPairSample4dVecInfo;

extern __managed__ double d_fBohrRadius;

extern __managed__ float *dA, *dB;

/*
__device__ double d_gBohrRadius = parallel_gBohrRadiusXiK;
__device__ double d_hbarc = parallel_hbarc;
*/


class ParallelWaveFunction {

public:

  //Constructor, destructor, copy constructor, assignment operator
  ParallelWaveFunction(int aNThreadsPerBlock=512, int aNBlocks=32); //TODO delete this constructor.  Only here for testing
  virtual ~ParallelWaveFunction();

  void LoadBohrRadius(double aRadius);
  //--------------Load the arrays from c++ program dealing with the histograms
  void LoadPairSample4dVec(td4dVec &aPairSample4dVec, BinInfoSamplePairs &aBinInfo);
  void UpdatePairSampleRadii(double aScaleFactor);

  void LoadPairKStar3dVec(td3dVec &aPairKStar3dVec, BinInfoKStar &aBinInfo);

  void LoadLednickyHFunction(td1dVec &aHFunc);

  void LoadGTildeReal(td2dVec &aGTildeReal);
  void LoadGTildeImag(td2dVec &aGTildeImag);

  void LoadHyperGeo1F1Real(td3dVec &aHyperGeo1F1Real);
  void LoadHyperGeo1F1Imag(td3dVec &aHyperGeo1F1Imag);


  //--------------------------------------------------------------------------

  //----------Load the struct objects from the c++ program
  void LoadGTildeInfo(BinInfoGTilde &aBinInfo);
  void LoadHyperGeo1F1Info(BinInfoHyperGeo1F1 &aBinInfo);
  void LoadScattLenInfo(BinInfoScattLen &aBinInfo);
  //------------------------------------------------------

//  double* RunInterpolateWfSquared(td2dVec &aPairs, double aReF0, double aImF0, double aD0);
//TODO EVERYBODY NEEDS aRadiusScale, as in RunInterpolateEntireCfCompletewStaticPairs!!!!!!!!!!!!!!!!!!!!!!!!
  vector<double> RunInterpolateWfSquared(td2dVec &aPairs, double aReF0, double aImF0, double aD0);

  vector<double> RunInterpolateEntireCf(td3dVec &aPairs, double aReF0, double aImF0, double aD0);
  vector<double> RunInterpolateEntireCfComplete(td3dVec &aPairs, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t);

  td2dVec RunInterpolateEntireCfwStaticPairs(int aAnalysisNumber, double aRadiusScale, double aReF0, double aImF0, double aD0);
  td2dVec RunInterpolateEntireCfCompletewStaticPairs(int aAnalysisNumber, double aRadiusScale, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t);

  vector<double> RunInterpolateEntireCfComplete2(int aNSimPairsPerBin, double aKStarMin, double aKStarMax, double aNbinsK, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t);

//-----inline stuff
  void SetNThreadsPerBlock(double aNThreadsPerBlock);
  void SetNBlocks(double aNBlocks);
  void SetNThreadsPerBlockAndNBlocks(double aNThreadsPerBlock, double aNBlocks);


private:
  bool fInterpScattLen;

  int fNThreadsPerBlock;
  int fNBlocks;

  double *fGTildeReal, *fGTildeImag;
  double *fHyperGeo1F1Real, *fHyperGeo1F1Imag;

/*
  BinInfoGTilde *fGTildeInfo;
  BinInfoHyperGeo1F1 *fHyperGeo1F1Info;
*/
  BinInfoSamplePairs fSamplePairsBinInfo;


};


//inline stuff
inline void ParallelWaveFunction::SetNThreadsPerBlock(double aNThreadsPerBlock) {fNThreadsPerBlock = aNThreadsPerBlock;}
inline void ParallelWaveFunction::SetNBlocks(double aNBlocks) {fNBlocks = aNBlocks;}
inline void ParallelWaveFunction::SetNThreadsPerBlockAndNBlocks(double aNThreadsPerBlock, double aNBlocks) {fNThreadsPerBlock = aNThreadsPerBlock; fNBlocks = aNBlocks;}




#endif
