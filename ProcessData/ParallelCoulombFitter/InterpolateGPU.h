///////////////////////////////////////////////////////////////////////////
// InterpolateGPU:                                                        //
///////////////////////////////////////////////////////////////////////////

#ifndef INTERPOLATEGPU_H
#define INTERPOLATEGPU_H

//includes and any constant variable declarations
#include <stdio.h>
#include <iostream>
#include <vector>

#include "timer.h"
#include "utils.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "Types.h"
#include "ParallelTypes.h"

using std::cout;
using std::endl;
using std::vector;

extern __managed__ double *d_fGTildeReal, *d_fGTildeImag;
extern __managed__ BinInfoGTilde *d_fGTildeInfo;
extern __managed__ BinInfoHyperGeo1F1 *d_fHyperGeo1F1Info;
extern __managed__ BinInfoScattLen *d_fScattLenInfo;

class InterpolateGPU {

public:
  //Any enum types



  //Constructor, destructor, copy constructor, assignment operator
  InterpolateGPU(int aNThreadsPerBlock=10, int aNBlocks=1000); //TODO delete this constructor.  Only here for testing
  virtual ~InterpolateGPU();

  //--------------Load the arrays from c++ program dealing with the histograms

  void LoadGTildeReal(td2dVec &aGTildeReal);
  void LoadGTildeImag(td2dVec &aGTildeImag);
  void LoadGTildeInfo(BinInfoGTilde &aBinInfo);

  vector<double> RunBilinearInterpolate(vector<vector<double> > &aPairsIn);
//  double* RunBilinearInterpolate(double* host_out, double* aPairsIn, double* a2dVecIn);


private:
  int fNThreadsPerBlock;
  int fNBlocks;

};


//inline stuff

#endif
