/* ParallelSharedResidualCollection.h */

#ifndef PARALLELSHAREDRESIDUALCOLLECTION_H
#define PARALLELSHAREDRESIDUALCOLLECTION_H

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

using std::cout;
using std::endl;
using std::vector;

extern __managed__ BinInfoTransformMatrix* d_fBinInfoTransformMatrix;
extern __managed__ double* d_fTransformMatrices;


class ParallelSharedResidualCollection {

public:

  //Constructor, destructor, copy constructor, assignment operator
  ParallelSharedResidualCollection(int aNThreadsPerBlock=512, int aNBlocks=32); //TODO delete this constructor.  Only here for testing
  virtual ~ParallelSharedResidualCollection();

  void LoadTransformMatrix(td2dVec &aTransformMatrix, BinInfoTransformMatrix &aBinInfo);
  void SetOffsets();
  void BuildDeviceTransformMatrices();

  int GetOffset(AnalysisType aAnType, AnalysisType aResType, CentralityType aCentType);
  td2dVec GetTransformedResidualCorrelation(AnalysisType aAnType, CentralityType aCentType, td2dVec &aCfs);

private:
  int fNThreadsPerBlock;
  int fNBlocks;

  vector<BinInfoTransformMatrix> fBinInfoTransformMatrices;
  td3dVec fTransformMatrices;


};


//inline stuff





#endif
