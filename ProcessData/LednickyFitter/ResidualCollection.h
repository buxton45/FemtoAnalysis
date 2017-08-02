/* ResidualCollection.h */

#ifndef RESIDUALCOLLECTION_H
#define RESIDUALCOLLECTION_H

#include <cassert>
#include <iostream>
#include <iomanip>

#include "TH2.h"


#include "Types.h"

#include "NeutralResidualCf.h"
class NeutralResidualCf;

//#include "ChargedResidualCf.h"
//class ChargedResidualCf;

using namespace std;

class ResidualCollection {

public:
  ResidualCollection(AnalysisType aAnalysisType, td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping);
  virtual ~ResidualCollection();
  void BuildStandardCollection(td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping);
  int GetNeutralIndex(AnalysisType aResidualType);
  td1dVec GetNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams);
  td1dVec GetTransformedNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams);
  td1dVec CombinePrimaryWithResiduals(double *aCfParams, td1dVec &aPrimaryCf);



protected:
AnalysisType fAnalysisType;
vector<NeutralResidualCf> fNeutralCfCollection;
//vector<ChargedResidualCf> fChargedCfCollection;


#ifdef __ROOT__
  ClassDef(ResidualCollection, 1)
#endif
};



#endif
