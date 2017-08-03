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

#include "SimpleChargedResidualCf.h"
class SimpleChargedResidualCf;

using namespace std;

class ResidualCollection {

public:
  ResidualCollection(AnalysisType aAnalysisType, td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping, CentralityType aCentType);
  virtual ~ResidualCollection();
  void BuildStandardCollection(td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping, CentralityType aCentType);
  int GetNeutralIndex(AnalysisType aResidualType);
  td1dVec GetNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams);
  td1dVec GetTransformedNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams);
  td1dVec CombinePrimaryWithResiduals(double *aCfParams, td1dVec &aPrimaryCf);

//----Inline functins
  vector<NeutralResidualCf> GetNeutralCollection();
  vector<SimpleChargedResidualCf> GetChargedCollection();

protected:
AnalysisType fAnalysisType;
vector<NeutralResidualCf> fNeutralCfCollection;
vector<SimpleChargedResidualCf> fChargedCfCollection;


#ifdef __ROOT__
  ClassDef(ResidualCollection, 1)
#endif
};

inline vector<NeutralResidualCf> ResidualCollection::GetNeutralCollection() {return fNeutralCfCollection;}
inline vector<SimpleChargedResidualCf> ResidualCollection::GetChargedCollection() {return fChargedCfCollection;}


#endif
