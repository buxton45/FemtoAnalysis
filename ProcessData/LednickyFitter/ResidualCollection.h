/* ResidualCollection.h */

#ifndef RESIDUALCOLLECTION_H
#define RESIDUALCOLLECTION_H

#include <cassert>
#include <iostream>
#include <iomanip>

#include "TH2.h"


#include "Types.h"
#include "Types_LambdaValues.h"

#include "NeutralResidualCf.h"
class NeutralResidualCf;

#include "SimpleChargedResidualCf.h"
class SimpleChargedResidualCf;

using namespace std;

class ResidualCollection {

public:
  ResidualCollection(AnalysisType aAnalysisType, IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping, CentralityType aCentType);
  virtual ~ResidualCollection();
  void BuildStandardCollection(td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping, CentralityType aCentType);
  void SetChargedResidualsType(TString aFileDirectory="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/", ChargedResidualsType aChargedResidualsType=kUseXiDataAndCoulombOnlyInterp);
  void SetRadiusFactorForSigStResiduals(double aFactor=1.);
  int GetNeutralIndex(AnalysisType aResidualType);
  td1dVec GetNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams);
  td1dVec GetTransformedNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams);
  td1dVec CombinePrimaryWithResiduals(double *aCfParams, td1dVec &aPrimaryCf);

  void SetUsemTScalingOfRadii(double aPower=-0.5);

//----Inline functins
  vector<NeutralResidualCf> GetNeutralCollection();
  vector<SimpleChargedResidualCf> GetChargedCollection();

protected:
AnalysisType fAnalysisType;
IncludeResidualsType fIncludeResidualsType;
ResPrimMaxDecayType fResPrimMaxDecayType;
vector<NeutralResidualCf> fNeutralCfCollection;
vector<SimpleChargedResidualCf> fChargedCfCollection;


#ifdef __ROOT__
  ClassDef(ResidualCollection, 1)
#endif
};

inline vector<NeutralResidualCf> ResidualCollection::GetNeutralCollection() {return fNeutralCfCollection;}
inline vector<SimpleChargedResidualCf> ResidualCollection::GetChargedCollection() {return fChargedCfCollection;}


#endif
