/* NeutralResidualCf.h */

#ifndef NEUTRALRESIDUALCF_H
#define NEUTRALRESIDUALCF_H

#include <cassert>

#include "TH2.h"


#include "Types.h"
#include "LednickyFitter.h"

using namespace std;

class NeutralResidualCf {

public:
  NeutralResidualCf(AnalysisType aResidualType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters);
  virtual ~NeutralResidualCf();

  td1dVec GetNeutralResidualCorrelation(double *aParentCfParams);
  td1dVec GetTransformedNeutralResidualCorrelation(double *aParentCfParams);
  TH1D* GetNeutralResidualCorrelationHistogram(double *aParentCfParams, TString aTitle);
  TH1D* GetTransformedNeutralResidualCorrelationHistogram(double *aParentCfParams, TString aTitle);





protected:
AnalysisType fResidualType;
TH2D* fTransformMatrix;
td1dVec fKStarBinCenters;
td1dVec fResCf;
td1dVec fTransformedResCf;



#ifdef __ROOT__
  ClassDef(NeutralResidualCf, 1)
#endif
};



#endif
