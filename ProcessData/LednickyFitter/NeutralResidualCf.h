/* NeutralResidualCf.h */

#ifndef NEUTRALRESIDUALCF_H
#define NEUTRALRESIDUALCF_H

#include <cassert>
#include <iostream>

#include "Faddeeva.hh"

#include "TH2.h"
#include "TMath.h"

#include "Types.h"

using namespace std;

class NeutralResidualCf {

public:
  NeutralResidualCf(AnalysisType aResidualType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters);
  virtual ~NeutralResidualCf();

  static double GetLednickyF1(double z);
  static double GetLednickyF2(double z);
  static double LednickyEq(double *x, double *par);
  static TH1D* Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle = "tCf");

  td1dVec GetNeutralResidualCorrelation(double *aParentCfParams);
  td1dVec GetTransformedNeutralResidualCorrelation(double *aParentCfParams);
  TH1D* GetNeutralResidualCorrelationHistogram(double *aParentCfParams, TString aTitle);
  TH1D* GetTransformedNeutralResidualCorrelationHistogram(double *aParentCfParams, TString aTitle);

  double* AdjustLambdaParam(double *aParamSet, double aNewLambda, int aNEntries=6);
  td1dVec GetContributionToFitCf(double *aParams);  //Note: aParams[0] should be OverallLambda!

  //inline
  AnalysisType GetResidualType();


protected:
AnalysisType fResidualType;
double fLambdaFactor;
TH2D* fTransformMatrix;
td1dVec fKStarBinCenters;
td1dVec fResCf;
td1dVec fTransformedResCf;



#ifdef __ROOT__
  ClassDef(NeutralResidualCf, 1)
#endif
};

inline AnalysisType NeutralResidualCf::GetResidualType() {return fResidualType;}

#endif
