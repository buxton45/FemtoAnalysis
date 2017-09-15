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
  void SetDaughtersAndMothers();

  td1dVec GetNeutralResidualCorrelation(double *aParentCfParams);
  td1dVec GetTransformedNeutralResidualCorrelation(double *aParentCfParams);
  TH1D* GetNeutralResidualCorrelationHistogram(double *aParentCfParams, TString aTitle);
  TH1D* GetTransformedNeutralResidualCorrelationHistogram(double *aParentCfParams, TString aTitle);

  double* AdjustLambdaParam(double *aParamSet, double aNewLambda, int aNEntries=6);
  td1dVec GetContributionToFitCf(double *aParams);  //Note: aParams[0] should be OverallLambda!

  //inline
  AnalysisType GetResidualType();
  TH1D* GetNeutralResidualCorrelationHistogram(TString aTitle="fResCf");
  TH1D* GetTransformedNeutralResidualCorrelationHistogram(TString aTitle="fTransformedResCf");
  double GetLambdaFactor();
  TH2D* GetTransformMatrix();

  ParticlePDGType GetMotherType1();
  ParticlePDGType GetDaughterType1();
  ParticlePDGType GetMotherType2();
  ParticlePDGType GetDaughterType2();

protected:
  AnalysisType fResidualType;
  ParticlePDGType fDaughterType1, fMotherType1;
  ParticlePDGType fDaughterType2, fMotherType2;
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
inline TH1D* NeutralResidualCf::GetNeutralResidualCorrelationHistogram(TString aTitle) {return Convert1dVecToHist(fResCf, fKStarBinCenters, aTitle);}
inline TH1D* NeutralResidualCf::GetTransformedNeutralResidualCorrelationHistogram(TString aTitle) {return Convert1dVecToHist(fTransformedResCf, fKStarBinCenters, aTitle);}
inline double NeutralResidualCf::GetLambdaFactor() {return fLambdaFactor;}
inline TH2D* NeutralResidualCf::GetTransformMatrix() {return fTransformMatrix;}

inline ParticlePDGType NeutralResidualCf::GetMotherType1() {return fMotherType1;}
inline ParticlePDGType NeutralResidualCf::GetDaughterType1() {return fDaughterType1;}
inline ParticlePDGType NeutralResidualCf::GetMotherType2() {return fMotherType2;}
inline ParticlePDGType NeutralResidualCf::GetDaughterType2() {return fDaughterType2;}
#endif
