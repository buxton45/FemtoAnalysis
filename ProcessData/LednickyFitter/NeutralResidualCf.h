/* NeutralResidualCf.h */

#ifndef NEUTRALRESIDUALCF_H
#define NEUTRALRESIDUALCF_H

#include <cassert>
#include <iostream>
#include <math.h>

#include <omp.h>
#include "ChronoTimer.h"

#include "Faddeeva.hh"

#include "TH2.h"
#include "TMath.h"

#include "Types.h"
#include "Types_LambdaValues.h"

using namespace std;

class NeutralResidualCf {

public:
  NeutralResidualCf(AnalysisType aResidualType, IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters);
  virtual ~NeutralResidualCf();

  static double GetLednickyF1(double z);
  static double GetLednickyF2(double z);
  static double LednickyEq(double *x, double *par);
  static TH1D* Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle = "tCf");
  void SetDaughtersAndMothers();

  td1dVec GetNeutralResidualCorrelation(double *aCfParams);
  td1dVec GetTransformedNeutralResidualCorrelation(double *aCfParams);
  TH1D* GetNeutralResidualCorrelationHistogram(double *aCfParams, TString aTitle);
  TH1D* GetTransformedNeutralResidualCorrelationHistogram(double *aCfParams, TString aTitle);

  double* AdjustLambdaParam(double *aParamSet, double aNewLambda, int aNEntries=6);
  td1dVec GetContributionToFitCf(double *aParamsOverall);  //Note: aParams[0] should be OverallLambda!
  TH1D* GetResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double *aParamsOverall);
  TH1D* GetTransformedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double *aParamsOverall);

  void SetUsemTScalingOfRadii(AnalysisType aParentAnType, double aPower=-0.5);

  //inline
  AnalysisType GetResidualType();
  double GetLambdaFactor();
  TH2D* GetTransformMatrix();

  ParticlePDGType GetMotherType1();
  ParticlePDGType GetDaughterType1();
  ParticlePDGType GetMotherType2();
  ParticlePDGType GetDaughterType2();

  void SetRadiusFactor(double aFactor);

protected:
  AnalysisType fResidualType;
  IncludeResidualsType fIncludeResidualsType;
  ResPrimMaxDecayType fResPrimMaxDecayType;
  ParticlePDGType fDaughterType1, fMotherType1;
  ParticlePDGType fDaughterType2, fMotherType2;
  double fLambdaFactor;
  TH2D* fTransformMatrix;
  td1dVec fKStarBinCenters;
  td1dVec fResCf;
  td1dVec fTransformedResCf;

  double fRadiusFactor;  //Allow me to give SigSt different radii



#ifdef __ROOT__
  ClassDef(NeutralResidualCf, 1)
#endif
};

inline AnalysisType NeutralResidualCf::GetResidualType() {return fResidualType;}
inline double NeutralResidualCf::GetLambdaFactor() {return fLambdaFactor;}
inline TH2D* NeutralResidualCf::GetTransformMatrix() {return fTransformMatrix;}

inline ParticlePDGType NeutralResidualCf::GetMotherType1() {return fMotherType1;}
inline ParticlePDGType NeutralResidualCf::GetDaughterType1() {return fDaughterType1;}
inline ParticlePDGType NeutralResidualCf::GetMotherType2() {return fMotherType2;}
inline ParticlePDGType NeutralResidualCf::GetDaughterType2() {return fDaughterType2;}

inline void NeutralResidualCf::SetRadiusFactor(double aFactor) {fRadiusFactor = aFactor;}
#endif
