/* ThermChargedResidual.h */

#ifndef THERMCHARGEDRESIDUAL_H
#define THERMCHARGEDRESIDUAL_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cassert>
#include <complex>

#include "TObjArray.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TVector3.h"

#include "Types.h"
#include "PIDMapping.h"

#include "Interpolator.h"
class Interpolator;


using namespace std;

class ThermChargedResidual {

public:
  ThermChargedResidual(AnalysisType aAnType);
  ThermChargedResidual(const ThermChargedResidual& aRes);
  ThermChargedResidual& operator=(const ThermChargedResidual& aRes);
  virtual ~ThermChargedResidual();

  void SetPartTypes();
  static double GetBohrRadius(AnalysisType aAnalysisType);
  void LoadLednickyHFunctionFile(TString aFileBaseName="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");
  void LoadInterpHistFile(TString aFileBaseName="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/InterpHists", 
                          TString aLednickyHFunctionFileBaseName="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");

  double GetEta(double aKStar);
  double GetGamowFactor(double aKStar);
  complex<double> GetExpTerm(double aKStar, double aRStar, double aTheta);
  complex<double> BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0);
  double InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0);
  double InterpolateWfSquared(TVector3* aKStar3Vec, TVector3* aRStar3Vec, double aReF0, double aImF0, double aD0);

  //inline
  AnalysisType GetResidualType();
  vector<ParticlePDGType> GetPartTypes();

private:
  AnalysisType fResidualType;
  ParticlePDGType fPartType1, fPartType2;
  double fBohrRadius;
  bool fTurnOffCoulomb;

  TFile *fInterpHistFile, *fInterpHistFileLednickyHFunction;

  TH1D* fLednickyHFunctionHist;

  TH2D* fGTildeRealHist;
  TH2D* fGTildeImagHist;

  TH3D* fHyperGeo1F1RealHist;
  TH3D* fHyperGeo1F1ImagHist;

  double fMinInterpKStar, fMinInterpRStar, fMinInterpTheta;
  double fMaxInterpKStar, fMaxInterpRStar, fMaxInterpTheta;

#ifdef __ROOT__
  ClassDef(ThermChargedResidual, 1)
#endif
};

inline AnalysisType ThermChargedResidual::GetResidualType() {return fResidualType;}
inline vector<ParticlePDGType> ThermChargedResidual::GetPartTypes() {return vector<ParticlePDGType>{fPartType1, fPartType2};}


#endif







