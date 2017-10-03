///////////////////////////////////////////////////////////////////////////
// SimpleLednickyFitter:                                                 //
///////////////////////////////////////////////////////////////////////////

#ifndef SIMPLELEDNICKYFITTER_H
#define SIMPLELEDNICKYFITTER_H

//includes and any constant variable declarations
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <ctime>
#include <random>
#include <chrono>
#include <algorithm>
#include <limits>

#include "Faddeeva.hh"

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TVector3.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

//const double hbarc = 0.197327;
//const std::complex<double> ImI (0.,1.);

#include <omp.h>

#include "ChronoTimer.h"

using std::cout;
using std::endl;
using std::vector;

#include "FitSharedAnalyses.h"
class FitSharedAnalyses;

#include "LednickyFitter.h"

class SimpleLednickyFitter {

public:
  //Any enum types

  //Constructor, destructor, copy constructor, assignment operator
  SimpleLednickyFitter(AnalysisType aAnalysisType, CfLite *aCfLite, double aMaxFitKStar = 0.3);
  SimpleLednickyFitter(AnalysisType aAnalysisType, TString aFileLocation, TString aBaseName, double aMaxFitKStar = 0.3);
  SimpleLednickyFitter(AnalysisType aAnalysisType, vector<double> &aSimParams, double aMaxBuildKStar=1.0, double aMaxFitKStar = 0.3);
  virtual ~SimpleLednickyFitter();

  void SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax);
  complex<double> GetStrongOnlyWaveFunction(TVector3* aKStar3Vec, TVector3* aRStar3Vec, vector<double> &aSimParams);
  double GetStrongOnlyWaveFunctionSq(TVector3* aKStar3Vec, TVector3 *aRStar3Vec, vector<double> &aSimParams);
  TH1D* GetSimluatedNumDen(bool aBuildNum, vector<double> &aSimParams, double aMaxBuildKStar=0.5, int aNPairsPerKStarBin = 20000, double aKStarBinSize=0.01);

  TH1D* Get1dHisto(TString FileName, TString HistoName);
  void CreateMinuitParameters();
  void CalculateFitFunction(int &npar, double &chi2, double *par);
  TF1* CreateFitFunction(TString aName);

  void InitializeFitter();  //Called within DoFit
  void DoFit();
  void Finalize();

  void DrawCfWithFit(TPad *aPad);

  //inline (i.e. simple) functions

  vector<double> GetMinParams();
  vector<double> GetParErrors();

  void SetVerbose(bool aSet);

  double GetChi2();
  int GetNDF();
  TMinuit* GetMinuitObject();

protected:
  AnalysisType fAnalysisType;
  CfLite* fCfLite;
  FitType fFitType;
  bool fVerbose;
  TMinuit* fMinuit;
  td1dVec fCorrectedFitVec;

  double fMaxFitKStar;
  int fNbinsXToBuild;
  int fNbinsXToFit;
  double fKStarBinWidth;
  td1dVec fKStarBinCenters;
  //vector<double> fMaxFitKStarVec;

  bool fRejectOmega;

  double fChi2;
  double fChi2GlobalMin;
  vector<double> fChi2Vec;

  double fEdm, fErrDef;
  int fNvpar, fNparx, fIcstat;

  int fNpFits;

  int fNDF;
  int fErrFlg;

  vector<double> fMinParams;
  vector<double> fParErrors;


#ifdef __ROOT__
  ClassDef(SimpleLednickyFitter, 1)
#endif
};


//inline stuff
inline vector<double> SimpleLednickyFitter::GetMinParams() {return fMinParams;}
inline vector<double> SimpleLednickyFitter::GetParErrors() {return fParErrors;}

inline void SimpleLednickyFitter::SetVerbose(bool aSet) {fVerbose=aSet;}

inline double SimpleLednickyFitter::GetChi2() {return fChi2;}
inline int SimpleLednickyFitter::GetNDF() {return fNDF;}

inline TMinuit* SimpleLednickyFitter::GetMinuitObject() {return fMinuit;}
#endif
