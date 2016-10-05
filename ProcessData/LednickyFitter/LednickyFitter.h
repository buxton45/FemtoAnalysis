///////////////////////////////////////////////////////////////////////////
// LednickyFitter:                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef LEDNICKYFITTER_H
#define LEDNICKYFITTER_H

//includes and any constant variable declarations
#include <iostream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>

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

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

//const double hbarc = 0.197327;
const std::complex<double> ImI (0.,1.);

#include <omp.h>

#include "ChronoTimer.h"

using std::cout;
using std::endl;
using std::vector;

#include "FitSharedAnalyses.h"
class FitSharedAnalyses;




class LednickyFitter {

public:
  //Any enum types



  //Constructor, destructor, copy constructor, assignment operator
  LednickyFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3);
  virtual ~LednickyFitter();

  double GetLednickyMomResCorrectedPoint(double aKStar, double* aPar, TH2* aMomResMatrix);
  double ApplyResidualCorrelationToPoint(double aKStar, double* aPar, TH2* aTransformMatrix);

  //void CalculateChi2(int &npar, double &chi2, double *par);
  void CalculateChi2PML(int &npar, double &chi2, double *par);
  void CalculateChi2PMLwMomResCorrection(int &npar, double &chi2, double *par);
  void CalculateChi2PMLwCorrectedCfs(int &npar, double &chi2, double *par);
  void DoFit();
  TF1* CreateFitFunction(TString aName, int aAnalysisNumber);

  vector<double> FindGoodInitialValues();


  //inline (i.e. simple) functions
  FitSharedAnalyses* GetFitSharedAnalyses();

  vector<double> GetMinParams();
  vector<double> GetParErrors();

private:

  FitSharedAnalyses* fFitSharedAnalyses;
  TMinuit* fMinuit;
  int fNAnalyses;
  vector<TH1F*> fCfsToFit;
  vector<TF1*> fFits;

  double fMaxFitKStar;
  //vector<double> fMaxFitKStarVec;

  bool fRejectOmega;

  double fChi2;

  double fEdm, fErrDef;
  int fNvpar, fNparx, fIcstat;

  vector<double> fChi2Vec;



  int fNpFits;
  vector<int> fNpFitsVec;

  int fNDF;

  int fErrFlg;

  vector<double> fMinParams;
  vector<double> fParErrors;


#ifdef __ROOT__
  ClassDef(LednickyFitter, 1)
#endif
};


//inline stuff
inline FitSharedAnalyses* LednickyFitter::GetFitSharedAnalyses() {return fFitSharedAnalyses;}

inline vector<double> LednickyFitter::GetMinParams() {return fMinParams;}
inline vector<double> LednickyFitter::GetParErrors() {return fParErrors;}


#endif
