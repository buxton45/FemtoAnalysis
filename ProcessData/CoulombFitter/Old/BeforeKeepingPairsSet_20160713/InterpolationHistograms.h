///////////////////////////////////////////////////////////////////////////
// InterpolationHistograms:                                              //
//  Class to write all (9) histograms that I will need to interpolate    //
//  the Coulomb wave function when fitting                               //
///////////////////////////////////////////////////////////////////////////

#ifndef INTERPOLATIONHISTOGRAMS_H
#define INTERPOLATIONHISTOGRAMS_H

//includes and any constant variable declarations
#include <iostream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "TFile.h"
#include "TApplication.h"
#include "TAxis.h"
#include "TROOT.h"

using std::cout;
using std::endl;
using std::vector;

#include "WaveFunction.h"
class WaveFunction;


class InterpolationHistograms {

public:
  InterpolationHistograms(TString aSaveFileName, AnalysisType aAnalysisType);
  virtual ~InterpolationHistograms();

  void SetKStarBinning(int aNbins, double aMin, double aMax);
  void SetRStarBinning(int aNbins, double aMin, double aMax);
  void SetThetaBinning(int aNbins, double aMin, double aMax);
  void SetReF0Binning(int aNbins, double aMin, double aMax);
  void SetImF0Binning(int aNbins, double aMin, double aMax);
  void SetD0Binning(int aNbins, double aMin, double aMax);

  void BuildGamowFactor(int aNbinsK, double aKLow, double aKHigh);
  void BuildLednickyHFunction(int aNbinsK, double aKLow, double aKHigh);
  void BuildHyperGeo1F1Histograms(int aNbinsK, double aKLow, double aKHigh, int aNbinsR, double aRLow, double aRHigh, int aNbinsTheta, double aThetaLow, double aThetaHigh);
  void BuildGTildeHistograms(int aNbinsK, double aKLow, double aKHigh, int aNbinsR, double aRLow, double aRHigh);
  void BuildExpTermHistograms(int aNbinsK, double aKLow, double aKHigh, int aNbinsR, double aRLow, double aRHigh, int aNbinsTheta, double aThetaLow, double aThetaHigh);
  void BuildScatteringLengthHistograms(int aNbinsK, double aKLow, double aKHigh, int aNbinsReF0, double aReF0Low, double aReF0High, int aNbinsImF0, double aImF0Low, double aImF0High, int aNbinsD0, double aD0Low, double aD0High);

  void BuildAndSaveSplitScatteringLengthHistograms();
  void BuildAndSaveAllOthers();
  void BuildAndSaveAll();  //DO NOT USE THIS UNLESS SCATTERING LENGTH HISTOGRAMS ARE VERY SMALL!!!!!!!!!
  void BuildAndSaveLednickyHFunction();


  //TODO Could probably combine all of these into one master function, definitely faster, but not sure how much
  //Obviously, BuildHyperGeo1F1Histograms and BuildExpTermHistograms fit together well, but all could easily fit together



private:
  TString fSaveFileName;
  AnalysisType fAnalysisType;
  TFile* fSaveFile1;
  TFile* fSaveFile2;
  TFile* fSaveFile3;

  TFile *fSaveFileReal1, *fSaveFileReal2, *fSaveFileImag1, *fSaveFileImag2;

  int fNbinsK, fNbinsR, fNbinsTheta, fNbinsReF0, fNbinsImF0, fNbinsD0;
  double fKStarMin, fRStarMin, fThetaMin, fReF0Min, fImF0Min, fD0Min;
  double fKStarMax, fRStarMax, fThetaMax, fReF0Max, fImF0Max, fD0Max;

  WaveFunction* fWaveFunction;
  MathematicaSession* fSession;

  TH1D* fGamowFactor;

  TH1D* fLednickyHFunction;

  TH3D* fHyperGeo1F1Real;
  TH3D* fHyperGeo1F1Imag;

  TH2D* fGTildeReal;
  TH2D* fGTildeImag;

  TH3D* fExpTermReal;
  TH3D* fExpTermImag;

  THnD* fCoulombScatteringLengthReal;
  THnD* fCoulombScatteringLengthImag;


#ifdef __ROOT__
  ClassDef(InterpolationHistograms, 1)
#endif
};


//inline stuff


#endif



