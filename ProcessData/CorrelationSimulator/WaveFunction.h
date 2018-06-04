///////////////////////////////////////////////////////////////////////////
// WaveFunction:                                                         //
//  Generates a wave function including both Coulomb and strong effects  //
///////////////////////////////////////////////////////////////////////////

#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include <iostream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <limits>  //so I can use really small number to test for 0 value
#include <cassert>

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

#include "Math/SpecFuncMathCore.h"
#include "TVector3.h"

#include "MathematicaSession.h"
class MathematicaSession;

#include "Types.h"

//const std::complex<double> ImI (0.,1.);

class WaveFunction {

public:
  //Constructor, destructor, copy constructor, assignment operator
  WaveFunction();
  virtual ~WaveFunction();

  void SetCurrentAnalysisType(AnalysisType aAnalysisType);
  static double GetBohrRadius(AnalysisType aAnalysisType);
  void SetCurrentBohrRadius(AnalysisType aAnalysisType);
  void SetCurrentBohrRadius(double aBohrRadius);

  complex<double> GetTest(complex<double> aCmplx);

  double GetEta(double aKStar);
  double GetLowerCaseXi(TVector3* aKStar3Vec, TVector3* aRStar3Vec);
  double GetLowerCaseXi(double aKStarMag, double aRStarMag, double aTheta);

  double GetCoulombPhaseShift(double aKStar, double aL);
  double GetCoulombSWavePhaseShift(double aKStar);

  double GetGamowFactor(double aKStar);
  double GetLednickyHFunction(double aKStar);
  complex<double> GetLednickyChiFunction(double aKStar);
  complex<double> GetScatteringLength(double aKStar, double aReF0, double aImF0, double aD0);
  complex<double> GetGTilde(double aKStar, double aRStar);
  complex<double> GetGTildeConjugate(double aKStar, double aRStar);
  complex<double> GetWaveFunction(TVector3* aKStar3Vec, TVector3* aRStar3Vec, double aReF0, double aImF0, double aD0);
  complex<double> GetWaveFunction(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0);


  //inline (i.e. simple) functions
  MathematicaSession* GetSession();
  void SetTurnOffCoulomb(bool aTurnOffCoulomb);
  double GetCurrentBohrRadius();

private:
  MathematicaSession* fSession;
  AnalysisType fCurrentAnalysisType;
  double fCurrentBohrRadius;
  bool fTurnOffCoulomb;





#ifdef __ROOT__
  ClassDef(WaveFunction, 1)
#endif
};


//inline stuff
inline MathematicaSession* WaveFunction::GetSession() {return fSession;}
inline void WaveFunction::SetTurnOffCoulomb(bool aTurnOffCoulomb) {fTurnOffCoulomb = aTurnOffCoulomb;}
inline double WaveFunction::GetCurrentBohrRadius() {return fCurrentBohrRadius;}

#endif
