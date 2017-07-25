///////////////////////////////////////////////////////////////////////////
// SimulatedCoulombCf:                                                   //
///////////////////////////////////////////////////////////////////////////

#ifndef SIMULATEDCOULOMBCF_H
#define SIMULATEDCOULOMBCF_H

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

#include "TH1.h"
#include "TH1D.h"
#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "TVector3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TObjectTable.h"

#include "ChronoTimer.h"

#include <omp.h>

using std::cout;
using std::endl;
using std::vector;

#include "WaveFunction.h"
class WaveFunction;

#include "FitPairAnalysis.h"
class FitPairAnalysis;

#include "Interpolator.h"
class Interpolator;

#include "Types.h"

class SimulatedCoulombCf {

public:
  //Constructor, destructor, copy constructor, assignment operator
  SimulatedCoulombCf(ResidualType aResidualType, TString aInterpHistFileBaseName, TString aLednickyHFunctionFileBaseName);
  virtual ~SimulatedCoulombCf();

  AnalysisType GetDaughterAnalysisType();
  void SetBohrRadius();

  CoulombType GetCoulombType(AnalysisType aAnalysisType);
  double GetBohrRadius();

  void LoadLednickyHFunctionFile(TString aFileBaseName="~/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");
  void LoadInterpHistFile(TString aFileBaseName);  //TODO should this be a virtual function?

  bool AreParamsSameExcludingLambda(double *aCurrent, double *aNew, int aNEntries);
  void AdjustLambdaParam(td1dVec &aCoulombResidualCf, double aOldLambda, double aNewLambda);

  td3dVec BuildPairKStar3dVecFromTxt(double aMaxFitKStar=0.3, TString aFileBaseName="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/PairKStar3dVec_20160610_");
  void BuildPairSample3dVec(double aMaxFitKStar=1.0, int aNPairsPerKStarBin=16384);  //TODO decide appropriate value for aNPairsPerKStarBin
  void UpdatePairRadiusParameters(double aNewRadius);

  double GetEta(double aKStar);
  double GetGamowFactor(double aKStar);
  complex<double> GetExpTerm(double aKStar, double aRStar, double aTheta);
  complex<double> BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0);
  double InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0);

  bool CanInterpKStar(double aKStar);
  bool CanInterpRStar(double aRStar);
  bool CanInterpTheta(double aTheta);
  bool CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0);

  void SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax);
  double GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par);  //TODO!!!!!

  td1dVec GetCoulombParentCorrelation(double *aParentCfParams, vector<double> &aKStarBinCenters, CentralityType aCentType=k0010);


  //inline (i.e. simple) functions

  WaveFunction* GetWaveFunctionObject();
  void SetTurnOffCoulomb(bool aTurnOffCoulomb);

  void SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet);
  void SetUseRandomKStarVectors(bool aUseRandomKStarVectors);

protected:
  ResidualType fResidualType;

  bool fTurnOffCoulomb;
  bool fIncludeSingletAndTriplet;
  bool fUseRandomKStarVectors;
  td1dVec fCoulombCf;
  double* fCurrentFitParams;

  CoulombType fCoulombType;
  WaveFunction* fWaveFunction;
  double fBohrRadius;

  int fNPairsPerKStarBin;
  double fCurrentRadiusParameter;
  td3dVec fPairSample3dVec; //  1 2dVec for each k* bin, holding collection of td1dVec = (KStarMag, RStarMag, Theta)
                            //  Will be initialized by sampling RStar vectors from Gaussian distributions with mu=0 and sigma=1
                            //  When R parameter is updated, I simply scale all RStar magnitudes
  //---------------------------


  //------Histograms----- Note: Should be deleted if vectors are being built

  TFile *fInterpHistFile, *fInterpHistFileLednickyHFunction;

  TH1D* fLednickyHFunctionHist;

  TH2D* fGTildeRealHist;
  TH2D* fGTildeImagHist;

  TH3D* fHyperGeo1F1RealHist;
  TH3D* fHyperGeo1F1ImagHist;

  double fMinInterpKStar, fMinInterpRStar, fMinInterpTheta;
  double fMaxInterpKStar, fMaxInterpRStar, fMaxInterpTheta;
  //---------------------------


#ifdef __ROOT__
  ClassDef(SimulatedCoulombCf, 1)
#endif
};


//inline stuff


inline WaveFunction* SimulatedCoulombCf::GetWaveFunctionObject() {return fWaveFunction;}
inline void SimulatedCoulombCf::SetTurnOffCoulomb(bool aTurnOffCoulomb) {fTurnOffCoulomb = aTurnOffCoulomb; fWaveFunction->SetTurnOffCoulomb(fTurnOffCoulomb);}

inline void SimulatedCoulombCf::SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet) {fIncludeSingletAndTriplet = aIncludeSingletAndTriplet;}
inline void SimulatedCoulombCf::SetUseRandomKStarVectors(bool aUseRandomKStarVectors) {fUseRandomKStarVectors = aUseRandomKStarVectors;}

#endif
