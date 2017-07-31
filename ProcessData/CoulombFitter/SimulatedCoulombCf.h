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

#include "SimulatedPairCollection.h"
class SimulatedPairCollection;

#include "Types.h"

class SimulatedCoulombCf {

public:
  //Constructor, destructor, copy constructor, assignment operator
  SimulatedCoulombCf(vector<tmpAnalysisInfo> &aAnalysesInfo, TString aInterpHistFileBaseName, TString aLednickyHFunctionFileBaseName);
  virtual ~SimulatedCoulombCf();

  bool CheckIfAllOfSameCoulombType();  //i.e., assure all should use Bohr radius
  static double GetBohrRadius(AnalysisType aAnalysisType);
  void SetBohrRadius(double aBohrRadius);

  CoulombType GetCoulombType(AnalysisType aAnalysisType);
  double GetBohrRadius();

  void LoadLednickyHFunctionFile(TString aFileBaseName="~/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");
  void LoadInterpHistFile(TString aFileBaseName);  //TODO should this be a virtual function?

  bool AreParamsSameExcludingLambda(double *aCurrent, double *aNew, int aNEntries);
  void AdjustLambdaParam(td1dVec &aCoulombCf, double aOldLambda, double aNewLambda);

  double GetEta(double aKStar);
  double GetGamowFactor(double aKStar);
  complex<double> GetExpTerm(double aKStar, double aRStar, double aTheta);
  complex<double> BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0);
  double InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0);

  bool CanInterpKStar(double aKStar);
  bool CanInterpRStar(double aRStar);
  bool CanInterpTheta(double aTheta);
  bool CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0);

  void BuildSimPairCollection(double aKStarBinSize, double aMaxBuildKStar, int aNPairsPerKStarBin=16384, bool aUseRandomKStarVectors=true, bool aShareSingleSampleAmongstAll=false);

  double GetFitCfContentCompletewStaticPairs(int aAnalysisNumber, double aKStarMagMin, double aKStarMagMax, double *par);  //TODO!!!!!

  td1dVec GetCoulombParentCorrelation(int aAnalysisNumber, double *aParentCfParams, vector<double> &aKStarBinCenters, CentralityType aCentType=k0010);


  //inline (i.e. simple) functions

  WaveFunction* GetWaveFunctionObject();
  void SetTurnOffCoulomb(bool aTurnOffCoulomb);

  void SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet);
  void SetUseRandomKStarVectors(bool aUseRandomKStarVectors);

  void SetPairKStarNtupleDirLocation(TString aDirLocation);
  void SetPairKStarNtupleFileBaseName(TString aBaseName);

protected:
  int fNAnalyses;
  vector<tmpAnalysisInfo> ftmpAnalysesInfo;
  vector<AnalysisInfo> fAnalysesInfo;

  bool fTurnOffCoulomb;
  bool fIncludeSingletAndTriplet;
  bool fUseRandomKStarVectors;
  td1dVec fCoulombCf;
  double* fCurrentFitParams;

  CoulombType fCoulombType;
  WaveFunction* fWaveFunction;
  double fBohrRadius;

  bool bSimPairCollectionBuilt;
  SimulatedPairCollection *fSimPairCollection;


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

inline void SimulatedCoulombCf::SetPairKStarNtupleDirLocation(TString aDirLocation) {fSimPairCollection->SetPairKStarNtupleDirLocation(aDirLocation);}
inline void SimulatedCoulombCf::SetPairKStarNtupleFileBaseName(TString aBaseName) {fSimPairCollection->SetPairKStarNtupleFileBaseName(aBaseName);}

#endif
