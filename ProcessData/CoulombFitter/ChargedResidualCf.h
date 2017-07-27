///////////////////////////////////////////////////////////////////////////
// ChargedResidualCf:                                                    //
///////////////////////////////////////////////////////////////////////////

#ifndef CHARGEDRESIDUALCF_H
#define CHARGEDRESIDUALCF_H

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

#include "Types.h"

class ChargedResidualCf {

public:
  //Constructor, destructor, copy constructor, assignment operator
  ChargedResidualCf(AnalysisType aResidualType, TString aInterpHistFileBaseName, TString aLednickyHFunctionFileBaseName, int aTransRebin=2, TString aTransformMatricesLocation = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/TransformMatrices_Mix5.root");
  virtual ~ChargedResidualCf();

  AnalysisType GetDaughterAnalysisType();
  void LoadTransformMatrix(int aRebin, TString aFileLocation);
  void SetBohrRadius();

  CoulombType GetCoulombType(AnalysisType aAnalysisType);
  double GetBohrRadius();

  void LoadLednickyHFunctionFile(TString aFileBaseName="~/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");
  void LoadInterpHistFile(TString aFileBaseName);  //TODO should this be a virtual function?

  int GetBinNumber(double aBinSize, int aNbins, double aValue);
  int GetBinNumber(int aNbins, double aMin, double aMax, double aValue);
  int GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue);

  bool AreParamsSameExcludingLambda(double *aCurrent, double *aNew, int aNEntries);
  void AdjustLambdaParam(td1dVec &aCoulombResidualCf, double aOldLambda, double aNewLambda);

  td3dVec BuildPairKStar3dVecFromTxt(double aMaxFitKStar=0.3, TString aFileBaseName="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/PairKStar3dVec_20160610_");
  void BuildPairSample3dVec(double aMaxFitKStar=1.0, int aNPairsPerKStarBin=16384);  //TODO decide appropriate value for aNPairsPerKStarBin
  void UpdatePairRadiusParameters(double aNewRadius);

  //Note:  Linear, Bilinear, and Trilinear will essentially be copies of TH1::, TH2::, and TH3::Interpolate
  //       Rewriting these allows me more control, and allows me to find where and why error flags are thrown
  double LinearInterpolate(TH1* a1dHisto, double aX);
  double BilinearInterpolate(TH2* a2dHisto, double aX, double aY);
  double BilinearInterpolateVector(vector<vector<double> > &a2dVec, double aX, int aNbinsX, double aMinX, double aMaxX, double aY, int aNbinsY, double aMinY, double aMaxY);
  double TrilinearInterpolate(TH3* a3dHisto, double aX, double aY, double aZ);
  double QuadrilinearInterpolate(THn* a4dHisto, double aT, double aX, double aY, double aZ);

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
  td1dVec GetExpXiData(double aMaxKStar=1.0, CentralityType aCentType=k0010);
  double GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par);  //TODO!!!!!

  td1dVec GetCoulombParentCorrelation(double *aParentCfParams, vector<double> &aKStarBinCenters, bool aUseExpXiData=false, CentralityType aCentType=k0010);
  td1dVec GetCoulombResidualCorrelation(double *aParentCfParams, vector<double> &aKStarBinCenters, bool aUseExpXiData=false, CentralityType aCentType=k0010);


  //inline (i.e. simple) functions

  WaveFunction* GetWaveFunctionObject();
  void SetTurnOffCoulomb(bool aTurnOffCoulomb);

  void SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet);
  void SetUseRandomKStarVectors(bool aUseRandomKStarVectors);

protected:
  AnalysisType fResidualType;
  TH2D* fTransformMatrix;

  bool fTurnOffCoulomb;
  bool fIncludeSingletAndTriplet;
  bool fUseRandomKStarVectors;
  bool fUseExpXiData;
  td1dVec fExpXiData;
  td1dVec fCoulombCf;
  td1dVec fCoulombResidualCf;  //The above, but run through the transform matrix
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
  ClassDef(ChargedResidualCf, 1)
#endif
};


//inline stuff


inline WaveFunction* ChargedResidualCf::GetWaveFunctionObject() {return fWaveFunction;}
inline void ChargedResidualCf::SetTurnOffCoulomb(bool aTurnOffCoulomb) {fTurnOffCoulomb = aTurnOffCoulomb; fWaveFunction->SetTurnOffCoulomb(fTurnOffCoulomb);}

inline void ChargedResidualCf::SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet) {fIncludeSingletAndTriplet = aIncludeSingletAndTriplet;}
inline void ChargedResidualCf::SetUseRandomKStarVectors(bool aUseRandomKStarVectors) {fUseRandomKStarVectors = aUseRandomKStarVectors;}

#endif
