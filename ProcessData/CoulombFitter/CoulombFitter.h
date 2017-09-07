///////////////////////////////////////////////////////////////////////////
// CoulombFitter:                                                        //
///////////////////////////////////////////////////////////////////////////

#ifndef COULOMBFITTER_H
#define COULOMBFITTER_H

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
#include "TH3.h"
#include "THn.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TVector3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TObjectTable.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

#include <omp.h>

#include "ChronoTimer.h"

using std::cout;
using std::endl;
using std::vector;

#include "FitSharedAnalyses.h"
class FitSharedAnalyses;


#include "WaveFunction.h"
class WaveFunction;

#include "SimulatedCoulombCf.h"
class SimulatedCoulombCf;

#include "Interpolator.h"
class Interpolator;

class CoulombFitter {

public:
  //Constructor, destructor, copy constructor, assignment operator
  CoulombFitter(double aMaxFitKStar = 0.3); //TODO delete this constructor.  Only here for testing
  CoulombFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3);
  virtual ~CoulombFitter();

  CoulombType GetCoulombType(AnalysisType aAnalysisType);
  double GetBohrRadius(CoulombType aCoulombType);
  double GetBohrRadius(AnalysisType aAnalysisType);
  void CheckIfAllOfSameCoulombType();

  void LoadLednickyHFunctionFile(TString aFileBaseName="~/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");
  void LoadInterpHistFile(TString aFileBaseName);  //TODO should this be a virtual function?

  void ExtractPairKStar3dVecFromSingleFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill);
  td3dVec BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);

  void WriteRow(ostream &aOutput, vector<double> &aRow);
  void WritePairKStar3dVecFile(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  void WriteAllPairKStar3dVecFiles(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);
  td3dVec BuildPairKStar3dVecFromTxt(TString aFileName);

  void BuildPairKStar4dVecFromTxt(TString aFileBaseName);
  void BuildPairKStar4dVecOnFly(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);

  void BuildPairSample4dVec(int aNPairsPerKStarBin=16384, double aBinSize=0.01);
  void UpdatePairRadiusParameter(double aNewRadius, int aAnalysisNumber);
  void SetUseStaticPairs(bool aUseStaticPairs=true, int aNPairsPerKStarBin=16384, double aBinSize=0.01);

  double GetEta(double aKStar);
  double GetGamowFactor(double aKStar);
  complex<double> GetExpTerm(double aKStar, double aRStar, double aTheta);
  complex<double> BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0);
  double InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0);
  vector<double> InterpolateWfSquaredSerialv2(vector<vector<double> > &aPairs, double aKStarMagMin, double aKStarMagMax, double aReF0, double aImF0, double aD0);

  bool CanInterpKStar(double aKStar);
  bool CanInterpRStar(double aRStar);
  bool CanInterpTheta(double aTheta);
  bool CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0);

  void SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax);

  double GetFitCfContent(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!
  double GetFitCfContentwStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!

  double GetFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!
  double GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!

  double GetFitCfContentSerialv2(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!

  //void CalculateChi2(int &npar, double &chi2, double *par);
  void PrintCurrentParamValues(int aNpar, double* aPar);
  bool AreParamsSame(double *aCurrent, double *aNew, int aNEntries);

  double GetChi2Value(int aKStarBin, TH1* aCfToFit, double aFitVal);
  double GetPmlValue(double aNumContent, double aDenContent, double aCfContent);

  void CalculateChi2PML(int &npar, double &chi2, double *par);  //TODO change default to true when matrices are ready
  vector<double> ApplyMomResCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TH2* aMomResMatrix);
  void CalculateChi2PMLwMomResCorrection(int &npar, double &chi2, double *par);

  void ApplyNonFlatBackgroundCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TF1* aNonFlatBgd);
  void ApplyNormalization(double aNorm, td1dVec &aCf);
  void CalculateFitFunction(int &npar, double &chi2, double *par);


  void CalculateFakeChi2(int &npar, double &chi2, double *par);
  double GetChi2(TH1* aFitHistogram);
  void DoFit();
  TH1* CreateFitHistogram(TString aName, int aAnalysisNumber);
  TH1* CreateFitHistogramSample(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm);
  TH1* CreateFitHistogramSampleComplete(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double aNorm);

  td1dVec GetCoulombResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams, vector<double> &aKStarBinCenters, TH2* aTransformMatrix);


  //inline (i.e. simple) functions
  FitSharedAnalyses* GetFitSharedAnalyses();

  vector<double> GetMinParams();
  vector<double> GetParErrors();

  WaveFunction* GetWaveFunctionObject();
  void SetTurnOffCoulomb(bool aTurnOffCoulomb);

  void SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet);
  void SetUseRandomKStarVectors(bool aUseRandomKStarVectors);

  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType);
  void SetApplyMomResCorrection(bool aApplyMomResCorrection);
  virtual void SetIncludeResidualCorrelations(bool aInclude);

  void SetVerbose(bool aSet);

  double GetChi2();
  int GetNDF();

protected:
  bool fVerbose;
  bool fTurnOffCoulomb;
  bool fInterpHistsLoaded;
  bool fIncludeSingletAndTriplet;
  bool fUseRandomKStarVectors;
  bool fUseStaticPairs;

  bool fApplyNonFlatBackgroundCorrection;
  bool fApplyMomResCorrection;
  bool fIncludeResidualCorrelations;
  bool fResidualsInitiated;
  bool fReturnPrimaryWithResidualsToAnalyses;
  NonFlatBgdFitType fNonFlatBgdFitType;

  int MasterRepeat;

  int fNCalls;  //TODO delete this
  TH1* fFakeCf; //TODO delete this

  FitSharedAnalyses* fFitSharedAnalyses;
  TMinuit* fMinuit;
  int fNAnalyses;
  td3dVec fCorrectedFitVecs;

  SimulatedCoulombCf *fSimCoulombCf;
  bool fAllOfSameCoulombType;
  CoulombType fCoulombType;
  WaveFunction* fWaveFunction;
  double fBohrRadius;

  td4dVec fPairKStar4dVec; //1 3dVec for each of fNAnalyses.  Holds td1dVec = (KStarMag, KStarOut, KStarSide, KStarLong)

  int fNPairsPerKStarBin;
  td1dVec fCurrentRadii;
  td4dVec fPairSample4dVec; //1 3dVec for each of fNAnalyses.  Hold td1dVec = (KStarMag, RStarMag, Theta)
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

  vector<TH1F*> fCfsToFit;
  vector<TF1*> fFits;

  double fMaxFitKStar;
  //vector<double> fMaxFitKStarVec;

  bool fRejectOmega;

  double fChi2;
  double fChi2GlobalMin;

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
  ClassDef(CoulombFitter, 1)
#endif
};


//inline stuff
inline FitSharedAnalyses* CoulombFitter::GetFitSharedAnalyses() {return fFitSharedAnalyses;}

inline vector<double> CoulombFitter::GetMinParams() {return fMinParams;}
inline vector<double> CoulombFitter::GetParErrors() {return fParErrors;}

inline WaveFunction* CoulombFitter::GetWaveFunctionObject() {return fWaveFunction;}
inline void CoulombFitter::SetTurnOffCoulomb(bool aTurnOffCoulomb) {fTurnOffCoulomb = aTurnOffCoulomb; fWaveFunction->SetTurnOffCoulomb(fTurnOffCoulomb);}

inline void CoulombFitter::SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet) {fIncludeSingletAndTriplet = aIncludeSingletAndTriplet;}
inline void CoulombFitter::SetUseRandomKStarVectors(bool aUseRandomKStarVectors) {fUseRandomKStarVectors = aUseRandomKStarVectors;}

inline void CoulombFitter::SetApplyNonFlatBackgroundCorrection(bool aApply) {fApplyNonFlatBackgroundCorrection = aApply;}
inline void CoulombFitter::SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType) {fNonFlatBgdFitType = aNonFlatBgdFitType;}
inline void CoulombFitter::SetApplyMomResCorrection(bool aApplyMomResCorrection) {fApplyMomResCorrection = aApplyMomResCorrection;}
inline void CoulombFitter::SetIncludeResidualCorrelations(bool aInclude) {fIncludeResidualCorrelations = aInclude;}

inline void CoulombFitter::SetVerbose(bool aSet) {fVerbose=aSet;}

inline double CoulombFitter::GetChi2() {return fChi2;}
inline int CoulombFitter::GetNDF() {return fNDF;}
#endif
