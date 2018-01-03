///////////////////////////////////////////////////////////////////////////
// CoulombFitter:                                                        //
///////////////////////////////////////////////////////////////////////////

#ifndef COULOMBFITTER_H
#define COULOMBFITTER_H

//includes and any constant variable declarations

#include "TH3.h"
#include "THn.h"
#include "TVector3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TObjectTable.h"

#include <omp.h>

#include "ChronoTimer.h"

using std::cout;
using std::endl;
using std::vector;

#include "WaveFunction.h"
class WaveFunction;

#include "SimulatedCoulombCf.h"
class SimulatedCoulombCf;

#include "Interpolator.h"
class Interpolator;

#include "LednickyFitter.h"

class CoulombFitter : public LednickyFitter {

public:
  //Constructor, destructor, copy constructor, assignment operator
  CoulombFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3);
  CoulombFitter(AnalysisType aAnalysisType, double aMaxFitKStar = 0.3);
  virtual ~CoulombFitter();

  CoulombType GetCoulombType(AnalysisType aAnalysisType);
  double GetBohrRadius(AnalysisType aAnalysisType);
  void SetCoulombAttributes(AnalysisType aAnalysisType);

  void LoadLednickyHFunctionFile(TString aFileBaseName="~/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");
  void LoadInterpHistFile(TString aFileBaseName, TString aLednickyHFunctionFileBaseName="~/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");  //TODO should this be a virtual function?

  void ExtractPairKStar3dVecFromSingleFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill);
  td3dVec BuildPairKStar3dVecFull(TString aPairKStarNtupleBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);

  void WriteRow(ostream &aOutput, vector<double> &aRow);
  void WritePairKStar3dVecFile(TString aOutputBaseName, TString aPairKStarNtupleBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  void WriteAllPairKStar3dVecFiles(TString aOutputBaseName, TString aPairKStarNtupleBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);
  td3dVec BuildPairKStar3dVecFromTxt(TString aFileName);

  td1dVec BuildPairKStar4dVecFromTxt(TString aFileBaseName);
  void BuildPairKStar4dVecOnFly(TString aPairKStarNtupleBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);

  void BuildPairSample4dVec(int aNPairsPerKStarBin, double aBinSize);  //TODO make this parallel!!!!!
  void BuildPairSample4dVec();
  void UpdatePairRadiusParameter(double aNewRadius, int aAnalysisNumber);


  double GetEta(double aKStar);
  double GetGamowFactor(double aKStar);
  complex<double> GetExpTerm(double aKStar, double aRStar, double aTheta);
  complex<double> BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0);
  double InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0);
  vector<double> InterpolateWfSquaredSerialv2(vector<vector<double> > &aPairs, double aKStarMagMin, double aKStarMagMax, double aReF0, double aImF0, double aD0);

  bool CanInterpKStar(double aKStar);
  bool CanInterpRStar(double aRStar);
  bool CanInterpTheta(double aTheta);
  bool CanInterpAll(double aKStar, double aRStar, double aTheta);

  void SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax);


  double GetFitCfContent(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!
  double GetFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!

  double GetFitCfContentSerialv2(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!

  //void CalculateChi2(int &npar, double &chi2, double *par);
  bool AreParamsSame(double *aCurrent, double *aNew, int aNEntries);

  void CalculateChi2PML(int &npar, double &chi2, double *par);
  void CalculateFitFunction(int &npar, double &chi2, double *par);


  void CalculateFakeChi2(int &npar, double &chi2, double *par);
  double GetChi2(TH1* aFitHistogram);

  TH1* CreateFitHistogram(TString aName, int aAnalysisNumber);
  TH1* CreateFitHistogramSample(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm);
  TH1* CreateFitHistogramSampleComplete(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double aNorm);

  td1dVec GetCoulombResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams, vector<double> &aKStarBinCenters, TH2* aTransformMatrix);

  void InitializeFitter();  //Called within DoFit
  void DoFit();
  void Finalize();  //Send things back to analyses, etc.

  //inline (i.e. simple) functions
  WaveFunction* GetWaveFunctionObject();
  void SetTurnOffCoulomb(bool aTurnOffCoulomb);

  void SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet);
  void SetUseRandomKStarVectors(bool aUseRandomKStarVectors);
  void SetReadPairsFromTxtFiles(bool aRead);
  void SetUseStaticPairs(bool aUseStaticPairs=true);

  void SetPairKStarNtupleBaseName(TString aName, int aNFiles=27);
  void SetPairKStar3dVecBaseName(TString aName);

  void SetNPairsPerKStarBin(int aNPairsPerBin);
  void SetBinSizeKStar(double aBinSize);

//TODO
  double GetChi2();  //Why do I need this, it's defined in LednickyFitter.  Stupid compiler


protected:
  bool fTurnOffCoulomb;
  bool fInterpHistsLoaded;
  bool fIncludeSingletAndTriplet;

  bool fUseRandomKStarVectors;
  bool fReadPairsFromTxtFiles; //if fUseRandomKStarVectors is true, this does nothing
  bool fUseStaticPairs;

  TString fPairKStarNtupleBaseName;
  int fNFilesNtuple;
  TString fPairKStar3dVecBaseName;

  int fNCalls;  //TODO delete this
  TH1* fFakeCf; //TODO delete this


  SimulatedCoulombCf *fSimCoulombCf;
  WaveFunction* fWaveFunction;
  double fBohrRadius;

  int fNPairsPerKStarBin;
  td1dVec fCurrentRadii;

  td4dVec fPairKStar4dVec; //1 3dVec for each of fNAnalyses.  Holds td1dVec = (KStarMag, KStarOut, KStarSide, KStarLong)
                           //Only needed if fUseRandomKStarVectors=false

  BinInfoSamplePairs fSamplePairsBinInfo;  
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


#ifdef __ROOT__
  ClassDef(CoulombFitter, 1)
#endif
};


//inline stuff
inline WaveFunction* CoulombFitter::GetWaveFunctionObject() {return fWaveFunction;}
inline void CoulombFitter::SetTurnOffCoulomb(bool aTurnOffCoulomb) {fTurnOffCoulomb = aTurnOffCoulomb; fWaveFunction->SetTurnOffCoulomb(fTurnOffCoulomb);}

inline void CoulombFitter::SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet) {fIncludeSingletAndTriplet = aIncludeSingletAndTriplet;}
inline void CoulombFitter::SetUseRandomKStarVectors(bool aUseRandomKStarVectors) {fUseRandomKStarVectors = aUseRandomKStarVectors;}
inline void CoulombFitter::SetReadPairsFromTxtFiles(bool aRead) {fReadPairsFromTxtFiles = aRead;}
inline void CoulombFitter::SetUseStaticPairs(bool aUse) {fUseStaticPairs = aUse;}

inline void CoulombFitter::SetPairKStarNtupleBaseName(TString aName, int aNFiles) {fPairKStarNtupleBaseName = aName; fNFilesNtuple = aNFiles;}
inline void CoulombFitter::SetPairKStar3dVecBaseName(TString aName) {fPairKStar3dVecBaseName = aName;}

inline void CoulombFitter::SetNPairsPerKStarBin(int aNPairsPerBin) {fNPairsPerKStarBin = aNPairsPerBin;}
inline void CoulombFitter::SetBinSizeKStar(double aBinSize) {if(fKStarBinWidth==0.) fKStarBinWidth = aBinSize;}

inline double CoulombFitter::GetChi2() {return LednickyFitter::GetChi2();}

#endif
