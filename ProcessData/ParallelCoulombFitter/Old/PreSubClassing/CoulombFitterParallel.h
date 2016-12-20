///////////////////////////////////////////////////////////////////////////
// CoulombFitterParallel:                                                //
///////////////////////////////////////////////////////////////////////////

#ifndef COULOMBFITTERPARALLEL_H
#define COULOMBFITTERPARALLEL_H

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

#include "ParallelTypes.h"
#include "ChronoTimer.h"

using std::cout;
using std::endl;
using std::vector;

#include "FitSharedAnalyses.h"
class FitSharedAnalyses;

#include "WaveFunction.h"
class WaveFunction;

#include "ParallelWaveFunction.h"
class ParallelWaveFunction;

class CoulombFitterParallel {

public:
  //Constructor, destructor, copy constructor, assignment operator
  CoulombFitterParallel(); //TODO delete this constructor.  Only here for testing
  CoulombFitterParallel(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3, bool aUseScattLenHists=false);
  virtual ~CoulombFitterParallel();

  CoulombType GetCoulombType(AnalysisType aAnalysisType);
  double GetBohrRadius(CoulombType aCoulombType);
  double GetBohrRadius(AnalysisType aAnalysisType);
  void CheckIfAllOfSameCoulombType();

  int GetBinNumber(double aBinSize, int aNbins, double aValue);
  int GetBinNumber(int aNbins, double aMin, double aMax, double aValue);
  int GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue);

  td3dVec BuildPairKStar3dVec(TString aPairKStarNtupleLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  void ExtractPairKStar3dVecFromSingleFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill);
  td3dVec BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, vector<int> &aNFilesPerSubDir, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  td3dVec BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);

  void WriteRow(ostream &aOutput, vector<double> &aRow);
  void WritePairKStar3dVecFile(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  void WriteAllPairKStar3dVecFiles(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);
  td3dVec BuildPairKStar3dVecFromTxt(TString aFileName);

  void BuildPairKStar4dVecFromTxt(TString aFileBaseName);
  void BuildPairKStar4dVecOnFly(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);

  void LoadLednickyHFunctionFile(TString aFileBaseName="~/Analysis/MathematicaNumericalIntegration/LednickyHFunction");
  void LoadInterpHistFile(TString aFileBaseName);  //TODO delete most of this function after testing
  void MakeNiceScattLenVectors(TString aFileBaseName);
  void MakeOtherVectors(TString aFileBaseName);

  //Note:  Linear, Bilinear, and Trilinear will essentially be copies of TH1::, TH2::, and TH3::Interpolate
  //       Rewriting these allows me more control, and allows me to find where and why error flags are thrown
  int GetInterpLowBin(InterpType aInterpType, InterpAxisType aAxisType, double aVal); //only need low bin, because high bin will be +1;
  double GetInterpLowBinCenter(InterpType aInterpType, InterpAxisType aAxisType, double aVal);
  vector<int> GetRelevantKStarBinNumbers(double aKStarMagMin, double aKStarMagMax);


  //TODO delete the following functions after testing
  double LinearInterpolate(TH1* a1dHisto, double aX);
  double BilinearInterpolate(TH2* a2dHisto, double aX, double aY);
  double BilinearInterpolateVector(vector<vector<double> > &a2dVec, double aX, int aNbinsX, double aMinX, double aMaxX, double aY, int aNbinsY, double aMinY, double aMaxY);
  double TrilinearInterpolate(TH3* a3dHisto, double aX, double aY, double aZ);
  double QuadrilinearInterpolate(THn* a4dHisto, double aT, double aX, double aY, double aZ);
  double ScattLenInterpolate(vector<vector<vector<vector<double> > > > &aScatLen4dSubVec, double aReF0, double aImF0, double aD0, double aKStarMin, double aKStarMax, double aKStarVal);

  double GetEta(double aKStar);
  double GetGamowFactor(double aKStar);
  complex<double> GetExpTerm(double aKStar, double aRStar, double aTheta);
  complex<double> BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0);
  vector<double> InterpolateWfSquaredSerial(vector<vector<double> > &aPairs, double aKStarMagMin, double aKStarMagMax, double aReF0, double aImF0, double aD0);
  //end TODO

  void CreateScattLenSubs(double aReF0, double aImF0, double aD0);
  vector<double> InterpolateWfSquared(vector<vector<double> > &aPairs, double aReF0, double aImF0, double aD0);

  bool CanInterpKStar(double aKStar);
  bool CanInterpRStar(double aRStar);
  bool CanInterpTheta(double aTheta);
  bool CanInterpReF0(double aReF0);
  bool CanInterpImF0(double aImF0);
  bool CanInterpD0(double aD0);
  bool CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0);

  td4dVec Get3dPairs(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par); //One 3dvec for GPU pairs and 1 3dvec for CPU pairs
  td1dVec GetEntireFitCfContent(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par);
  td1dVec GetEntireFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par);
  td1dVec GetEntireFitCfContentComplete2(int aNSimPairsPerBin, double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par);

  double GetFitCfContent(double aKStarMagMin, double aKStarMagMax, double *par);  //TODO!!!!!
  double GetFitCfContentSerial(double aKStarMagMin, double aKStarMagMax, double *par);  //TODO delete this

  //void CalculateChi2(int &npar, double &chi2, double *par);
  bool AreParamsSame(double *aCurrent, double *aNew, int aNEntries);
  void CalculateChi2PML(int &npar, double &chi2, double *par);
  void CalculateChi2(int &npar, double &chi2, double *par);
  void CalculateFakeChi2(int &npar, double &chi2, double *par);
  void DoFit();
  TH1* CreateFitHistogram(TString aName, int aAnalysisNumber);
  TH1* CreateFitHistogramSample(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm);
  TH1* CreateFitHistogramSampleComplete(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s,  double aReF0t, double aImF0t, double aD0t, double aNorm);

  //inline (i.e. simple) functions
  FitSharedAnalyses* GetFitSharedAnalyses();

  vector<double> GetMinParams();
  vector<double> GetParErrors();

  WaveFunction* GetWaveFunctionObject();
  ParallelWaveFunction* GetParallelWaveFunctionObject();

private:
  bool fUseScattLenHists;

  int MasterRepeat;
  BinInfoKStar fPairKStar3dVecInfo;
  BinInfoGTilde fGTildeInfo;
  BinInfoHyperGeo1F1 fHyperGeo1F1Info;
  BinInfoScattLen fScattLenInfo;

  int fNCalls;  //TODO delete this
  TH1* fFakeCf; //TODO delete this

  FitSharedAnalyses* fFitSharedAnalyses;
  TMinuit* fMinuit;
  int fNAnalyses;
  bool fAllOfSameCoulombType;
  CoulombType fCoulombType;
  WaveFunction* fWaveFunction;
  ParallelWaveFunction* fParallelWaveFunction;
  double fBohrRadius;

  vector<vector<vector<double> > > fPairKStar3dVec;
  td4dVec fPairKStar4dVec;
  //---------------------------
  td1dVec fLednickyHFunction;

  td2dVec fGTildeReal;
  td2dVec fGTildeImag;

  td3dVec fHyperGeo1F1Real;
  td3dVec fHyperGeo1F1Imag;

  td4dVec fCoulombScatteringLengthReal;
  td4dVec fCoulombScatteringLengthImag;

  td4dVec fCoulombScatteringLengthRealSub;
  td4dVec fCoulombScatteringLengthImagSub;
  //---------------------------

  //TODO delete the following after testing
  TFile *fInterpHistFile, *fInterpHistFileScatLenReal1, *fInterpHistFileScatLenImag1, *fInterpHistFileScatLenReal2, *fInterpHistFileScatLenImag2, *fInterpHistFileLednickyHFunction;

  TH1D* fLednickyHFunctionHist;

  TH3D* fHyperGeo1F1RealHist;
  TH3D* fHyperGeo1F1ImagHist;

  TH2D* fGTildeRealHist;
  TH2D* fGTildeImagHist;

  THnD* fCoulombScatteringLengthRealHist1;  // 0. < k* < 0.2
  THnD* fCoulombScatteringLengthImagHist1;

  THnD* fCoulombScatteringLengthRealHist2;  // 0.2 < k* < 0.4
  THnD* fCoulombScatteringLengthImagHist2;
  //end TODO--------------------------------------


  double fMinInterpKStar1, fMinInterpKStar2, fMinInterpRStar, fMinInterpTheta, fMinInterpReF0, fMinInterpImF0, fMinInterpD0;
  double fMaxInterpKStar1, fMaxInterpKStar2, fMaxInterpRStar, fMaxInterpTheta, fMaxInterpReF0, fMaxInterpImF0, fMaxInterpD0;
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
  ClassDef(CoulombFitterParallel, 1)
#endif
};


//inline stuff
inline FitSharedAnalyses* CoulombFitterParallel::GetFitSharedAnalyses() {return fFitSharedAnalyses;}

inline vector<double> CoulombFitterParallel::GetMinParams() {return fMinParams;}
inline vector<double> CoulombFitterParallel::GetParErrors() {return fParErrors;}

inline WaveFunction* CoulombFitterParallel::GetWaveFunctionObject() {return fWaveFunction;}
inline ParallelWaveFunction* CoulombFitterParallel::GetParallelWaveFunctionObject() {return fParallelWaveFunction;}

#endif
