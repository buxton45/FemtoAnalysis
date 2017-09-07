///////////////////////////////////////////////////////////////////////////
// GeneralFitter:                                                        //
//                Instead of having separate LednickyFitter and          //
// CoulombFitter, this GeneralFitter will derive from LednickyFitter and //
// include all functionality of CoulombFitter
///////////////////////////////////////////////////////////////////////////

#ifndef GENERALFITTER_H
#define GENERALFITTER_H


#include "TH3.h"
#include "THn.h"
#include "TVector3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TObjectTable.h"


#include "WaveFunction.h"
class WaveFunction;

#include "LednickyFitter.h"

class GeneralFitter : public LednickyFitter {

public:
  //Constructor, destructor, copy constructor, assignment operator
//  GeneralFitter(); //TODO delete this constructor.  Only here for testing
  GeneralFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3);
  virtual ~GeneralFitter();

  CoulombType GetCoulombType(AnalysisType aAnalysisType);
  double GetBohrRadius(CoulombType aCoulombType);
  double GetBohrRadius(AnalysisType aAnalysisType);
  void CheckIfAllOfSameCoulombType();

  void LoadLednickyHFunctionFile(TString aFileBaseName="~/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/LednickyHFunction");
  void LoadInterpHistFile(TString aFileBaseName);  //TODO should this be a virtual function?

  int GetBinNumber(double aBinSize, int aNbins, double aValue);
  int GetBinNumber(int aNbins, double aMin, double aMax, double aValue);
  int GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue);

  void ExtractPairKStar3dVecFromSingleFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill);
  td3dVec BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);

  void WriteRow(ostream &aOutput, vector<double> &aRow);
  void WritePairKStar3dVecFile(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  void WriteAllPairKStar3dVecFiles(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);
  td3dVec BuildPairKStar3dVecFromTxt(TString aFileName);

  void BuildPairKStar4dVecFromTxt(TString aFileBaseName);
  void BuildPairKStar4dVecOnFly(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);

  void BuildPairSample4dVec(int aNPairsPerKStarBin=16384);
  void UpdatePairRadiusParameters(double aNewRadius);
  void SetUseStaticPairs(bool aUseStaticPairs=true, int aNPairsPerKStarBin=16384);
  virtual void SetIncludeResidualCorrelations(bool aInclude);

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
  vector<double> InterpolateWfSquaredSerialv2(vector<vector<double> > &aPairs, double aKStarMagMin, double aKStarMagMax, double aReF0, double aImF0, double aD0);

  bool CanInterpKStar(double aKStar);
  bool CanInterpRStar(double aRStar);
  bool CanInterpTheta(double aTheta);
  bool CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0);

  void SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax);

  double GetFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!
  double GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!

  double GetFitCfContentSerialv2(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!

  td1dVec GetCoulombResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams, vector<double> &aKStarBinCenters, TH2* aTransformMatrix);

  bool AreParamsSame(double *aCurrent, double *aNew, int aNEntries);
  double* AdjustLambdaParam(double *aParamSet, double aNewLambda, int aNEntries);
  void CalculateFitFunction(int &npar, double &chi2, double *par);

  void CalculateFakeChi2(int &npar, double &chi2, double *par);

  void DoFit();

  TH1* CreateFitHistogram(TString aName, int aAnalysisNumber);





  //inline (i.e. simple) functions

  WaveFunction* GetWaveFunctionObject();
  void SetTurnOffCoulomb(bool aTurnOffCoulomb);

  void SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet);
  void SetUseRandomKStarVectors(bool aUseRandomKStarVectors);

protected:
  GeneralFitterType fGeneralFitterType;

  bool fTurnOffCoulomb;
  bool fInterpHistsLoaded;
  bool fIncludeSingletAndTriplet;
  bool fUseRandomKStarVectors;
  bool fUseStaticPairs;

  int MasterRepeat;

  int fNCalls;  //TODO delete this
  TH1* fFakeCf; //TODO delete this

  bool fAllOfSameCoulombType;
  CoulombType fCoulombType;
  WaveFunction* fWaveFunction;
  double fBohrRadius;

  td4dVec fPairKStar4dVec; //1 3dVec for each of fNAnalyses.  Holds td1dVec = (KStarMag, KStarOut, KStarSide, KStarLong)

  int fNPairsPerKStarBin;
  double fCurrentRadiusParameter;
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
  ClassDef(GeneralFitter, 1)
#endif
};


//inline stuff


inline WaveFunction* GeneralFitter::GetWaveFunctionObject() {return fWaveFunction;}
inline void GeneralFitter::SetTurnOffCoulomb(bool aTurnOffCoulomb) {fTurnOffCoulomb = aTurnOffCoulomb; fWaveFunction->SetTurnOffCoulomb(fTurnOffCoulomb);}

inline void GeneralFitter::SetIncludeSingletAndTriplet(bool aIncludeSingletAndTriplet) {fIncludeSingletAndTriplet = aIncludeSingletAndTriplet;}
inline void GeneralFitter::SetUseRandomKStarVectors(bool aUseRandomKStarVectors) {fUseRandomKStarVectors = aUseRandomKStarVectors;}

#endif
