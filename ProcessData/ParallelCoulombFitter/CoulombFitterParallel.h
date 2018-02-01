///////////////////////////////////////////////////////////////////////////
// CoulombFitterParallel:                                                //
///////////////////////////////////////////////////////////////////////////

#ifndef COULOMBFITTERPARALLEL_H
#define COULOMBFITTERPARALLEL_H

//includes and any constant variable declarations
#include "CoulombFitter.h"
#include "ParallelTypes.h"


using std::cout;
using std::endl;
using std::vector;

#include "FitSharedAnalyses.h"
class FitSharedAnalyses;

#include "WaveFunction.h"
class WaveFunction;

#include "ParallelWaveFunction.h"
class ParallelWaveFunction;

class CoulombFitterParallel : public CoulombFitter {

public:
  //Constructor, destructor, copy constructor, assignment operator
//  CoulombFitterParallel(); //TODO delete this constructor.  Only here for testing
  CoulombFitterParallel(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3);
  virtual ~CoulombFitterParallel();

  void SetCoulombAttributes(AnalysisType aAnalysisType);

  void PassHyperGeo1F1AndGTildeToParallelWaveFunction();
  void LoadInterpHistFile(TString aFileBaseName);  //TODO should this be a vritual function?

  td3dVec BuildPairKStar3dVecFull(TString aPairKStarNtupleBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);  //TODO fix the fPairKStar3dVecInfo and should this be a virtual function?

  td3dVec BuildPairKStar3dVecFromTxt(TString aFileName);  //TODO fix the fPairKStar3dVecInfo and should this be a virtual function?

  td1dVec BuildPairKStar4dVecFromTxt(TString aFileBaseName);  //TODO fix the fPairKStar3dVecInfo and should this be a virtual function?
  void BuildPairKStar4dVecOnFly(TString aPairKStarNtupleBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);

  static bool Sort2dPairsByRadius(const td1dVec &aVec1, const td1dVec &aVec2);
  static bool Sort2dPairsByGlobalBin(const td1dVec &aVec1, const td1dVec &aVec2);
  void FullSort2dPairVec(td2dVec &aPairs, BinInfoHyperGeo1F1 &aBinInfo);
  void FullSort4dPairVec(td4dVec &a4dPairs, BinInfoHyperGeo1F1 &aBinInfo);
  void BuildPairSample4dVec(int aNPairsPerKStarBin=16384, double aBinSize=0.01, bool aSort=true);

  bool CanInterpAllSamplePairs();
  td3dVec GetCPUSamplePairs(int aAnalysisNumber);

  void CreateScattLenSubs(double aReF0, double aImF0, double aD0);

  double GetFitCfContentParallel(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);  //TODO!!!!!

  td4dVec Get3dPairs(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber); //One 3dvec for GPU pairs and 1 3dvec for CPU pairs
  td1dVec GetEntireFitCfContent(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber);

  td1dVec GetEntireFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber);

  td1dVec GetEntireFitCfContentwStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);
  td1dVec GetEntireFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);

  td1dVec GetEntireFitCfContentComplete2(int aNSimPairsPerBin, double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par);

  //void CalculateChi2(int &npar, double &chi2, double *par);

  void CalculateFitFunction(int &npar, double &chi2, double *par);
  void CalculateFitFunctionFake(int &npar, double &chi2, double *par);

  TH1* CreateFitHistogramParallel(TString aName, int aAnalysisNumber);
  TH1* CreateFitHistogramSampleParallel(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm);
  TH1* CreateFitHistogramSampleCompleteParallel(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s,  double aReF0t, double aImF0t, double aD0t, double aNorm);

  void InitializeFitter();  //Called within DoFit
  void DoFit();
  void Finalize();  //Send things back to analyses, etc.

  //inline (i.e. simple) functions
  ParallelWaveFunction* GetParallelWaveFunctionObject();

  void SetNThreadsPerBlock(double aNThreadsPerBlock);
  void SetNBlocks(double aNBlocks);
  void SetNThreadsPerBlockAndNBlocks(double aNThreadsPerBlock, double aNBlocks);

private:
  BinInfoHyperGeo1F1 fHyperGeo1F1Info;
  BinInfoGTilde fGTildeInfo;

  td2dVec fGTildeReal;
  td2dVec fGTildeImag;

  td3dVec fHyperGeo1F1Real;
  td3dVec fHyperGeo1F1Imag;

  BinInfoSamplePairs fSamplePairsBinInfo;  //TODO naming of fSamplePairsBinInfo and fPairKStar3dVecInfo confusing
  BinInfoKStar fPairKStar3dVecInfo;
  ParallelWaveFunction* fParallelWaveFunction;

  //---------------------------

#ifdef __ROOT__
  ClassDef(CoulombFitterParallel, 1)
#endif
};


//inline stuff
inline ParallelWaveFunction* CoulombFitterParallel::GetParallelWaveFunctionObject() {return fParallelWaveFunction;}

inline void CoulombFitterParallel::SetNThreadsPerBlock(double aNThreadsPerBlock) {fParallelWaveFunction->SetNThreadsPerBlock(aNThreadsPerBlock);}
inline void CoulombFitterParallel::SetNBlocks(double aNBlocks) {fParallelWaveFunction->SetNBlocks(aNBlocks);}
inline void CoulombFitterParallel::SetNThreadsPerBlockAndNBlocks(double aNThreadsPerBlock, double aNBlocks) {fParallelWaveFunction->SetNThreadsPerBlockAndNBlocks(aNThreadsPerBlock, aNBlocks);}

#endif
