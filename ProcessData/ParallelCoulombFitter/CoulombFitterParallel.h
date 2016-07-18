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
  CoulombFitterParallel(); //TODO delete this constructor.  Only here for testing
  CoulombFitterParallel(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3, bool aCreateInterpVectors=true, bool aUseScattLenHists=false);
  virtual ~CoulombFitterParallel();

  void LoadInterpHistFile(TString aFileBaseName);  //TODO should this be a vritual function?

  td3dVec BuildPairKStar3dVec(TString aPairKStarNtupleLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, int aNbinsKStar, double aKStarMin, double aKStarMax);  //TODO fix the fPairKStar3dVecInfo and should this be a virtual function?

  td3dVec BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, vector<int> &aNFilesPerSubDir, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);  //TODO fix the fPairKStar3dVecInfo and should this be a virtual function?
  td3dVec BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);  //TODO fix the fPairKStar3dVecInfo and should this be a virtual function?

  td3dVec BuildPairKStar3dVecFromTxt(TString aFileName);  //TODO fix the fPairKStar3dVecInfo and should this be a virtual function?
  void BuildPairKStar4dVecFromTxt(TString aFileBaseName);  //TODO fix the fPairKStar3dVecInfo and should this be a virtual function?


  void CreateScattLenSubs(double aReF0, double aImF0, double aD0);
  vector<double> InterpolateWfSquaredParallel(vector<vector<double> > &aPairs, double aReF0, double aImF0, double aD0);

  td4dVec Get3dPairs(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber); //One 3dvec for GPU pairs and 1 3dvec for CPU pairs
  td1dVec GetEntireFitCfContent(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber);
  td1dVec GetEntireFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par, int aAnalysisNumber);
  td1dVec GetEntireFitCfContentComplete2(int aNSimPairsPerBin, double aKStarMagMin, double aKStarMagMax, int aNbinsK, double *par);

  double GetFitCfContentParallel(double aKStarMagMin, double aKStarMagMax, double *par);  //TODO!!!!!

  //void CalculateChi2(int &npar, double &chi2, double *par);

  void CalculateChi2PMLParallel(int &npar, double &chi2, double *par);
  void CalculateChi2Parallel(int &npar, double &chi2, double *par);
  void CalculateFakeChi2Parallel(int &npar, double &chi2, double *par);

  TH1* CreateFitHistogramParallel(TString aName, int aAnalysisNumber);
  TH1* CreateFitHistogramSampleParallel(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm);
  TH1* CreateFitHistogramSampleCompleteParallel(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s,  double aReF0t, double aImF0t, double aD0t, double aNorm);

  //inline (i.e. simple) functions
  ParallelWaveFunction* GetParallelWaveFunctionObject();

private:
  BinInfoKStar fPairKStar3dVecInfo;
  ParallelWaveFunction* fParallelWaveFunction;

  td3dVec fPairKStar3dVec;  //TODO delete this
  //---------------------------

#ifdef __ROOT__
  ClassDef(CoulombFitterParallel, 1)
#endif
};


//inline stuff
inline ParallelWaveFunction* CoulombFitterParallel::GetParallelWaveFunctionObject() {return fParallelWaveFunction;}

#endif
