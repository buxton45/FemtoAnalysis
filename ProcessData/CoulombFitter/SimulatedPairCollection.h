// file SimulatedPairCollection.h


#ifndef SIMULATEDPAIRCOLLECTION_H
#define SIMULATEDPAIRCOLLECTION_H

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
#include <cassert>

#include "TVector3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TFile.h"
#include "TList.h"

#include "ChronoTimer.h"

#include <omp.h>

using namespace std;

#include "Types.h"

#include "Interpolator.h"
class Interpolator;

class SimulatedPairCollection {

public:
  //Constructor, destructor, copy constructor, assignment operator
  SimulatedPairCollection();
  virtual ~SimulatedPairCollection();

  void ExtractDataPairKStar3dVecFromSingleRootFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill);
  td3dVec BuildDataPairKStar3dVecFromAllRootFiles(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  void BuildPairKStar4dVecOnFly(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);

  static void WriteRow(ostream &aOutput, vector<double> &aRow);
  void WriteDataPairKStar3dVecTxtFile(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  void WriteAllDataPairKStar3dVecTxtFiles(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax);

  td3dVec BuildDataPairKStar3dVecFromTxt(TString aFileName);
  void BuildDataPairKStar4dVecFromTxt(TString aFileBaseName);

  void SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax);
  void BuildPair4dVec(int aNPairsPerKStarBin=16384, double aBinSize=0.01);
  void UpdatePairRadiusParameter(double aNewRadius, int aAnalysisNumber);

  int GetAnalysisNumber(AnalysisType aAnalysisType, CentralityType aCentralityType);


  //---- INLINE FUNCTIONS -------------------------------
  unsigned int GetNumberOfPairsInBin(int aAnalysisNumber, int aBin);
  double GetPairKStarMag(int aAnalysisNumber, int aBin, int aPairNum);
  double GetPairRStarMag(int aAnalysisNumber, int aBin, int aPairNum);
  double GetPairTheta(int aAnalysisNumber, int aBin, int aPairNum);

protected:
  bool fUseStaticPairs;
  bool fUseRandomKStarVectors; //if false, use vectors from data
  bool bShareSingleSampleAmongstAll;  //if true, fNAnalyses = 1

  int fNPairsPerKStarBin;
  double fKStarBinSize;
  double fMaxBuildKStar;

  int fNAnalyses;
  vector<tmpAnalysisInfo> fAnalysesInfo;  //should have size = fNAnalyses
  td1dVec fCurrentRadii; //should have size = fNAnalyses
  td4dVec fDataPairKStar4dVec;  // This is comprised of k* vectors from the data, and will be empty if fUseRandomKStarVectors = true
                                // 1 3dVec for each of fNAnalyses.
                                // 1 2dVec for each k* bin
                                // 1 1dVec for each particle = [KStarMag, KStarOut, KStarSide, KStarLong]
  td4dVec fPair4dVec;  // 1 3dVec for each of fNAnalyses
                       // 1 2dVec for each k* bin
                       // 1 1dVec for each particle = [KStarMag, RStarMag, Theta], where Thera is angle between k* and r*
                       //  Will be initialized by sampling RStar vectors from Gaussian distributions with mu=0 and sigma=1
                       //  When R parameter is updated, I simply scale all RStar magnitudes

#ifdef __ROOT__
  ClassDef(SimulatedPairCollection, 1)
#endif
};

inline unsigned int SimulatedPairCollection::GetNumberOfPairsInBin(int aAnalysisNumber, int aBin) {return fPair4dVec[aAnalysisNumber][aBin].size();}
inline double SimulatedPairCollection::GetPairKStarMag(int aAnalysisNumber, int aBin, int aPairNum) {return fPair4dVec[aAnalysisNumber][aBin][aPairNum][0];}
inline double SimulatedPairCollection::GetPairRStarMag(int aAnalysisNumber, int aBin, int aPairNum) {return fPair4dVec[aAnalysisNumber][aBin][aPairNum][1];}
inline double SimulatedPairCollection::GetPairTheta(int aAnalysisNumber, int aBin, int aPairNum) {return fPair4dVec[aAnalysisNumber][aBin][aPairNum][2];}

#endif
