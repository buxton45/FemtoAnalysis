// file SimulatedLednickyCf.h
//TODO Better to use NumIntLednickyCf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ifndef SIMULATEDLEDNICKYCF_H
#define SIMULATEDLEDNICKYCF_H

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

class SimulatedLednickyCf {

public:
  //Constructor, destructor, copy constructor, assignment operator
//  SimulatedLednickyCf();  //used for stand-alone running for writing text files 
  SimulatedLednickyCf(double aKStarBinSize=0.01, double aMaxBuildKStar=1.0, int aNPairsPerKStarBin=16384);
  virtual ~SimulatedLednickyCf();

  void SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax);
  void BuildPair3dVec(int aNPairsPerKStarBin=16384, double aBinSize=0.01);
  void UpdatePairRadiusParameter(double aNewRadius, double aMuOut);

  bool AreParamsSameExcludingLambda(double *aCurrent, double *aNew, int aNEntries);
  void AdjustLambdaParam(td1dVec &aCoulombCf, double aOldLambda, double aNewLambda);

  complex<double> GetStrongOnlyWaveFunction(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec, const double aRef0, const double aImf0, const double ad0);
  double GetStrongOnlyWaveFunctionSq(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec, const double aRef0, const double aImf0, const double ad0); 

  double GetFitCfContent(double aKStarMagMin,/* double aKStarMagMax,*/ double *par);

  //---- INLINE FUNCTIONS -------------------------------
  unsigned int GetNumberOfPairsInBin(int aBin);

protected:
  std::default_random_engine fGenerator;

  int fNPairsPerKStarBin;
  double fKStarBinSize;
  double fMaxBuildKStar;

  double fCurrentRadius;

  td3dVec fPair3dVec;  // 1 2dVec for each k* bin
                       // 1 1dVec for each particle = [KStarOut, KStarSide, KStarLong, RStarOut, RStarSide, RStarLong]
                       //  Will be initialized by sampling RStar vectors from Gaussian distributions with mu=0 and sigma=sqrt(2)*1
                       //  When R parameter is updated, I simply scale all RSide and RLong
                       //  ROut is a little different, since now I allow a shift, muOut
                       //  For that reason, I keep RStarOutOG in the vector as well
                       //  In future, if I want to include muSide and muLong, it may be easiest to keep two td3dVec objects
                       //    One with original distribution (with mu=0 sigma=1), and one with current pair with any mu and sigma

#ifdef __ROOT__
  ClassDef(SimulatedLednickyCf, 1)
#endif
};

inline unsigned int SimulatedLednickyCf::GetNumberOfPairsInBin(int aBin) {return fPair3dVec[aBin].size();}


#endif
