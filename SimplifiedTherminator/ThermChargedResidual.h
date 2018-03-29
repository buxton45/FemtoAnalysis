/* ThermChargedResidual.h */

#ifndef THERMCHARGEDRESIDUAL_H
#define THERMCHARGEDRESIDUAL_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cassert>
#include <complex>

#include "TObjArray.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TVector3.h"

#include "Types.h"
#include "PIDMapping.h"

using namespace std;

class ThermChargedResidual {

public:
  ThermChargedResidual(AnalysisType aResType);
  ThermChargedResidual(const ThermChargedResidual& aRes);
  ThermChargedResidual& operator=(const ThermChargedResidual& aRes);
  virtual ~ThermChargedResidual();

  void SetPartTypes();

  void LoadCoulombOnlyInterpWfs(TString aFileDirectory="/home/jbuxton/SimplifiedTherminator/CoulombOnlyInterpWfs/");
  bool CanInterp(double aKStarMag, double aRStarMag, double aTheta);
  bool CanInterp(TVector3* aKStar3Vec, TVector3* aRStar3Vec);

  double InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta);
  double InterpolateWfSquared(TVector3* aKStar3Vec, TVector3* aRStar3Vec);

  //inline
  AnalysisType GetResidualType();
  vector<ParticlePDGType> GetPartTypes();

private:
  AnalysisType fResidualType;
  ParticlePDGType fPartType1, fPartType2;

  TH3D* f3dCoulombOnlyInterpWfs;
  double fInterpKStarMagMin, fInterpKStarMagMax;
  double fInterpRStarMagMin, fInterpRStarMagMax;
  double fInterpThetaMin, fInterpThetaMax;

#ifdef __ROOT__
  ClassDef(ThermChargedResidual, 1)
#endif
};

inline AnalysisType ThermChargedResidual::GetResidualType() {return fResidualType;}
inline vector<ParticlePDGType> ThermChargedResidual::GetPartTypes() {return vector<ParticlePDGType>{fPartType1, fPartType2};}


#endif







