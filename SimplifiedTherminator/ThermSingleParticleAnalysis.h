/* ThermSingleParticleAnalysis.h */

#ifndef THERMSINGLEPARTICLEANALYSIS_H
#define THERMSINGLEPARTICLEANALYSIS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cassert>
#include <random>

#include "TObjArray.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TFile.h"

#include "Types.h"
#include "PIDMapping.h"

#include "ThermEvent.h"
class ThermEvent;


using namespace std;

class ThermSingleParticleAnalysis {

public:
  ThermSingleParticleAnalysis(ParticlePDGType aParticlePDGType);
  virtual ~ThermSingleParticleAnalysis();

  double GetSampledCTau(double aMeanCTau);
  double GetLabDecayLength(double aMeanCTau, double aMass, double aE, double aMagP);

  void MapAndFillParents(ThermParticle &aParticle);
  void MapAndFillRadiiHistograms(ThermParticle &aParticle);
  void FillUniqueParticleParents(ThermParticle &aParticle);
  void PrintUniqueParents();


  void ProcessEventForV0(ThermEvent &aEvent);
  void ProcessEventForParticle(ThermEvent &aEvent);
  void ProcessEvent(ThermEvent &aEvent);

  void SaveAll(TFile *aFile);

  //--inline
  void SetBuildUniqueParents(bool aBuild);

private:
  ParticlePDGType fParticlePDGType;

  bool fBuildUniqueParents;
  vector<int> fUniqueParents;
  vector<int> fAllFathers;

  TH1* fParents;
  TH1* fRadii;
  TH2* f2dRadiiVsPid;
  TH2* f2dRadiiVsBeta;
  TH3* f3dRadii;

  TH3* fTransverseEmission;
  TH3* fTransverseEmissionPrimaryOnly;

  TH3* fTransverseEmissionVsTau;
  TH3* fTransverseEmissionVsTauPrimaryOnly;

#ifdef __ROOT__
  ClassDef(ThermSingleParticleAnalysis, 1)
#endif
};

inline void ThermSingleParticleAnalysis::SetBuildUniqueParents(bool aBuild) {fBuildUniqueParents = aBuild;}

#endif















