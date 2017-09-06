/* ThermPairAnalysis.h */

#ifndef THERMPAIRANALYSIS_H
#define THERMPAIRANALYSIS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cassert>

#include "TObjArray.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"

#include "Types.h"
#include "PIDMapping.h"

#include "ThermEvent.h"
class ThermEvent;


using namespace std;

class ThermPairAnalysis {

public:
  ThermPairAnalysis(AnalysisType aAnType);
  virtual ~ThermPairAnalysis();

  void InitiateTransformMatrices();

  double GetFatherKStar(ThermParticle &aParticle1, ThermParticle &aParticle2, bool aUseParticleFather1, bool aUseParticleFather2);
  double GetKStar(ThermParticle &aParticle1, ThermParticle &aParticle2);

  void FillTransformMatrixParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix);
  void FillTransformMatrixV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix);

  void BuildTransformMatrixParticleV0(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix);
  void BuildTransformMatrixV0V0(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix);
  void BuildAllTransformMatrices(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection);
  void SaveAllTransformMatrices(TFile *aFile);


  void MapAndFillParentsMatrixParticleV0(TH2* aMatrix, int aV0FatherType, int aTrackFatherType);
  void MapAndFillParentsMatrixV0V0(TH2* aMatrix, int aV01FatherType, int aV02Type);

  void FillUniqueParents(vector<int> &aUniqueParents, int aFatherType);
  static vector<int> UniqueCombineVectors(vector<int> &aVec1, vector<int> &aVec2);
  void PrintUniqueParents();

  void FillPrimaryAndOtherPairInfo(int aParentType1, int aParentType2, double aMaxPrimaryDecayLength=-1.);
  void PrintPrimaryAndOtherPairInfo();

  static void MapAndFillPairFractionHistogramParticleV0(TH1* aHistogram, int aV0FatherType, int aTrackFatherType, double aMaxPrimaryDecayLength=-1., double tWeight=1.);
  static void MapAndFillPairFractionHistogramV0V0(TH1* aHistogram, int aV01FatherType, int aV02FatherType, double aMaxPrimaryDecayLength=-1., double tWeight=1.);


  void BuildPairFractionHistogramsParticleV0(ThermEvent &aEvent, double aMaxPrimaryDecayLength=-1.);
  void BuildPairFractionHistogramsV0V0(ThermEvent &aEvent, double aMaxPrimaryDecayLength=-1.);

  void SavePairFractionsAndParentsMatrix(TFile *aFile);

  //-- inline
  void SetUseMixedEvents(bool aUse);
  void SetBuildUniqueParents(bool aBuild);
  TH2D* GetTransformMatrix(int aIndex);
private:
  AnalysisType fAnalysisType;

  double fKStarMin, fKStarMax;
  int fNBinsKStar;
  
  bool fMixEvents;
  bool fBuildUniqueParents;
  vector<int> fUniqueParents1;
  vector<int> fUniqueParents2;

  vector<AnalysisType> fTransformStorageMapping;
  vector<TransformInfo> fTransformInfo;
  TObjArray* fTransformMatrices;

  TH1* fPairFractions;
  TH2* fParentsMatrix;

  vector<vector<PidInfo> > fPrimaryPairInfo;  //each vector<PidInfo> has 2 elements for each particle in pair
  vector<vector<PidInfo> > fOtherPairInfo;

#ifdef __ROOT__
  ClassDef(ThermPairAnalysis, 1)
#endif
};

inline void ThermPairAnalysis::SetUseMixedEvents(bool aUse) {fMixEvents = aUse;}
inline void ThermPairAnalysis::SetBuildUniqueParents(bool aBuild) {fBuildUniqueParents = aBuild;}

inline TH2D* ThermPairAnalysis::GetTransformMatrix(int aIndex) {return ((TH2D*)fTransformMatrices->At(aIndex));}

#endif















