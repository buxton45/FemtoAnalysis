/* ThermPairAnalysis.h */

#ifndef THERMPAIRANALYSIS_H
#define THERMPAIRANALYSIS_H

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
#include "TFile.h"
#include "TLorentzVector.h"
#include "TVector3.h"

#include "Types.h"
#include "PIDMapping.h"

#include "ThermEvent.h"
class ThermEvent;


using namespace std;

class ThermPairAnalysis {

public:
  ThermPairAnalysis(AnalysisType aAnType);
  virtual ~ThermPairAnalysis();

  void SetPartTypes();
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


  double CalcKStar(ThermParticle &tPart1, ThermParticle &tPart2);

//TODO combine GetKStar3Vec and GetRStar3Vec
  TVector3 GetKStar3Vec(ThermParticle &tPart1, ThermParticle &tPart2);
  TVector3 GetRStar3Vec(ThermParticle &tPart1, ThermParticle &tPart2);
  double CalcRStar(ThermParticle &tPart1, ThermParticle &tPart2);
  complex<double> GetStrongOnlyWaveFunction(TVector3 &aKStar3Vec, TVector3 &aRStar3Vec);
  double GetStrongOnlyWaveFunctionSq(TVector3 aKStar3Vec, TVector3 aRStar3Vec);

  void FillCorrelationFunctionsParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, bool aMixedEvents);
  void FillCorrelationFunctionsV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection, bool aMixedEvents);

  void BuildCorrelationFunctionsParticleV0(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection);
  void BuildCorrelationFunctionsV0V0(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection);

  TH1* BuildFinalCf(TH1* aNum, TH1* aDen, TString aName);
  void SaveAllCorrelationFunctions(TFile *aFile);

  void ProcessEvent(ThermEvent &aEvent, vector<ThermEvent> &aMixingEventsCollection, double aMaxPrimaryDecayLength=-1.);

  //-- inline
  void SetUseMixedEvents(bool aUse);
  void SetBuildUniqueParents(bool aBuild);
  TH2D* GetTransformMatrix(int aIndex);
private:
  AnalysisType fAnalysisType;
  ParticlePDGType fPartType1, fPartType2;

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

  TH1* fPairSourceFull;
  TH1* fNumFull;
  TH1* fDenFull;
  TH1* fCfFull;

  TH1* fPairSourcePrimaryOnly;
  TH1* fNumPrimaryOnly;
  TH1* fDenPrimaryOnly;
  TH1* fCfPrimaryOnly;

  TH1* fPairSourcePrimaryAndShortDecays;
  TH1* fNumPrimaryAndShortDecays;
  TH1* fDenPrimaryAndShortDecays;
  TH1* fCfPrimaryAndShortDecays;

  TH1* fPairSourceWithoutSigmaSt;
  TH1* fNumWithoutSigmaSt;
  TH1* fDenWithoutSigmaSt;
  TH1* fCfWithoutSigmaSt;

  TH1* fPairSourceSigmaStOnly;
  TH1* fNumSigmaStOnly;
  TH1* fDenSigmaStOnly;
  TH1* fCfSigmaStOnly;

#ifdef __ROOT__
  ClassDef(ThermPairAnalysis, 1)
#endif
};

inline void ThermPairAnalysis::SetUseMixedEvents(bool aUse) {fMixEvents = aUse;}
inline void ThermPairAnalysis::SetBuildUniqueParents(bool aBuild) {fBuildUniqueParents = aBuild;}

inline TH2D* ThermPairAnalysis::GetTransformMatrix(int aIndex) {return ((TH2D*)fTransformMatrices->At(aIndex));}

#endif















