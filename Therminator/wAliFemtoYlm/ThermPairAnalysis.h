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
#include "TH3.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TVector3.h"

#include "Types.h"
#include "PIDMapping.h"

#include "ThermEvent.h"
class ThermEvent;

#include "ThermChargedResidual.h"
class ThermChargedResidual;

#include "CorrFctnDirectYlm.h"
#include "AliFemtoCorrFctnDirectYlm.h"

using namespace std;

class ThermPairAnalysis {

public:
  ThermPairAnalysis(AnalysisType aAnType);
  virtual ~ThermPairAnalysis();

  static vector<ParticlePDGType> GetPartTypes(AnalysisType aAnType);
  void SetPartTypes();

  void InitiateTransformMatrices();
  void SetBuildTransformMatrices(bool aBuild);

  void InitiateCorrelations();
  void SetBuildCorrelationFunctions(bool aBuild, bool aBuild3dHists=false, bool aBuildPairSourcewmTInfo=false);
  void SetBuildCfYlm(bool aSet);
  void SetBuildAliFemtoCfYlm(bool aSet);

  void InitiatePairFractionsAndParentsMatrix();
  void SetBuildPairFractions(bool aBuild);

  void LoadChargedResiduals();
  void SetWeightCfsWithParentInteraction(bool aSet);
  void SetOnlyWeightLongDecayParents(bool aSet);
  bool IsChargedResidual(ParticlePDGType aType1, ParticlePDGType aType2);
  AnalysisType GetChargedResidualType(ParticlePDGType aType1, ParticlePDGType aType2);
  int GetChargedResidualIndex(ParticlePDGType aType1, ParticlePDGType aType2);

  double GetFatherKStar(const ThermParticle &aParticle1, const ThermParticle &aParticle2, bool aUseParticleFather1, bool aUseParticleFather2);
  double GetKStar(const ThermParticle &aParticle1, const ThermParticle &aParticle2);

  void FillTransformMatrixParticleV0(const vector<ThermParticle> &aParticleCollection, const vector<ThermV0Particle> &aV0Collection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix);
  void FillTransformMatrixV0V0(const vector<ThermV0Particle> &aV01Collection, const vector<ThermV0Particle> &aV02Collection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix);

  void BuildTransformMatrixParticleV0(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix);
  void BuildTransformMatrixV0V0(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix);
  void BuildAllTransformMatrices(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection);
  void SaveAllTransformMatrices(TFile *aFile);


  void MapAndFillParentsMatrixParticleV0(TH2* aMatrix, int aV0FatherType, int aTrackFatherType);
  void MapAndFillParentsMatrixV0V0(TH2* aMatrix, int aV01FatherType, int aV02Type);

  void FillUniqueParents(vector<int> &aUniqueParents, int aFatherType);
  static vector<int> UniqueCombineVectors(const vector<int> &aVec1, const vector<int> &aVec2);
  void PrintUniqueParents();

  void FillPrimaryAndOtherPairInfo(int aParentType1, int aParentType2);
  void PrintPrimaryAndOtherPairInfo();

  static void MapAndFillPairFractionHistogramParticleV0(TH1* aHistogram, int aV0FatherType, int aTrackFatherType, double aMaxPrimaryDecayLength=-1., double tWeight=1.);
  static void MapAndFillPairFractionHistogramV0V0(TH1* aHistogram, int aV01FatherType, int aV02FatherType, double aMaxPrimaryDecayLength=-1., double tWeight=1.);


  void BuildPairFractionHistogramsParticleV0(const ThermEvent &aEvent);
  void BuildPairFractionHistogramsV0V0(const ThermEvent &aEvent);

  void SavePairFractionsAndParentsMatrix(TFile *aFile);


  double CalcKStar(const TLorentzVector &p1, const TLorentzVector &p2);
  double CalcKStar(const ThermParticle &tPart1, const ThermParticle &tPart2);

  double CalcKStar_RotatePar2(const TLorentzVector &p1, const TLorentzVector &p2);           //Rotate the second particle in the pair by 180 degrees about the z-axis
  double CalcKStar_RotatePar2(const ThermParticle &tPart1, const ThermParticle &tPart2);     //Rotate the second particle in the pair by 180 degrees about the z-axis

  TLorentzVector Boost4VecToOSLinLCMS(const TLorentzVector &p1, const TLorentzVector &p2, const TLorentzVector &aVecToBoost);
  TLorentzVector Boost4VecToOSLinPRF(const TLorentzVector &p1, const TLorentzVector &p2, const TLorentzVector &aVecToBoost);

  TVector3 GetKStar3Vec(const TLorentzVector &p1, const TLorentzVector &p2);
  TVector3 GetKStar3Vec(const ThermParticle &tPart1, const ThermParticle &tPart2);

  TVector3 GetKStar3Vec_RotatePar2(const TLorentzVector &p1, const TLorentzVector &p2);
  TVector3 GetKStar3Vec_RotatePar2(const ThermParticle &tPart1, const ThermParticle &tPart2);

  TVector3 DrawRStar3VecFromGaussian(double tROut, double tMuOut, double tRSide, double tMuSide, double tRLong, double tMuLong);
  TVector3 DrawRStar3VecFromGaussian();

  TLorentzVector GetRStar4Vec(const TLorentzVector &p1, const TLorentzVector &x1, const TLorentzVector &p2, const TLorentzVector &x2);
  TLorentzVector GetRStar4Vec(const ThermParticle &tPart1, const ThermParticle &tPart2);

  TVector3 GetRStar3Vec(const TLorentzVector &p1, const TLorentzVector &x1, const TLorentzVector &p2, const TLorentzVector &x2);
  TVector3 GetRStar3Vec(const ThermParticle &tPart1, const ThermParticle &tPart2);

  double CalcRStar(const TLorentzVector &p1, const TLorentzVector &x1, const TLorentzVector &p2, const TLorentzVector &x2);
  double CalcRStar(const ThermParticle &tPart1, const ThermParticle &tPart2);
  vector<double> CalcR1R2inPRF(const TLorentzVector &p1, const TLorentzVector &x1, const TLorentzVector &p2, const TLorentzVector &x2);
  vector<double> CalcR1R2inPRF(const ThermParticle &tPart1, const ThermParticle &tPart2);

  double CalcmT(const TLorentzVector &p1, const TLorentzVector &p2);
  double CalcmT(const ThermParticle &tPart1, const ThermParticle &tPart2);
  void FillParentmT3d(TH3* aPairmT3d, const ThermParticle &tPart1, const ThermParticle &tPart2);

  complex<double> GetStrongOnlyWaveFunction(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec);  //TODO decide what to do about resonances! (i.e. R=0 pairs)
  double GetStrongOnlyWaveFunctionSq(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec); 
  double GetParentPairWaveFunctionSq(const ThermParticle &tPart1, const ThermParticle &tPart2);
  double GetParentPairWaveFunctionSq_RotatePar2(const ThermParticle &tPart1, const ThermParticle &tPart2);

  void FillCorrelations(const ThermParticle &aParticle1, const ThermParticle &aParticle2, bool aFillNumerator);  //For ParticleV0, aParticle1=V0 and aParticle2=Particle

  void FillCorrelationFunctionsNumOrDenParticleParticle(const vector<ThermParticle> &aParticleCollection1, const vector<ThermParticle> &aParticleCollection2, bool aFillNumerator);
  void FillCorrelationFunctionsNumOrDenParticleV0(const vector<ThermParticle> &aParticleCollection, const vector<ThermV0Particle> &aV0Collection, bool aFillNumerator);
  void FillCorrelationFunctionsNumOrDenV0V0(const vector<ThermV0Particle> &aV01Collection, const vector<ThermV0Particle> &aV02Collection, bool aFillNumerator);

  void FillCorrelationFunctionsNumAndDenParticleParticle(const vector<ThermParticle> &aParticleCollection1, const vector<ThermParticle> &aParticleCollection2);
  void FillCorrelationFunctionsNumAndDenParticleV0(const vector<ThermParticle> &aParticleCollection, const vector<ThermV0Particle> &aV0Collection);
  void FillCorrelationFunctionsNumAndDenV0V0(const vector<ThermV0Particle> &aV01Collection, const vector<ThermV0Particle> &aV02Collection);

  td2dVec GetTrueRoslContributions(const ThermParticle &aParticle1, const ThermParticle &aParticle2);
  void AddRoslContributionToMasterVector(td2dVec &aTrueRoslMaster, int &aNPairs, const ThermParticle &aParticle1, const ThermParticle &aParticle2);
  td2dVec FinalizeTrueRoslMaster(td2dVec &aTrueRoslMaster, int aNPairs);

  void BuildTrueRoslParticleParticle(const vector<ThermParticle> &aParticleCollection1, const vector<ThermParticle> &aParticleCollection2);
  void BuildTrueRoslParticleV0(const vector<ThermParticle> &aParticleCollection, const vector<ThermV0Particle> &aV0Collection);
  void BuildTrueRoslV0V0(const vector<ThermV0Particle> &aV01Collection, const vector<ThermV0Particle> &aV02Collection);

  void BuildCorrelationFunctionsParticleParticle(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection);
  void BuildCorrelationFunctionsParticleV0(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection);
  void BuildCorrelationFunctionsV0V0(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection);

  TH1* BuildFinalCf(TH1* aNum, TH1* aDen, TString aName);
  void SaveAllCorrelationFunctions(TFile *aFile);

  void ProcessEvent(const ThermEvent &aEvent, const vector<ThermEvent> &aMixingEventsCollection);

  //-- inline
  void SetUseMixedEventsForTransforms(bool aUse);
  void SetBuildUniqueParents(bool aBuild);
  TH2D* GetTransformMatrix(int aIndex) const;

  void SetBuildSingleParticleAnalyses(bool aBuild);

  void SetBuildMixedEventNumerators(bool aBuild);
  void SetUnitWeightCfNums(bool aUse);

  void SetDrawRStarFromGaussian(bool aSet);

  void SetMaxPrimaryDecayLength(double aMax);

private:
  std::default_random_engine fGenerator;

  AnalysisType fAnalysisType;
  ParticlePDGType fPartType1, fPartType2;

  double fKStarMin, fKStarMax;
  int fNBinsKStar;

  double fMaxPrimaryDecayLength;
  
  bool fMixEventsForTransforms;
  bool fBuildUniqueParents;
  vector<int> fUniqueParents1;
  vector<int> fUniqueParents2;

  bool fBuildPairFractions;
  bool fBuildTransformMatrices;
  bool fBuildCorrelationFunctions;
  bool fBuild3dHists;                //Advantage of 3dHists is ability to tweak primary definition after processing
                                     //Disadvantage is they consume a huge amount of memory
  bool fBuildPairSourcewmTInfo;
  bool fBuildMixedEventNumerators;

  vector<AnalysisType> fTransformStorageMapping;
  vector<TransformInfo> fTransformInfo;
  TObjArray* fTransformMatrices;

  TH1* fPairFractions;
  TH2* fParentsMatrix;

  vector<vector<PidInfo> > fPrimaryPairInfo;  //each vector<PidInfo> has 2 elements for each particle in pair
  vector<vector<PidInfo> > fOtherPairInfo;

  vector<AnalysisType> fChargedResidualsTypeMap;
  vector<ThermChargedResidual> fChargedResiduals;

  bool fUnitWeightCfNums;
  bool fWeightCfsWithParentInteraction;
  bool fOnlyWeightLongDecayParents;

  bool fDrawRStarFromGaussian;

  TH3* fPairSource3d;
  TH3* fNum3d;
  TH3* fDen3d;

  TH1* fPairSourceFull;
  TH1* fNumFull;
  TH1* fDenFull;
  TH1* fCfFull;
  TH1* fNumFull_RotatePar2;

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

  TH1* fPairSourceSecondaryOnly;
  TH1* fNumSecondaryOnly;
  TH1* fDenSecondaryOnly;
  TH1* fCfSecondaryOnly;

  TH2* fPairKStarVsmT;
  TH3* fPairmT3d;

  TH3* fPairSource3d_OSLinPRF;
  TH3* fPairSource3d_OSLinPRFPrimaryOnly;

  TH1* fPairDeltaT_inPRF;
  TH1* fPairDeltaT_inPRFPrimaryOnly;

  TH3* fTrueRosl;
  TH3* fSimpleRosl;

  TH3* fPairSource3d_mT1vmT2vRinv;
  TH2* fPairSource2d_PairmTvRinv;
  TH2* fPairSource2d_mT1vRinv;
  TH2* fPairSource2d_mT2vRinv;
  TH2* fPairSource2d_mT1vR1PRF;
  TH2* fPairSource2d_mT2vR2PRF;

  bool fBuildCfYlm;
  CorrFctnDirectYlm* fCfYlm;

  bool fBuildAliFemtoCfYlm;
  AliFemtoCorrFctnDirectYlm* fAliFemtoCfYlm;

#ifdef __ROOT__
  ClassDef(ThermPairAnalysis, 1)
#endif
};

inline void ThermPairAnalysis::SetUseMixedEventsForTransforms(bool aUse) {fMixEventsForTransforms = aUse;}
inline void ThermPairAnalysis::SetBuildUniqueParents(bool aBuild) {fBuildUniqueParents = aBuild;}

inline TH2D* ThermPairAnalysis::GetTransformMatrix(int aIndex) const {return ((TH2D*)fTransformMatrices->At(aIndex));}

inline void ThermPairAnalysis::SetBuildMixedEventNumerators(bool aBuild) {fBuildMixedEventNumerators = aBuild;}
inline void ThermPairAnalysis::SetUnitWeightCfNums(bool aUse) {fUnitWeightCfNums = aUse;}

inline void ThermPairAnalysis::SetDrawRStarFromGaussian(bool aSet) {fDrawRStarFromGaussian = aSet;}

inline void ThermPairAnalysis::SetMaxPrimaryDecayLength(double aMax) {fMaxPrimaryDecayLength = aMax;}

#endif














