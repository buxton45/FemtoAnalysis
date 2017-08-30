///////////////////////////////////////////////////////////////////////////
// ThermEventsCollection:                                                //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMEVENTSCOLLECTION_H
#define THERMEVENTSCOLLECTION_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <random>

#include "TSystemDirectory.h"
#include "TSystemFile.h"
#include "TH2.h"
#include "TH2D.h"
#include "TFile.h"
#include "TCanvas.h"

#include "ThermEvent.h"
class ThermEvent;

#include "PIDMapping.h"

using std::string;
using std::stringstream;
using std::istringstream;

class ThermEventsCollection {

public:
  ThermEventsCollection(TString aEventsDirectory = "/home/jesse/Analysis/Therminator2/events/lhyqid3v_LHCPbPb_2760_b2/");
  virtual ~ThermEventsCollection();

  int ReturnEventIndex(unsigned int aEventID);

  void WriteRow(ostream &aOutput, vector<double> &aRow);
  vector<double> PackageV0ParticleForWriting(ThermV0Particle &aV0);
  vector<double> PackageParticleForWriting(ThermParticle &aParticle);
  void WriteThermEventV0s(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent aThermEvent);
  void WriteThermEventParticles(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent aThermEvent);
  void WriteAllEventsParticlesOfType(TString aOutputName, ParticlePDGType aParticleType);
  void WriteAllEvents(TString aOutputNameBase);

  void ExtractV0ParticleCollectionsFromTxtFile(TString aFileName, ParticlePDGType aPDGType);
  void ExtractParticleCollectionsFromTxtFile(TString aFileName, ParticlePDGType aPDGType);
  void ExtractEventsFromAllTxtFiles(TString aFileLocationBase);

  void ExtractEventsFromRootFile(TString aFileLocation);
  void ExtractFromAllRootFiles(const char *aDirName, bool bBuildUniqueParents=false);

  //---------------------------------------------------

  double GetFatherKStar(ThermParticle &aParticle1, ThermParticle &aParticle2, bool aUseParticleFather1, bool aUseParticleFather2);
  double GetKStar(ThermParticle &aParticle1, ThermParticle &aParticle2);

  void FillTransformMatrixParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix);
  void FillTransformMatrixV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix);

  void BuildTransformMatrixParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix);
  void BuildTransformMatrixV0V0(ParticlePDGType aV01Type, ParticlePDGType aV02Type, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix);

  void BuildAllTransformMatrices();  //TODO
  void SaveAllTransformMatrices(TString aSaveFileLocation);

  void MapAndFillParentsMatrixParticleV0(TH2* aMatrix, int aV0FatherType, int aTrackFatherType);
  void MapAndFillParentsMatrixV0V0(TH2* aMatrix, int aV01FatherType, int aV02Type);
  void MapAndFillProtonParents(TH1* aHist, int aFatherType);
  void MapAndFillProtonRadii(TH2* a2dHist, ThermParticle &aParticle);
  void MapAndFillLambdaRadii(TH2* a2dHist, ThermV0Particle &aParticle);

  void FillPrimaryAndOtherPairInfo(int aType1, int aType2, int aParentType1, int aParentType2, double aMaxPrimaryDecayLength=-1.);
  void PrintPrimaryAndOtherPairInfo(int aType1, int aType2);
  void PrintAllPrimaryAndOtherPairInfo();

  static void MapAndFillPairFractionHistogramParticleV0(TH1* aHistogram, int aV0FatherType, int aTrackFatherType, double aMaxPrimaryDecayLength=-1., double tWeight=1.);
  static void MapAndFillPairFractionHistogramV0V0(TH1* aHistogram, int aV01FatherType, int aV02FatherType, double aMaxPrimaryDecayLength=-1., double tWeight=1.);

  void BuildPairFractionHistogramsParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, TH1* aHistogram, TH2* aMatrix, double aMaxPrimaryDecayLength=-1.);
  void BuildPairFractionHistogramsV0V0(ParticlePDGType aV01Type, ParticlePDGType aV02Type, TH1* aHistogram, TH2* aMatrix, double aMaxPrimaryDecayLength=-1.);
  void BuildProtonParents();
  void BuildLambdaParents();

  void BuildAllPairFractionHistograms(double aMaxPrimaryDecayLength=-1.);
  void BuildUniqueParents(int aParticleType, int aFatherType);
  vector<int> UniqueCombineVectors(vector<int> &aVec1, vector<int> &aVec2);
  void PrintUniqueParents();

  void SaveAllPairFractionHistograms(TString aSaveFileLocation);
  TCanvas* DrawAllPairFractionHistograms();

  double GetProperDecayLength(double aMeanDecayLength);
  double GetLabDecayLength(double aMeanDecayLength, double aMass, double aE);

  //inline
  void SetUseMixedEvents(bool aMixEvents);
  void SetNEventsToMix(int aNEventsToMix);

  vector<ThermEvent> GetEventsCollection();
  vector<ThermEvent> GetMixingEventsCollection();

private:
  int fNFiles;
  int fNEvents;
  TString fEventsDirectory;
  vector<TString> fFileNameCollection;
  vector<ThermEvent> fEventsCollection;

  bool fMixEvents;
  unsigned int fNEventsToMix;
  vector<ThermEvent> fMixingEventsCollection;

  double fKStarMin, fKStarMax;
  int fNBinsKStar;

  bool fBuildUniqueParents;

  vector<int> fUniqueLamParents;
  vector<int> fUniqueALamParents;
  vector<int> fUniquecLamParents; //cLam = Lam and ALam

  vector<int> fUniqueK0Parents;

  vector<int> fUniqueKchPParents;
  vector<int> fUniqueKchMParents;
  vector<int> fUniquecKchParents; //cKch = KchP and KchM

  vector<int> fUniqueProtParents;
  vector<int> fUniqueAProtParents;
  vector<int> fUniquecProtParents;

  //LamKchP
  TH2* fSigToLamKchPTransform;
  TH2* fXiCToLamKchPTransform;
  TH2* fXi0ToLamKchPTransform;
  TH2* fOmegaToLamKchPTransform;
  //--
  TH2* fSigStPToLamKchPTransform;
  TH2* fSigStMToLamKchPTransform;
  TH2* fSigSt0ToLamKchPTransform;
  TH2* fLamKSt0ToLamKchPTransform;
  TH2* fSigKSt0ToLamKchPTransform;
  TH2* fXiCKSt0ToLamKchPTransform;
  TH2* fXi0KSt0ToLamKchPTransform;

  TH1* fPairFractionsLamKchP;
  TH2* fParentsMatrixLamKchP;

  vector<vector<PidInfo> > fPrimaryPairInfoLamKchP;
  vector<vector<PidInfo> > fOtherPairInfoLamKchP;

  //ALamKchP
  TH2* fASigToALamKchPTransform;
  TH2* fAXiCToALamKchPTransform;
  TH2* fAXi0ToALamKchPTransform;
  TH2* fAOmegaToALamKchPTransform;
  //--
  TH2* fASigStMToALamKchPTransform;
  TH2* fASigStPToALamKchPTransform;
  TH2* fASigSt0ToALamKchPTransform;
  TH2* fALamKSt0ToALamKchPTransform;
  TH2* fASigKSt0ToALamKchPTransform;
  TH2* fAXiCKSt0ToALamKchPTransform;
  TH2* fAXi0KSt0ToALamKchPTransform;

  TH1* fPairFractionsALamKchP;
  TH2* fParentsMatrixALamKchP;

  vector<vector<PidInfo> > fPrimaryPairInfoALamKchP;
  vector<vector<PidInfo> > fOtherPairInfoALamKchP;

  //LamKchM
  TH2* fSigToLamKchMTransform;
  TH2* fXiCToLamKchMTransform;
  TH2* fXi0ToLamKchMTransform;
  TH2* fOmegaToLamKchMTransform;
  //--
  TH2* fSigStPToLamKchMTransform;
  TH2* fSigStMToLamKchMTransform;
  TH2* fSigSt0ToLamKchMTransform;
  TH2* fLamAKSt0ToLamKchMTransform;
  TH2* fSigAKSt0ToLamKchMTransform;
  TH2* fXiCAKSt0ToLamKchMTransform;
  TH2* fXi0AKSt0ToLamKchMTransform;

  TH1* fPairFractionsLamKchM;
  TH2* fParentsMatrixLamKchM;

  vector<vector<PidInfo> > fPrimaryPairInfoLamKchM;
  vector<vector<PidInfo> > fOtherPairInfoLamKchM;

  //ALamKchM
  TH2* fASigToALamKchMTransform;
  TH2* fAXiCToALamKchMTransform;
  TH2* fAXi0ToALamKchMTransform;
  TH2* fAOmegaToALamKchMTransform;
  //--
  TH2* fASigStMToALamKchMTransform;
  TH2* fASigStPToALamKchMTransform;
  TH2* fASigSt0ToALamKchMTransform;
  TH2* fALamAKSt0ToALamKchMTransform;
  TH2* fASigAKSt0ToALamKchMTransform;
  TH2* fAXiCAKSt0ToALamKchMTransform;
  TH2* fAXi0AKSt0ToALamKchMTransform;

  TH1* fPairFractionsALamKchM;
  TH2* fParentsMatrixALamKchM;

  vector<vector<PidInfo> > fPrimaryPairInfoALamKchM;
  vector<vector<PidInfo> > fOtherPairInfoALamKchM;

  //LamK0s
  TH2* fSigToLamK0Transform;
  TH2* fXiCToLamK0Transform;
  TH2* fXi0ToLamK0Transform;
  TH2* fOmegaToLamK0Transform;
  //--
  TH2* fSigStPToLamK0Transform;
  TH2* fSigStMToLamK0Transform;
  TH2* fSigSt0ToLamK0Transform;
  TH2* fLamKSt0ToLamK0Transform;
  TH2* fSigKSt0ToLamK0Transform;
  TH2* fXiCKSt0ToLamK0Transform;
  TH2* fXi0KSt0ToLamK0Transform;

  TH1* fPairFractionsLamK0;
  TH2* fParentsMatrixLamK0;

  vector<vector<PidInfo> > fPrimaryPairInfoLamK0;
  vector<vector<PidInfo> > fOtherPairInfoLamK0;

  //ALamK0s
  TH2* fASigToALamK0Transform;
  TH2* fAXiCToALamK0Transform;
  TH2* fAXi0ToALamK0Transform;
  TH2* fAOmegaToALamK0Transform;
  //--
  TH2* fASigStMToALamK0Transform;
  TH2* fASigStPToALamK0Transform;
  TH2* fASigSt0ToALamK0Transform;
  TH2* fALamKSt0ToALamK0Transform;
  TH2* fASigKSt0ToALamK0Transform;
  TH2* fAXiCKSt0ToALamK0Transform;
  TH2* fAXi0KSt0ToALamK0Transform;

  TH1* fPairFractionsALamK0;
  TH2* fParentsMatrixALamK0;

  vector<vector<PidInfo> > fPrimaryPairInfoALamK0;
  vector<vector<PidInfo> > fOtherPairInfoALamK0;

  //LamLam to check with Jai
  TH2* fSigToLamLamTransform;

  //Protons
  TH1* fProtonParents;
  TH1* fAProtonParents;

  TH1* fProtonRadii;
  TH1* fAProtonRadii;

  TH2* f2dProtonRadii;
  TH2* f2dAProtonRadii;

  TH1* fLamRadii;
  TH1* fALamRadii;

  TH2* f2dLamRadii;
  TH2* f2dALamRadii;

#ifdef __ROOT__
  ClassDef(ThermEventsCollection, 1)
#endif
};


//inline stuff
inline void ThermEventsCollection::SetUseMixedEvents(bool aMixEvents) {fMixEvents = aMixEvents;}
inline void ThermEventsCollection::SetNEventsToMix(int aNEventsToMix) {fNEventsToMix = aNEventsToMix; fMixEvents=true;}


inline vector<ThermEvent> ThermEventsCollection::GetEventsCollection() {return fEventsCollection;}
inline vector<ThermEvent> ThermEventsCollection::GetMixingEventsCollection() {return fMixingEventsCollection;}

#endif




