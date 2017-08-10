///////////////////////////////////////////////////////////////////////////
// ThermEventsCollection:                                                //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMEVENTSCOLLECTION
#define THERMEVENTSCOLLECTION

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

#include "TSystemDirectory.h"
#include "TSystemFile.h"
#include "TH2.h"
#include "TH2D.h"
#include "TFile.h"
#include "TCanvas.h"

#include "ThermEvent.h"
class ThermEvent;

using std::string;
using std::stringstream;
using std::istringstream;

class ThermEventsCollection {

public:
  ThermEventsCollection();
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

  bool DoubleCheckLamAttributes(ThermV0Particle &aV0);
  bool DoubleCheckALamAttributes(ThermV0Particle &aV0);
  bool DoubleCheckK0Attributes(ThermV0Particle &aV0);
  bool DoubleCheckV0Attributes(ThermV0Particle &aV0);

  double GetKStar(ThermParticle &aParticle, ThermV0Particle &aV0);
  double GetKStar(ThermV0Particle &aV01, ThermV0Particle &aV02);
  double GetFatherKStar(ThermParticle &aParticle, ThermV0Particle &aV0, bool aUseParticleFather=false, bool aUseV0Father=true);
  double GetFatherKStar(ThermV0Particle &aV01, ThermV0Particle &aV02, bool aUseV01Father=true, bool aUseV02Father=false);

  void FillTransformMatrixParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix);
  void FillTransformMatrixV0V0(vector<ThermV0Particle> &aV01Collection, vector<ThermV0Particle> &aV02Collection, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix);

  void BuildTransformMatrixParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, ParticlePDGType aParticleFatherType, ParticlePDGType aV0FatherType, TH2* aMatrix);
  void BuildTransformMatrixV0V0(ParticlePDGType aV01Type, ParticlePDGType aV02Type, ParticlePDGType aV01FatherType, ParticlePDGType aV02FatherType, TH2* aMatrix);

  void BuildAllTransformMatrices();  //TODO
  void SaveAllTransformMatrices(TString aSaveFileLocation);

  void MapAndFillParentsMatrixParticleV0(TH2* aMatrix, int aV0FatherType, int aTrackFatherType);
  void MapAndFillParentsMatrixV0V0(TH2* aMatrix, int aV01FatherType, int aV02Type);
  void MapAndFillProtonParents(TH1* aHist, int aFatherType);

  void MapAndFillPairFractionHistogramParticleV0(TH1* aHistogram, int aV0FatherType, int aTrackFatherType);
  void MapAndFillPairFractionHistogramV0V0(TH1* aHistogram, int aV01FatherType, int aV02FatherType);

  void BuildPairFractionHistogramsParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, TH1* aHistogram, TH2* aMatrix);
  void BuildPairFractionHistogramsV0V0(ParticlePDGType aV01Type, ParticlePDGType aV02Type, TH1* aHistogram, TH2* aMatrix);
  void BuildProtonParents();

  void BuildAllPairFractionHistograms();
  void BuildUniqueParents(int aParticleType, int aFatherType);
  vector<int> UniqueCombineVectors(vector<int> &aVec1, vector<int> &aVec2);
  void PrintUniqueParents();

  void SaveAllPairFractionHistograms(TString aSaveFileLocation);
  TCanvas* DrawAllPairFractionHistograms();


  //inline
  void SetUseMixedEvents(bool aMixEvents);
  void SetNEventsToMix(int aNEventsToMix);

private:
  int fNFiles;
  int fNEvents;
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

  //ALamK0s
  TH2* fSigToALamK0Transform;
  TH2* fXiCToALamK0Transform;
  TH2* fXi0ToALamK0Transform;
  TH2* fOmegaToALamK0Transform;
  //--
  TH2* fSigStPToALamK0Transform;
  TH2* fSigStMToALamK0Transform;
  TH2* fSigSt0ToALamK0Transform;
  TH2* fALamKSt0ToALamK0Transform;
  TH2* fSigKSt0ToALamK0Transform;
  TH2* fXiCKSt0ToALamK0Transform;
  TH2* fXi0KSt0ToALamK0Transform;

  TH1* fPairFractionsALamK0;
  TH2* fParentsMatrixALamK0;

  //LamLam to check with Jai
  TH2* fSigToLamLamTransform;

  //Protons
  TH1* fProtonParents;
  TH1* fAProtonParents;



#ifdef __ROOT__
  ClassDef(ThermEventsCollection, 1)
#endif
};


//inline stuff
inline void ThermEventsCollection::SetUseMixedEvents(bool aMixEvents) {fMixEvents = aMixEvents;}
inline void ThermEventsCollection::SetNEventsToMix(int aNEventsToMix) {fNEventsToMix = aNEventsToMix; fMixEvents=true;}

#endif




