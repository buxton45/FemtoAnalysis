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
  double GetFatherKStar(ThermParticle &aParticle, ThermV0Particle &aV0);
  double GetFatherKStar(ThermV0Particle &aV02, ThermV0Particle &aV0);

  void FillTransformMatrixParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aFatherType, TH2* aMatrix);
  void FillTransformMatrixV0V0(vector<ThermV0Particle> &aV0wFatherCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aFatherType, TH2* aMatrix);

  void BuildTransformMatrixParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, ParticlePDGType aFatherType, TH2* aMatrix);
  void BuildTransformMatrixV0V0(ParticlePDGType aV0wFatherType, ParticlePDGType aV0Type, ParticlePDGType aFatherType, TH2* aMatrix);

  void BuildAllTransformMatrices();  //TODO
  void SaveAllTransformMatrices(TString aSaveFileLocation);

  void MapAndFillParentsMatrix(TH2* aMatrix, int aV0FatherType, int aTrackFatherType);
  void BuildPairFractionHistogramsParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, TH1* aHistogram, TH2* aMatrix);
  void BuildAllPairFractionHistograms();
  void BuildUniqueParents(int aV0FatherType, int aTrackFatherType);
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
  vector<int> fUniqueV0Parents;
  vector<int> fUniqueTrackParents;

  //LamKchP
  TH2* fSigToLamKchPTransform;
  TH2* fXiCToLamKchPTransform;
  TH2* fXi0ToLamKchPTransform;
  TH2* fOmegaToLamKchPTransform;

  TH1* fPairFractionsLamKchP;
  TH2* fParentsMatrixLamKchP;

  //ALamKchP
  TH2* fASigToALamKchPTransform;
  TH2* fAXiCToALamKchPTransform;
  TH2* fAXi0ToALamKchPTransform;
  TH2* fAOmegaToALamKchPTransform;

  TH1* fPairFractionsALamKchP;
  TH2* fParentsMatrixALamKchP;

  //LamKchM
  TH2* fSigToLamKchMTransform;
  TH2* fXiCToLamKchMTransform;
  TH2* fXi0ToLamKchMTransform;
  TH2* fOmegaToLamKchMTransform;

  TH1* fPairFractionsLamKchM;
  TH2* fParentsMatrixLamKchM;

  //ALamKchM
  TH2* fASigToALamKchMTransform;
  TH2* fAXiCToALamKchMTransform;
  TH2* fAXi0ToALamKchMTransform;
  TH2* fAOmegaToALamKchMTransform;

  TH1* fPairFractionsALamKchM;
  TH2* fParentsMatrixALamKchM;

  //LamLam to check with Jai
  TH2* fSigToLamLamTransform;

  //TODO K0 analyses will invole feed-down from both Lam and K0

#ifdef __ROOT__
  ClassDef(ThermEventsCollection, 1)
#endif
};


//inline stuff
inline void ThermEventsCollection::SetUseMixedEvents(bool aMixEvents) {fMixEvents = aMixEvents;}
inline void ThermEventsCollection::SetNEventsToMix(int aNEventsToMix) {fNEventsToMix = aNEventsToMix; fMixEvents=true;}

#endif




