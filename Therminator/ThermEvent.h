///////////////////////////////////////////////////////////////////////////
// ThermEvent:                                                           //
//             Basically takes an event root file from the Therminator2  //
// output and creates an object that's easier for me to work with        //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMEVENT
#define THERMEVENT

#include "assert.h"

#include "TTree.h"
#include "TFile.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"

#include "ThermParticle.h"
class ThermParticle;

#include "ThermV0Particle.h"
class ThermV0Particle;

#include "ParticleCoor.h"
class ParticleCoor;
//TODO emplace_back
class ThermEvent {

public:
  ThermEvent();
  ThermEvent(TTree* aThermEventsTree, int aEntryNumber);
  ThermEvent(const ThermEvent& aEvent);  //TODO make this deep copy
  ThermEvent& operator=(const ThermEvent& aEvent);  //TODO make this deep copy
  virtual ThermEvent* clone();
  virtual ~ThermEvent();

  bool IsParticleOfInterest(ParticleCoor* aParticle);
  bool IsDaughterOfInterest(ThermParticle& aDaughterParticle);
  bool IsDaughterOfInterest(ParticleCoor* aDaughterParticle);

  void PushBackThermParticle(ParticleCoor *aParticle);
  void PushBackThermDaughterOfInterest(ParticleCoor *aParticle);
  void PushBackThermParticleOfInterest(ParticleCoor *aParticle);

  void ClearCollection(vector<ThermParticle> &aCollection);
  void ClearCollection(vector<ThermV0Particle> &aCollection);
  void ClearThermEvent();

  bool CheckCollectionForRedundancies();

  void AssertAllLambdaFathersFoundDaughters();  //Not all K0 have daughters (those without likely K0L, those with likely K0s
  void AssertAllK0FathersFound0or2Daughters();
  void FindFatherandLoadDaughter(ThermParticle &aDaughterParticle);
  void MatchDaughtersWithFathers();

  void FindFather(ThermParticle &aParticle);
  void FindAllFathers();
  void EnforceKinematicCuts();

  vector<ThermV0Particle> GetV0ParticleCollection(ParticlePDGType aPDGType) const;
  vector<ThermParticle> GetParticleCollection(ParticlePDGType aPDGType) const;

  vector<ThermParticle> GetGoodParticleCollectionCastAsThermParticle(ParticlePDGType aPDGType) const;
  vector<ThermParticle> GetGoodParticleCollectionCastAsThermParticle(int aPID) const;
  vector<ThermParticle> GetGoodParticleCollectionwConjCastAsThermParticle(int aPID) const;

  void SetV0ParticleCollection(unsigned int aEventID, ParticlePDGType aPDGType, vector<ThermV0Particle> &aCollection);
  void SetParticleCollection(unsigned int aEventID, ParticlePDGType aPDGType, vector<ThermParticle> &aCollection);

  void CheckCoECoM();

  double CalculateEventPlane(vector<ThermParticle> &aCollection);
  double CalculateEventPlane(vector<ThermV0Particle> &aCollection);

  void RotateParticlesByRandomAzimuthalAngle(double aPhi, vector<ThermParticle> &aCollection, bool aOutputEP=false);
  void RotateParticlesByRandomAzimuthalAngle(double aPhi, vector<ThermV0Particle> &aCollection, bool aOutputEP=false);
  void RotateAllParticlesByRandomAzimuthalAngle(bool aOutputEP=false);

  bool IncludeInV3(int aV3InclusionProb1, ThermParticle& aParticle);
  void BuildArtificialV3SignalInCollection(int aV3InclusionProb1, double aPsi3, TF1* aDist, vector<ThermParticle> &aCollection);  //NOTE: This kills v2 signal and builds v3 signal
  void BuildArtificialV3SignalInCollection(int aV3InclusionProb1, double aPsi3, TF1* aDist, vector<ThermV0Particle> &aCollection);  //NOTE: This kills v2 signal and builds v3 signal
  void BuildArtificialV3Signal(int aV3InclusionProb1=25, bool aRotateEventsByRandAzAngles=false);  //NOTE: This kills v2 signal and builds v3 signal

  void BuildArtificialV2Signal(int aV2InclusionProb1=-1, bool aRotateEventsByRandAzAngles=false);

  //inline stuff
  void SetEventID(unsigned int aEventID);
  unsigned int GetEventID() const;
  vector<ThermParticle> GetAllParticlesCollection() const;

private:
  std::default_random_engine fGenerator;

  unsigned int fEventID;

  vector<ThermParticle> fAllParticlesCollection;
  vector<ThermParticle> fAllDaughtersCollection;

  vector<ThermV0Particle> fLambdaCollection;
  vector<ThermV0Particle> fAntiLambdaCollection;
  vector<ThermV0Particle> fK0ShortCollection;

  vector<ThermParticle> fKchPCollection;
  vector<ThermParticle> fKchMCollection;

  vector<ThermParticle> fProtCollection;
  vector<ThermParticle> fAProtCollection;

#ifdef __ROOT__
  ClassDef(ThermEvent, 1)
#endif
};


//inline stuff
inline void ThermEvent::SetEventID(unsigned int aEventID) {fEventID = aEventID;}
inline unsigned int ThermEvent::GetEventID() const {return fEventID;}
inline vector<ThermParticle> ThermEvent::GetAllParticlesCollection() const {return fAllParticlesCollection;}

#endif

