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

#include "ThermParticle.h"
class ThermParticle;

#include "ThermV0Particle.h"
class ThermV0Particle;

#include "ParticleCoor.h"

class ThermEvent {

public:
  ThermEvent();
  ThermEvent(TTree* aThermEventsTree, int aEntryNumber);
  virtual ~ThermEvent();

  void PushBackThermParticle(ThermParticle* aParticle);
  void PushBackThermParticleOfInterest(ThermParticle* aParticle);
  void ClearThermEvent();

  void FindFatherandLoadDaughter(ThermParticle* aDaughterParticle);
  bool IsDaughterOfInterest(ParticlePDGType aFatherOfDaughterType);
  void MatchDaughtersWithFathers(ParticlePDGType aDaughterType);
  void MatchAllDaughtersWithFathers();
  void AssertAllFathersFoundDaughters();

  vector<ThermV0Particle*> GetLambdaParticleCollection(ParticlePDGType aPDGType);
  vector<ThermParticle*> GetParticleCollection(ParticlePDGType aPDGType);

private:

  vector<ThermParticle*> fAllParticlesCollection;

  vector<ThermV0Particle*> fLambdaCollection;
  vector<ThermV0Particle*> fAntiLambdaCollection;
  vector<ThermV0Particle*> fK0ShortCollection;

  vector<ThermParticle*> fPiPCollection;
  vector<ThermParticle*> fPiMCollection;
  vector<ThermParticle*> fKchPCollection;
  vector<ThermParticle*> fKchMCollection;
  vector<ThermParticle*> fProtCollection;
  vector<ThermParticle*> fAProtCollection;




#ifdef __ROOT__
  ClassDef(ThermEvent, 1)
#endif
};


//inline stuff


#endif

