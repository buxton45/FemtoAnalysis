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
//TODO emplace_back
class ThermEvent {

public:
  ThermEvent();
  ThermEvent(TTree* aThermEventsTree, int aEntryNumber);
  ThermEvent(const ThermEvent& aEvent);  //TODO make this deep copy
  ThermEvent& operator=(ThermEvent& aEvent);  //TODO make this deep copy
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


  vector<ThermV0Particle> GetV0ParticleCollection(ParticlePDGType aPDGType);
  vector<ThermParticle> GetParticleCollection(ParticlePDGType aPDGType);

private:

  vector<ThermParticle> fAllParticlesCollection;
  vector<ThermParticle> fAllDaughtersCollection;

  vector<ThermV0Particle> fLambdaCollection;
  vector<ThermV0Particle> fAntiLambdaCollection;
  vector<ThermV0Particle> fK0ShortCollection;

  vector<ThermParticle> fKchPCollection;
  vector<ThermParticle> fKchMCollection;




#ifdef __ROOT__
  ClassDef(ThermEvent, 1)
#endif
};


//inline stuff


#endif

