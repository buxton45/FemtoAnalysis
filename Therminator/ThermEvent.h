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

  void FindV0Father(ThermV0Particle &aV0Particle);
  void FindAllV0sFathers();


  vector<ThermV0Particle> GetV0ParticleCollection(ParticlePDGType aPDGType);
  vector<ThermParticle> GetParticleCollection(ParticlePDGType aPDGType);

  void SetV0ParticleCollection(int aEventID, ParticlePDGType aPDGType, vector<ThermV0Particle> &aCollection);
  void SetParticleCollection(int aEventID, ParticlePDGType aPDGType, vector<ThermParticle> &aCollection);

  bool DoubleCheckLamAttributes(ThermV0Particle &aV0);
  bool DoubleCheckALamAttributes(ThermV0Particle &aV0);
  bool DoubleCheckK0Attributes(ThermV0Particle &aV0);
  bool DoubleCheckV0Attributes(ThermV0Particle &aV0);

  double GetKStar(ThermParticle &aParticle, ThermV0Particle &aV0);
  double GetKStar(ThermV0Particle &aV01, ThermV0Particle &aV02);
  double GetFatherKStar(ThermParticle &aParticle, ThermV0Particle &aV0);
  double GetFatherKStar(ThermV0Particle &aV02, ThermV0Particle &aV0);
  void FillTransformMatrixParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, ParticlePDGType aFatherType, TH2* aMatrix);
  void FillTransformMatrixV0V0(ParticlePDGType aV0wFatherType, ParticlePDGType aV0Type, ParticlePDGType aFatherType, TH2* aMatrix);

  //inline stuff
  void SetEventID(int aEventID);
  int GetEventID();

private:
  int fEventID;

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
inline void ThermEvent::SetEventID(int aEventID) {fEventID = aEventID;}
inline int ThermEvent::GetEventID() {return fEventID;}

#endif

