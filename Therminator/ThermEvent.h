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

#include "ThermLambdaParticle.h"
class ThermLambdaParticle;

#include "ParticleCoor.h"

class ThermEvent {

public:
  ThermEvent();
  ThermEvent(TTree* aThermEventsTree, int aEntryNumber);
  virtual ~ThermEvent();

  void PushBackThermParticle(ThermParticle* aParticle);
  void ClearThermEvent();

private:


  vector<ThermLambdaParticle*> fLambdaCollection;
  vector<ThermLambdaParticle*> fAntiLambdaCollection;
  vector<ThermLambdaParticle*> fK0ShortCollection;
  vector<ThermParticle*> fKchPCollection;
  vector<ThermParticle*> fKchMCollection;




#ifdef __ROOT__
  ClassDef(ThermEvent, 1)
#endif
};


//inline stuff


#endif

