///////////////////////////////////////////////////////////////////////////
// ThermEvent:                                                           //
///////////////////////////////////////////////////////////////////////////


#include "ThermEvent.h"

#ifdef __ROOT__
ClassImp(ThermEvent)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

//________________________________________________________________________________________________________________
ThermEvent::ThermEvent() :
  fLambdaCollection(0),
  fAntiLambdaCollection(0),
  fK0ShortCollection(0),
  fKchPCollection(0),
  fKchMCollection(0)

{

}

//________________________________________________________________________________________________________________
ThermEvent::ThermEvent(TTree* aThermEventsTree, int aEntryNumber) :
  fLambdaCollection(0),
  fAntiLambdaCollection(0),
  fK0ShortCollection(0),
  fKchPCollection(0),
  fKchMCollection(0)

{

}


//________________________________________________________________________________________________________________
ThermEvent::~ThermEvent()
{
  cout << "ThermEvent object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
//TODO
void ThermEvent::PushBackThermParticle(ThermParticle* aParticle)
{
  if(aParticle->GetParticleType() == kLam) fLambdaCollection.push_back((ThermLambdaParticle*)aParticle);
  else if(aParticle->GetParticleType() == kALam) fAntiLambdaCollection.push_back((ThermLambdaParticle*)aParticle);
  else if(aParticle->GetParticleType() == kK0) fK0ShortCollection.push_back((ThermLambdaParticle*)aParticle);
  else if(aParticle->GetParticleType() == kKchP) fKchPCollection.push_back(aParticle);
  else if(aParticle->GetParticleType() == kKchM) fKchMCollection.push_back(aParticle);
  else
  {
    cout << "Particle of wrong type trying to be added collection via ThermEvent::PushBackThermParticle" << endl;
    cout << "PREPARE FOR CRASH" << endl;
    assert(0);
  }
}

//________________________________________________________________________________________________________________
void ThermEvent::ClearThermEvent()
{
  fLambdaCollection.clear();
  fAntiLambdaCollection.clear();
  fK0ShortCollection.clear();
  fKchPCollection.clear();
  fKchMCollection.clear();
}






