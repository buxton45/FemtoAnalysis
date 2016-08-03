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
  fAllParticlesCollection(0),
  fLambdaCollection(0),
  fAntiLambdaCollection(0),
  fK0ShortCollection(0),

  fKchPCollection(0),
  fKchMCollection(0)

{

}

//________________________________________________________________________________________________________________
ThermEvent::ThermEvent(TTree* aThermEventsTree, int aEntryNumber) :
  fAllParticlesCollection(0),
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
void ThermEvent::PushBackThermParticle(ThermParticle* aParticle)
{
  fAllParticlesCollection.push_back(aParticle);
}

//________________________________________________________________________________________________________________
//TODO
void ThermEvent::PushBackThermParticleOfInterest(ThermParticle* aParticle)
{
  if(aParticle->GetPID() == kPDGLam) fLambdaCollection.push_back((ThermV0Particle*)aParticle);
  else if(aParticle->GetPID() == kPDGALam) fAntiLambdaCollection.push_back((ThermV0Particle*)aParticle);
  else if(aParticle->GetPID() == kPDGK0) fK0ShortCollection.push_back((ThermV0Particle*)aParticle);

  else if(aParticle->GetPID() == kPDGKchP) fKchPCollection.push_back(aParticle);
  else if(aParticle->GetPID() == kPDGKchM) fKchMCollection.push_back(aParticle);

  else
  {
    cout << "Particle of wrong type trying to be added collection via ThermEvent::PushBackThermParticleOfInterest" << endl;
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

//________________________________________________________________________________________________________________
void ThermEvent::AssertAllFathersFoundDaughters()
{
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) assert(fLambdaCollection[i]->BothDaughtersFound());
  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++) assert(fAntiLambdaCollection[i]->BothDaughtersFound());
  for(unsigned int i=0; i<fK0ShortCollection.size(); i++) assert(fK0ShortCollection[i]->BothDaughtersFound());

}


//________________________________________________________________________________________________________________
//TODO this and MatchDaughtersWithFathers are inefficient!
//TODO Figure out how to deep copy so this method works!
void ThermEvent::FindFatherandLoadDaughter(ThermParticle* aDaughterParticle)
{
  int tFatherPID = aDaughterParticle->GetFatherPID();
  int tFatherEID = aDaughterParticle->GetFatherEID();

  vector<ThermV0Particle*> tFatherCollection; //just a copy, when I load, I must use the actual object

  if(tFatherPID == kPDGLam) tFatherCollection = fLambdaCollection;
  else if(tFatherPID == kPDGALam) tFatherCollection = fAntiLambdaCollection;
  else if(tFatherPID == kPDGK0) tFatherCollection = fK0ShortCollection;
  else assert(0);
  //---------------------------------
  int tFatherLocation = -1;
  for(unsigned int i=0; i<tFatherCollection.size(); i++)
  {
    if(tFatherCollection[i]->GetEID() == tFatherEID)
    {
      tFatherLocation = i;
      break;
    }
  }
  assert(tFatherLocation >= 0);
  //---------------------------------
  if(tFatherPID == kPDGLam) fLambdaCollection[tFatherLocation]->LoadDaughter(aDaughterParticle);
  else if(tFatherPID == kPDGALam) fAntiLambdaCollection[tFatherLocation]->LoadDaughter(aDaughterParticle);
  else if(tFatherPID == kPDGK0) fK0ShortCollection[tFatherLocation]->LoadDaughter(aDaughterParticle);
}

//________________________________________________________________________________________________________________
bool ThermEvent::IsDaughterOfInterest(ThermParticle* aDaughterParticle)
{
  int tFatherPID = aDaughterParticle->GetFatherPID();
  int tFatherEID = aDaughterParticle->GetFatherEID();
 
  if(tFatherEID == -1) return false;  //this is a primordial particle with no father!
  else if(tFatherPID == kPDGLam) return true;
  else if(tFatherPID == kPDGALam) return true;
  else if(tFatherPID == kPDGK0) return true;
  else return false;
}

//________________________________________________________________________________________________________________
//TODO this and FindFatherAndLoadDaughter are inefficient!
void ThermEvent::MatchDaughtersWithFathers()
{
  for(unsigned int i=0; i<fAllParticlesCollection.size(); i++)
  {
    if(IsDaughterOfInterest(fAllParticlesCollection[i]))
    {
      FindFatherandLoadDaughter(fAllParticlesCollection[i]);
    }
  }

  AssertAllFathersFoundDaughters();
}


//________________________________________________________________________________________________________________
vector<ThermV0Particle*> ThermEvent::GetV0ParticleCollection(ParticlePDGType aPDGType)
{
  if(aPDGType == kPDGLam) return fLambdaCollection;
  else if(aPDGType == kPDGALam) return fAntiLambdaCollection;
  else if(aPDGType == kPDGK0) return fK0ShortCollection;

  else assert(0);
}


//________________________________________________________________________________________________________________
vector<ThermParticle*> ThermEvent::GetParticleCollection(ParticlePDGType aPDGType)
{

  if(aPDGType == kPDGKchP) return fKchPCollection;
  else if(aPDGType == kPDGKchM) return fKchMCollection;

  else assert(0);
}


