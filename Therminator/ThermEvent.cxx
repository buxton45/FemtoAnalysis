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

  fPiPCollection(0),
  fPiMCollection(0),
  fKchPCollection(0),
  fKchMCollection(0),
  fProtCollection(0),
  fAProtCollection(0)

{

}

//________________________________________________________________________________________________________________
ThermEvent::ThermEvent(TTree* aThermEventsTree, int aEntryNumber) :
  fAllParticlesCollection(0),
  fLambdaCollection(0),
  fAntiLambdaCollection(0),
  fK0ShortCollection(0),

  fPiPCollection(0),
  fPiMCollection(0),
  fKchPCollection(0),
  fKchMCollection(0),
  fProtCollection(0),
  fAProtCollection(0)

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
  if(aParticle->GetParticlePDGType() == kPDGLam) fLambdaCollection.push_back((ThermV0Particle*)aParticle);
  else if(aParticle->GetParticlePDGType() == kPDGALam) fAntiLambdaCollection.push_back((ThermV0Particle*)aParticle);
  else if(aParticle->GetParticlePDGType() == kPDGK0) fK0ShortCollection.push_back((ThermV0Particle*)aParticle);

  else if(aParticle->GetParticlePDGType() == kPDGPiP) fPiPCollection.push_back(aParticle);
  else if(aParticle->GetParticlePDGType() == kPDGPiM) fPiMCollection.push_back(aParticle);
  else if(aParticle->GetParticlePDGType() == kPDGKchP) fKchPCollection.push_back(aParticle);
  else if(aParticle->GetParticlePDGType() == kPDGKchM) fKchMCollection.push_back(aParticle);
  else if(aParticle->GetParticlePDGType() == kPDGProt) fProtCollection.push_back(aParticle);
  else if(aParticle->GetParticlePDGType() == kPDGAntiProt) fAProtCollection.push_back(aParticle);

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
  fPiPCollection.clear();
  fPiMCollection.clear();
}

//________________________________________________________________________________________________________________
//TODO this and MatchDaughtersWithFathers are inefficient!
void ThermEvent::FindFatherandLoadDaughter(ThermParticle* aDaughterParticle)
{
  ParticlePDGType tFatherType = aDaughterParticle->GetFatherPDGType();
  int tFatherEID = aDaughterParticle->GetFatherEID();

  vector<ThermV0Particle*> tFatherCollection; //just a copy, when I load, I must use the actual object

  if(tFatherType == kPDGLam) tFatherCollection = fLambdaCollection;
  else if(tFatherType == kPDGALam) tFatherCollection = fAntiLambdaCollection;
  else if(tFatherType == kPDGK0) tFatherCollection = fK0ShortCollection;
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
  if(tFatherType == kPDGLam) fLambdaCollection[tFatherLocation]->LoadDaughter(aDaughterParticle);
  else if(tFatherType == kPDGALam) fAntiLambdaCollection[tFatherLocation]->LoadDaughter(aDaughterParticle);
  else if(tFatherType == kPDGK0) fK0ShortCollection[tFatherLocation]->LoadDaughter(aDaughterParticle);


}

//________________________________________________________________________________________________________________
bool ThermEvent::IsDaughterOfInterest(ParticlePDGType aFatherOfDaughterType)
{
  if(aFatherOfDaughterType == kPDGLam) return true;
  else if(aFatherOfDaughterType == kPDGALam) return true;
  else if(aFatherOfDaughterType == kPDGK0) return true;
  else return false;
}

//________________________________________________________________________________________________________________
//TODO this and FindFatherAndLoadDaughter are inefficient!
void ThermEvent::MatchDaughtersWithFathers(ParticlePDGType aDaughterType)
{
  vector<ThermParticle*> tDaughterCollection;

  if(aDaughterType == kPDGPiP) tDaughterCollection = fPiPCollection;
  else if(aDaughterType == kPDGPiM) tDaughterCollection = fPiMCollection;

  else if(aDaughterType == kPDGKchP) tDaughterCollection = fKchPCollection;
  else if(aDaughterType == kPDGKchM) tDaughterCollection = fKchMCollection;

  else if(aDaughterType == kPDGProt) tDaughterCollection = fProtCollection;
  else if(aDaughterType == kPDGAntiProt) tDaughterCollection = fAProtCollection;

  else assert(0);

  //-------------------------------------------

  for(unsigned int i=0; i<tDaughterCollection.size(); i++)
  {
    if(IsDaughterOfInterest(tDaughterCollection[i]->GetFatherPDGType()))
    {
      FindFatherandLoadDaughter(tDaughterCollection[i]);
    }
  }

}

//________________________________________________________________________________________________________________
//TODO this and FindFatherAndLoadDaughter are inefficient!
void ThermEvent::MatchAllDaughtersWithFathers()
{
  MatchDaughtersWithFathers(kPDGPiP);
  MatchDaughtersWithFathers(kPDGPiM);

  MatchDaughtersWithFathers(kPDGKchP);
  MatchDaughtersWithFathers(kPDGKchM);

  MatchDaughtersWithFathers(kPDGProt);
  MatchDaughtersWithFathers(kPDGAntiProt);
}


//________________________________________________________________________________________________________________
void ThermEvent::AssertAllFathersFoundDaughters()
{
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) assert(fLambdaCollection[i]->BothDaughtersFound());
  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++) assert(fAntiLambdaCollection[i]->BothDaughtersFound());
  for(unsigned int i=0; i<fK0ShortCollection.size(); i++) assert(fK0ShortCollection[i]->BothDaughtersFound());

}


//________________________________________________________________________________________________________________
vector<ThermV0Particle*> ThermEvent::GetLambdaParticleCollection(ParticlePDGType aPDGType)
{
  if(aPDGType == kPDGLam) return fLambdaCollection;
  else if(aPDGType == kPDGALam) return fAntiLambdaCollection;
  else if(aPDGType == kPDGK0) return fK0ShortCollection;

  else assert(0);
}


//________________________________________________________________________________________________________________
vector<ThermParticle*> ThermEvent::GetParticleCollection(ParticlePDGType aPDGType)
{
  if(aPDGType == kPDGPiP) return fPiPCollection;
  else if(aPDGType == kPDGPiM) return fPiMCollection;

  else if(aPDGType == kPDGKchP) return fKchPCollection;
  else if(aPDGType == kPDGKchM) return fKchMCollection;

  else if(aPDGType == kPDGProt) return fProtCollection;
  else if(aPDGType == kPDGAntiProt) return fAProtCollection;

  else assert(0);
}


