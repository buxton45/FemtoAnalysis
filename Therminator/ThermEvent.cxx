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
  fEventID(0),
  fAllParticlesCollection(0),
  fAllDaughtersCollection(0),

  fLambdaCollection(0),
  fAntiLambdaCollection(0),
  fK0ShortCollection(0),

  fKchPCollection(0),
  fKchMCollection(0)

{

}

//________________________________________________________________________________________________________________
ThermEvent::ThermEvent(TTree* aThermEventsTree, int aEntryNumber) :
  fEventID(0),
  fAllParticlesCollection(0),
  fAllDaughtersCollection(0),

  fLambdaCollection(0),
  fAntiLambdaCollection(0),
  fK0ShortCollection(0),

  fKchPCollection(0),
  fKchMCollection(0)

{

}


//________________________________________________________________________________________________________________
ThermEvent::ThermEvent(const ThermEvent& aEvent) :
  fEventID(aEvent.fEventID),
  fAllParticlesCollection(aEvent.fAllParticlesCollection),
  fAllDaughtersCollection(aEvent.fAllDaughtersCollection),
  fLambdaCollection(aEvent.fLambdaCollection),
  fAntiLambdaCollection(aEvent.fAntiLambdaCollection),
  fK0ShortCollection(aEvent.fK0ShortCollection),
  fKchPCollection(aEvent.fKchPCollection),
  fKchMCollection(aEvent.fKchMCollection)
{

}

//________________________________________________________________________________________________________________
ThermEvent& ThermEvent::operator=(ThermEvent& aEvent)
{
  if(this == &aEvent) return *this;

  fEventID = aEvent.fEventID;

  fAllParticlesCollection = aEvent.fAllParticlesCollection;
  fAllDaughtersCollection = aEvent.fAllDaughtersCollection;
  fLambdaCollection = aEvent.fLambdaCollection;
  fAntiLambdaCollection = aEvent.fAntiLambdaCollection;
  fK0ShortCollection = aEvent.fK0ShortCollection;
  fKchPCollection = aEvent.fKchPCollection;
  fKchMCollection = aEvent.fKchMCollection;

  return *this;
}


//________________________________________________________________________________________________________________
ThermEvent* ThermEvent::clone()
{
  return(new ThermEvent(*this));
}


//________________________________________________________________________________________________________________
ThermEvent::~ThermEvent()
{
  /* no-op */
}


//________________________________________________________________________________________________________________
bool ThermEvent::IsParticleOfInterest(ParticleCoor* aParticle)
{
  int tPID = aParticle->pid;

  if(tPID == 3122) return true;        //Lam
  else if(tPID == -3122) return true;  //ALam

  else if(tPID == 311) return true;    //K0 (K0s = 310, but apparently there are no K0s)

  else if(tPID == 321) return true;    //KchP
  else if(tPID == -321) return true;   //KchM 

  else return false;
}

//________________________________________________________________________________________________________________
bool ThermEvent::IsDaughterOfInterest(ThermParticle &aDaughterParticle)
{
  int tFatherPID = aDaughterParticle.GetFatherPID();
 
  if(aDaughterParticle.IsPrimordial()) return false;  //this is a primordial particle with no father!
  else if(tFatherPID == kPDGLam) return true;
  else if(tFatherPID == kPDGALam) return true;
  else if(tFatherPID == kPDGK0) return true;
  else return false;
}

//________________________________________________________________________________________________________________
bool ThermEvent::IsDaughterOfInterest(ParticleCoor* aDaughterParticle)
{
  int tFatherPID = aDaughterParticle->fatherpid;
  int tFatherEID = aDaughterParticle->fathereid;

  if(tFatherEID == -1) return false;  //this is a primordial particle with no father!
  else if(tFatherPID == kPDGLam) return true;
  else if(tFatherPID == kPDGALam) return true;
  else if(tFatherPID == kPDGK0) return true;
  else return false;
}

//TODO PushBackThermParticle and PushBackThermParticleOfInterest should be able to be combined by including
//  a bool = aIsOfInterest into the arugments 
//________________________________________________________________________________________________________________
void ThermEvent::PushBackThermParticle(ParticleCoor *aParticle)
{
  fAllParticlesCollection.emplace_back(aParticle);
}


//________________________________________________________________________________________________________________
void ThermEvent::PushBackThermDaughterOfInterest(ParticleCoor *aParticle)
{
  fAllDaughtersCollection.emplace_back(aParticle);
}


//________________________________________________________________________________________________________________
//TODO
void ThermEvent::PushBackThermParticleOfInterest(ParticleCoor* aParticle)
{
  if(aParticle->pid == kPDGLam) fLambdaCollection.emplace_back(aParticle);
  else if(aParticle->pid == kPDGALam) fAntiLambdaCollection.emplace_back(aParticle);
  else if(aParticle->pid == kPDGK0) fK0ShortCollection.emplace_back(aParticle);

  else if(aParticle->pid == kPDGKchP) fKchPCollection.emplace_back(aParticle);
  else if(aParticle->pid == kPDGKchM) fKchMCollection.emplace_back(aParticle);

  else
  {
    cout << "Particle of wrong type trying to be added collection via ThermEvent::PushBackThermParticleOfInterest" << endl;
    cout << "PREPARE FOR CRASH" << endl;
    assert(0);
  }

}


//________________________________________________________________________________________________________________
void ThermEvent::ClearCollection(vector<ThermParticle> &aCollection)
{
  aCollection.clear();
  aCollection.shrink_to_fit();
}

//________________________________________________________________________________________________________________
void ThermEvent::ClearCollection(vector<ThermV0Particle> &aCollection)
{
  aCollection.clear();
  aCollection.shrink_to_fit();
}

//________________________________________________________________________________________________________________
void ThermEvent::ClearThermEvent()
{
  ClearCollection(fAllParticlesCollection);
  ClearCollection(fAllDaughtersCollection);

  ClearCollection(fLambdaCollection);
  ClearCollection(fAntiLambdaCollection);
  ClearCollection(fK0ShortCollection);

  ClearCollection(fKchPCollection);
  ClearCollection(fKchMCollection);
}

//________________________________________________________________________________________________________________
void ThermEvent::AssertAllLambdaFathersFoundDaughters()
{
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) assert(fLambdaCollection[i].BothDaughtersFound());
  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++) assert(fAntiLambdaCollection[i].BothDaughtersFound());

}


//________________________________________________________________________________________________________________
void ThermEvent::AssertAllK0FathersFound0or2Daughters()
{
  for(unsigned int i=0; i<fK0ShortCollection.size(); i++)
  {
    assert(fK0ShortCollection[i].BothDaughtersFound() || (!fK0ShortCollection[i].Daughter1Found() && !fK0ShortCollection[i].Daughter2Found()));
  }
}


//________________________________________________________________________________________________________________
//TODO this and MatchDaughtersWithFathers are inefficient!
//TODO Figure out how to deep copy so this method works!
void ThermEvent::FindFatherandLoadDaughter(ThermParticle &aDaughterParticle)
{
  int tFatherPID = aDaughterParticle.GetFatherPID();
  int tFatherEID = aDaughterParticle.GetFatherEID();

  vector<ThermV0Particle> tFatherCollection; //just a copy, when I load, I must use the actual object

  if(tFatherPID == kPDGLam) tFatherCollection = fLambdaCollection;
  else if(tFatherPID == kPDGALam) tFatherCollection = fAntiLambdaCollection;
  else if(tFatherPID == kPDGK0) tFatherCollection = fK0ShortCollection;
  else assert(0);
  //---------------------------------

  int tFatherLocation = -1;
  for(unsigned int i=0; i<tFatherCollection.size(); i++)
  {
    if(tFatherCollection[i].GetEID() == tFatherEID)
    {
      tFatherLocation = i;
      break;
    }
  }
  assert(tFatherLocation >= 0);
  //---------------------------------
  if(tFatherPID == kPDGLam) fLambdaCollection[tFatherLocation].LoadDaughter(aDaughterParticle);
  else if(tFatherPID == kPDGALam) fAntiLambdaCollection[tFatherLocation].LoadDaughter(aDaughterParticle);
  else if(tFatherPID == kPDGK0) fK0ShortCollection[tFatherLocation].LoadDaughter(aDaughterParticle);
}

//________________________________________________________________________________________________________________
//TODO this and FindFatherAndLoadDaughter are inefficient!
void ThermEvent::MatchDaughtersWithFathers()
{
  for(unsigned int i=0; i<fAllDaughtersCollection.size(); i++)
  {
    if(IsDaughterOfInterest(fAllDaughtersCollection[i]))
    {
      FindFatherandLoadDaughter(fAllDaughtersCollection[i]);
    }
  }

  AssertAllLambdaFathersFoundDaughters();
  AssertAllK0FathersFound0or2Daughters();

  //No longer need fAllDaughtersCollection, so I should clear it and free up memory before it is pushed to ThermEventsCollection
  ClearCollection(fAllDaughtersCollection);
}

//________________________________________________________________________________________________________________
void ThermEvent::FindV0Father(ThermV0Particle &aV0Particle)
{
  if(!aV0Particle.IsPrimordial())
  {
    int tFatherEID = aV0Particle.GetFatherEID();

    int tFatherLocation = -1;
    for(unsigned int i=0; i<fAllParticlesCollection.size(); i++)
    {
      if(fAllParticlesCollection[i].GetEID() == tFatherEID)
      {
        tFatherLocation = i;
        aV0Particle.LoadFather(fAllParticlesCollection[i]);
        break;
      }
    }
    assert(tFatherLocation >= 0);
  }
}


//________________________________________________________________________________________________________________
void ThermEvent::FindAllV0sFathers()
{
  for(unsigned int i=0; i<fLambdaCollection.size(); i++) FindV0Father(fLambdaCollection[i]);
  for(unsigned int i=0; i<fAntiLambdaCollection.size(); i++) FindV0Father(fAntiLambdaCollection[i]);
  for(unsigned int i=0; i<fK0ShortCollection.size(); i++) FindV0Father(fK0ShortCollection[i]);

  //No longer need fAllParticlesCollection, so I should clear it and free up memory before it is pushed to ThermEventsCollection
  ClearCollection(fAllParticlesCollection);
}

//________________________________________________________________________________________________________________
vector<ThermV0Particle> ThermEvent::GetV0ParticleCollection(ParticlePDGType aPDGType)
{
  if(aPDGType == kPDGLam) return fLambdaCollection;
  else if(aPDGType == kPDGALam) return fAntiLambdaCollection;
  else if(aPDGType == kPDGK0) return fK0ShortCollection;

  else assert(0);
}


//________________________________________________________________________________________________________________
vector<ThermParticle> ThermEvent::GetParticleCollection(ParticlePDGType aPDGType)
{

  if(aPDGType == kPDGKchP) return fKchPCollection;
  else if(aPDGType == kPDGKchM) return fKchMCollection;

  else assert(0);
}

//________________________________________________________________________________________________________________
void ThermEvent::SetV0ParticleCollection(unsigned int aEventID, ParticlePDGType aPDGType, vector<ThermV0Particle> &aCollection)
{
  assert(aEventID == fEventID);  //make sure I'm setting the collection for the correct event

  if(aPDGType == kPDGLam) fLambdaCollection = aCollection;
  else if(aPDGType == kPDGALam) fAntiLambdaCollection = aCollection;
  else if(aPDGType == kPDGK0) fK0ShortCollection = aCollection;

  else assert(0);
}


//________________________________________________________________________________________________________________
void ThermEvent::SetParticleCollection(unsigned int aEventID, ParticlePDGType aPDGType, vector<ThermParticle> &aCollection)
{
  assert(aEventID == fEventID);  //make sure I'm setting the collection for the correct event

  if(aPDGType == kPDGKchP) fKchPCollection = aCollection;
  else if(aPDGType == kPDGKchM) fKchMCollection = aCollection;

  else assert(0);
}



