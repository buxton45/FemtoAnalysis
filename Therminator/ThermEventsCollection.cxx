///////////////////////////////////////////////////////////////////////////
// ThermEventsCollection:                                                //
///////////////////////////////////////////////////////////////////////////

#include "ThermEventsCollection.h"

#ifdef __ROOT__
ClassImp(ThermEventsCollection)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermEventsCollection::ThermEventsCollection() :
  fEventsCollection(0)
{

}


//________________________________________________________________________________________________________________
ThermEventsCollection::~ThermEventsCollection()
{
  cout << "ThermEventsCollection object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
bool ThermEventsCollection::IsParticleOfInterest(ParticleCoor* tParticle)
{
  if(tParticle->pid == 3122) return true;
  else if(tParticle->pid == -3122) return true;

  else if(tParticle->pid == 310) return true;

  else if(tParticle->pid == 321) return true;
  else if(tParticle->pid == -321) return true;

  return false;
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractEventsFromRootFile(TString aFileLocation)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TTree *tTree = (TTree*)tFile->Get("particles");

  ParticleCoor *tParticleEntry = new ParticleCoor();
  TBranch *tParticleBranch = tTree->GetBranch("particle");
  tParticleBranch->SetAddress(tParticleEntry);

  int tNEvents = 1;
  unsigned int tEventID;
  ThermEvent *tThermEvent = new ThermEvent();

  for(int i=0; i<tParticleBranch->GetEntries(); i++)
  {
    tParticleBranch->GetEntry(i);


    if(i==0) tEventID = tParticleEntry->eventid;

    if(tParticleEntry->eventid != tEventID)
    {
      fEventsCollection.push_back(tThermEvent);
      tThermEvent->ClearThermEvent();

      tNEvents++;
      tEventID = tParticleEntry->eventid;
    }

    if(IsParticleOfInterest(tParticleEntry))
    {
      ThermParticle *tThermParticle = new ThermParticle(tParticleEntry);
      tThermEvent->PushBackThermParticle(tThermParticle);
    }

  }
  fEventsCollection.push_back(tThermEvent);

cout << "tNEvents = " << tNEvents << endl;
cout << "fEventsCollection.size() = " << fEventsCollection.size() << endl;

  tFile->Close();
  delete tFile;

  delete tParticleEntry;
  delete tThermEvent;
}
















