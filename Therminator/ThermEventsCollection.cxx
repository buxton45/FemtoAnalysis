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
  fFileNameCollection(0),
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

  else if(tParticle->pid == 311) return true;

  else if(tParticle->pid == 211) return true;
  else if(tParticle->pid == -211) return true;

  else if(tParticle->pid == 321) return true;
  else if(tParticle->pid == -321) return true;

  else if(tParticle->pid == 2212) return true;
  else if(tParticle->pid == -2212) return true;

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
      tThermEvent->MatchAllDaughtersWithFathers();
      tThermEvent->AssertAllFathersFoundDaughters();
      fEventsCollection.push_back(tThermEvent);
      tThermEvent->ClearThermEvent();

      tNEvents++;
      tEventID = tParticleEntry->eventid;
    }

    ThermParticle *tThermParticle = new ThermParticle(tParticleEntry);
    tThermEvent->PushBackThermParticle(tThermParticle);
    if(IsParticleOfInterest(tParticleEntry))
    {
//      ThermParticle *tThermParticle = new ThermParticle(tParticleEntry);
      tThermEvent->PushBackThermParticleOfInterest(tThermParticle);
    }

  }
  tThermEvent->MatchAllDaughtersWithFathers();
  tThermEvent->AssertAllFathersFoundDaughters();
  fEventsCollection.push_back(tThermEvent);

//cout << "aFileLocation = " << aFileLocation << endl;
//cout << "tNEvents = " << tNEvents << endl;
//cout << "fEventsCollection.size() = " << fEventsCollection.size() << endl << endl;

  tFile->Close();
  delete tFile;

//  delete tParticleEntry;
//  delete tThermEvent;
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractFromAllFiles(const char *aDirName)
{
  fFileNameCollection.clear();
  TString tCompleteFilePath;

  TSystemDirectory tDir(aDirName,aDirName);
  TList* tFiles = tDir.GetListOfFiles();

  const char* tBeginningText = "event";
  const char* tEndingText = ".root";

  int tNFiles = 0;

  if(tFiles)
  {
    TSystemFile* tFile;
    TString tName;
    TIter tIterNext(tFiles);

    while((tFile=(TSystemFile*)tIterNext()))
    {
      tName = tFile->GetName();
      if(!tFile->IsDirectory() && tName.BeginsWith(tBeginningText) && tName.EndsWith(tEndingText))
      {
        tNFiles++;
        fFileNameCollection.push_back(tName);
        tCompleteFilePath = TString(aDirName) + tName;
        ExtractEventsFromRootFile(tCompleteFilePath);
      }
    }
  }
  cout << "Total number of files = " << tNFiles << endl;
  cout << "Total number of events = " << fEventsCollection.size() << endl << endl;
}













