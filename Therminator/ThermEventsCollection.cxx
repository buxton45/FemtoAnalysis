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
  fEventsCollection(0),
  fSigLamKchPTransform(0)
{

}


//________________________________________________________________________________________________________________
ThermEventsCollection::~ThermEventsCollection()
{
  cout << "ThermEventsCollection object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
int ThermEventsCollection::ReturnEventIndex(int aEventID)
{
  int tEventIndex = -1;
  for(unsigned int i=0; i<fEventsCollection.size(); i++)
  {
    if(fEventsCollection[i]->GetEventID() == aEventID)
    {
      tEventIndex = i;
      break;
    }
  }

  return tEventIndex;
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteRow(ostream &aOutput, vector<double> &aRow)
{
  for(unsigned int i = 0; i < aRow.size(); i++)
  {
    if( i < aRow.size()-1) aOutput << aRow[i] << " ";
    else if(i == aRow.size()-1) aOutput << aRow[i] << endl;
    else 
    {
      cout << "SOMETHING IS WRONG!!!!!\n";
      assert(0);
    }
  }
}

//________________________________________________________________________________________________________________
vector<double> ThermEventsCollection::PackageV0ParticleForWriting(ThermV0Particle &aV0)
{
  vector<double> tReturnVector;
    tReturnVector.resize(53);
    // 18(ThermParticle) + 35(ThermV0Particle) = 53 total

  //------ThermParticle
  tReturnVector[0] = aV0.IsPrimordial();
  tReturnVector[1] = aV0.IsParticleOfInterest();

  tReturnVector[2] = aV0.GetMass();

  tReturnVector[3] = aV0.GetT();
  tReturnVector[4] = aV0.GetX();
  tReturnVector[5] = aV0.GetY();
  tReturnVector[6] = aV0.GetZ();

  tReturnVector[7] = aV0.GetE();
  tReturnVector[8] = aV0.GetPx();
  tReturnVector[9] = aV0.GetPy();
  tReturnVector[10] = aV0.GetPz();

  tReturnVector[11] = aV0.GetDecayed();
  tReturnVector[12] = aV0.GetPID();
  tReturnVector[13] = aV0.GetFatherPID();
  tReturnVector[14] = aV0.GetRootPID();
  tReturnVector[15] = aV0.GetEID();
  tReturnVector[16] = aV0.GetFatherEID();
  tReturnVector[17] = aV0.GetEventID();

  //------ThermV0Particle
  tReturnVector[18] = aV0.Daughter1Found();
  tReturnVector[19] = aV0.Daughter2Found();
  tReturnVector[20] = aV0.BothDaughtersFound();

  tReturnVector[21] = aV0.GoodV0();

  tReturnVector[22] = aV0.GetDaughter1PID();
  tReturnVector[23] = aV0.GetDaughter2PID();

  tReturnVector[24] = aV0.GetDaughter1EID();
  tReturnVector[25] = aV0.GetDaughter2EID();

  tReturnVector[26] = aV0.GetDaughter1Mass();

  tReturnVector[27] = aV0.GetDaughter1T();
  tReturnVector[28] = aV0.GetDaughter1X();
  tReturnVector[29] = aV0.GetDaughter1Y();
  tReturnVector[30] = aV0.GetDaughter1Z();

  tReturnVector[31] = aV0.GetDaughter1E();
  tReturnVector[32] = aV0.GetDaughter1Px();
  tReturnVector[33] = aV0.GetDaughter1Py();
  tReturnVector[34] = aV0.GetDaughter1Pz();

  tReturnVector[35] = aV0.GetDaughter2Mass();

  tReturnVector[36] = aV0.GetDaughter2T();
  tReturnVector[37] = aV0.GetDaughter2X();
  tReturnVector[38] = aV0.GetDaughter2Y();
  tReturnVector[39] = aV0.GetDaughter2Z();

  tReturnVector[40] = aV0.GetDaughter2E();
  tReturnVector[41] = aV0.GetDaughter2Px();
  tReturnVector[42] = aV0.GetDaughter2Py();
  tReturnVector[43] = aV0.GetDaughter2Pz();

  tReturnVector[44] = aV0.GetFatherMass();

  tReturnVector[45] = aV0.GetFatherT();
  tReturnVector[46] = aV0.GetFatherX();
  tReturnVector[47] = aV0.GetFatherY();
  tReturnVector[48] = aV0.GetFatherZ();

  tReturnVector[49] = aV0.GetFatherE();
  tReturnVector[50] = aV0.GetFatherPx();
  tReturnVector[51] = aV0.GetFatherPy();
  tReturnVector[52] = aV0.GetFatherPz();

  return tReturnVector;
}

//________________________________________________________________________________________________________________
vector<double> ThermEventsCollection::PackageParticleForWriting(ThermParticle &aParticle)
{
  vector<double> tReturnVector;
    tReturnVector.resize(18);
    // 18(ThermParticle)

  //------ThermParticle
  tReturnVector[0] = aParticle.IsPrimordial();
  tReturnVector[1] = aParticle.IsParticleOfInterest();

  tReturnVector[2] = aParticle.GetMass();

  tReturnVector[3] = aParticle.GetT();
  tReturnVector[4] = aParticle.GetX();
  tReturnVector[5] = aParticle.GetY();
  tReturnVector[6] = aParticle.GetZ();

  tReturnVector[7] = aParticle.GetE();
  tReturnVector[8] = aParticle.GetPx();
  tReturnVector[9] = aParticle.GetPy();
  tReturnVector[10] = aParticle.GetPz();

  tReturnVector[11] = aParticle.GetDecayed();
  tReturnVector[12] = aParticle.GetPID();
  tReturnVector[13] = aParticle.GetFatherPID();
  tReturnVector[14] = aParticle.GetRootPID();
  tReturnVector[15] = aParticle.GetEID();
  tReturnVector[16] = aParticle.GetFatherEID();
  tReturnVector[17] = aParticle.GetEventID();

  return tReturnVector;
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteThermEventV0s(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent* aThermEvent)
{
  vector<ThermV0Particle> tV0ParticleVec = aThermEvent->GetV0ParticleCollection(aParticleType);
  vector<double> tTempVec;

  aOutput << aThermEvent->GetEventID() << " " << aParticleType << " " << tV0ParticleVec.size() << endl;
  for(unsigned int i=0; i<tV0ParticleVec.size(); i++)
  {
    tTempVec = PackageV0ParticleForWriting(tV0ParticleVec[i]);
    WriteRow(aOutput,tTempVec);
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteThermEventParticles(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent* aThermEvent)
{
  vector<ThermParticle> tParticleVec = aThermEvent->GetParticleCollection(aParticleType);
  vector<double> tTempVec;

  aOutput << aThermEvent->GetEventID() << " " << aParticleType << " " << tParticleVec.size() << endl;
  for(unsigned int i=0; i<tParticleVec.size(); i++)
  {
    tTempVec = PackageParticleForWriting(tParticleVec[i]);
    WriteRow(aOutput,tTempVec);
  }
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteAllEventsParticlesOfType(TString aOutputName, ParticlePDGType aParticleType)
{
  ofstream tFileOut(aOutputName);

  for(unsigned int i=0; i<fEventsCollection.size(); i++)
  {
    if(aParticleType == kPDGLam || aParticleType == kPDGALam || aParticleType == kPDGK0) WriteThermEventV0s(tFileOut,aParticleType,fEventsCollection[i]);
    else if(aParticleType == kPDGKchP || aParticleType == kPDGKchM) WriteThermEventParticles(tFileOut,aParticleType,fEventsCollection[i]);
    else assert(0);
  }

  tFileOut.close();
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteAllEvents(TString aOutputNameBase)
{
  TString tNameLam = aOutputNameBase + TString("Lambda.txt");
  TString tNameALam = aOutputNameBase + TString("AntiLambda.txt");
  TString tNameK0 = aOutputNameBase + TString("K0.txt");
  TString tNameKchP = aOutputNameBase + TString("KchP.txt");
  TString tNameKchM = aOutputNameBase + TString("KchM.txt");

  WriteAllEventsParticlesOfType(tNameLam,kPDGLam);
    cout << "Done writing file: " << tNameLam << endl;

  WriteAllEventsParticlesOfType(tNameALam,kPDGALam);
    cout << "Done writing file: " << tNameALam << endl;

  WriteAllEventsParticlesOfType(tNameK0,kPDGK0);
    cout << "Done writing file: " << tNameK0 << endl;

  WriteAllEventsParticlesOfType(tNameKchP,kPDGKchP);
    cout << "Done writing file: " << tNameKchP << endl;

  WriteAllEventsParticlesOfType(tNameKchM,kPDGKchM);
    cout << "Done writing file: " << tNameKchM << endl;
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractV0ParticleCollectionsFromTxtFile(TString aFileName, ParticlePDGType aPDGType)
{
  assert(aPDGType == kPDGLam || aPDGType == kPDGALam || aPDGType == kPDGK0);

  ifstream tFileIn(aFileName);

  vector<ThermV0Particle> tTempV0Collection;
  vector<double> tTempParticle1dVec;

  int tEventID = 0;
  int tEventIndex;
  unsigned int tEntrySize = 53;  //size of V0 particle vector

  string tString;
  int tCount = 0;
  while(getline(tFileIn, tString))
  {
    tTempParticle1dVec.clear();
    istringstream tStream(tString);
    string tElement;
    while(tStream >> tElement)
    {
      stringstream ss (tElement);
      double dbl;
      ss >> dbl;
      tTempParticle1dVec.push_back(dbl);
    }

    if(tTempParticle1dVec.size() == 3)  //event header
    {           
      tCount++;
      
      if(tCount==1) tEventID = tTempParticle1dVec[0];
      else
      {
        tEventIndex = ReturnEventIndex(tEventID);
        if(tEventIndex == -1)  //Event does not already exist in fEventsCollection
        {
          ThermEvent* tThermEvent = new ThermEvent();
          tThermEvent->SetEventID(tEventID);
          tThermEvent->SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
          fEventsCollection.push_back(tThermEvent);
        }
        else  //Event already exists in fEventsCollection, so simply add particle collection to it
        {
          fEventsCollection[tEventIndex]->SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
        }
        tTempV0Collection.clear();
      }
    tEventID = tTempParticle1dVec[0];
    }
    else if(tTempParticle1dVec.size() == tEntrySize)  //a V0 particle
    {
      tTempV0Collection.emplace_back(tTempParticle1dVec);
    }
    else
    {
      cout << "ERROR: Incorrect row size in ExtractV0ParticleCollectionsFromTxtFile" << endl;
      assert(0);
    }
  }
  tEventIndex = ReturnEventIndex(tEventID);
  if(tEventIndex == -1)
  {
    ThermEvent* tThermEvent = new ThermEvent();
    tThermEvent->SetEventID(tEventID);
    tThermEvent->SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
    fEventsCollection.push_back(tThermEvent);
  }
  else
  {
    fEventsCollection[tEventIndex]->SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
  }
  tTempV0Collection.clear();

}

//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractParticleCollectionsFromTxtFile(TString aFileName, ParticlePDGType aPDGType)
{
  assert(aPDGType == kPDGKchP || aPDGType == kPDGKchM);

  ifstream tFileIn(aFileName);

  vector<ThermParticle> tTempCollection;
  vector<double> tTempParticle1dVec;

  int tEventID = 0;
  int tEventIndex;
  unsigned int tEntrySize = 18;  //size of particle vector

  string tString;
  int tCount = 0;
  while(getline(tFileIn, tString))
  {
    tTempParticle1dVec.clear();
    istringstream tStream(tString);
    string tElement;
    while(tStream >> tElement)
    {
      stringstream ss (tElement);
      double dbl;
      ss >> dbl;
      tTempParticle1dVec.push_back(dbl);
    }

    if(tTempParticle1dVec.size() == 3)  //event header
    {           
      tCount++;
      
      if(tCount==1) tEventID = tTempParticle1dVec[0];
      else
      {
        tEventIndex = ReturnEventIndex(tEventID);
        if(tEventIndex == -1)  //Event does not already exist in fEventsCollection
        {
          ThermEvent* tThermEvent = new ThermEvent();
          tThermEvent->SetEventID(tEventID);
          tThermEvent->SetParticleCollection(tEventID,aPDGType,tTempCollection);
          fEventsCollection.push_back(tThermEvent);
        }
        else  //Event already exists in fEventsCollection, so simply add particle collection to it
        {
          fEventsCollection[tEventIndex]->SetParticleCollection(tEventID,aPDGType,tTempCollection);
        }
        tTempCollection.clear();
      }
    tEventID = tTempParticle1dVec[0];
    }
    else if(tTempParticle1dVec.size() == tEntrySize)  //a particle
    {
      tTempCollection.emplace_back(tTempParticle1dVec);
    }
    else
    {
      cout << "ERROR: Incorrect row size in ExtractParticleCollectionsFromTxtFile" << endl;
      assert(0);
    }
  }
  tEventIndex = ReturnEventIndex(tEventID);
  if(tEventIndex == -1)
  {
    ThermEvent* tThermEvent = new ThermEvent();
    tThermEvent->SetEventID(tEventID);
    tThermEvent->SetParticleCollection(tEventID,aPDGType,tTempCollection);
    fEventsCollection.push_back(tThermEvent);
  }
  else
  {
    fEventsCollection[tEventIndex]->SetParticleCollection(tEventID,aPDGType,tTempCollection);
  }
  tTempCollection.clear();

}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractEventsFromAllTxtFiles(TString aFileLocationBase)
{
  TString tNameLam = aFileLocationBase + TString("Lambda.txt");
  TString tNameALam = aFileLocationBase + TString("AntiLambda.txt");
  TString tNameK0 = aFileLocationBase + TString("K0.txt");
  TString tNameKchP = aFileLocationBase + TString("KchP.txt");
  TString tNameKchM = aFileLocationBase + TString("KchM.txt");

  ExtractV0ParticleCollectionsFromTxtFile(tNameLam,kPDGLam);
    cout << "Done reading file: " << tNameLam << endl;

  ExtractV0ParticleCollectionsFromTxtFile(tNameALam,kPDGALam);
    cout << "Done reading file: " << tNameALam << endl;

  ExtractV0ParticleCollectionsFromTxtFile(tNameK0,kPDGK0);
    cout << "Done reading file: " << tNameK0 << endl;

  ExtractParticleCollectionsFromTxtFile(tNameKchP,kPDGKchP);
    cout << "Done reading file: " << tNameKchP << endl;

  ExtractParticleCollectionsFromTxtFile(tNameKchM,kPDGKchM);
    cout << "Done reading file: " << tNameKchM << endl;
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
      tThermEvent->MatchDaughtersWithFathers();
      tThermEvent->FindAllV0sFathers();
      tThermEvent->SetEventID(tEventID);
      fEventsCollection.push_back(tThermEvent);
      tThermEvent->ClearThermEvent();

      tNEvents++;
      tEventID = tParticleEntry->eventid;
    }

    tThermEvent->PushBackThermParticle(tParticleEntry);
    if(tThermEvent->IsDaughterOfInterest(tParticleEntry)) tThermEvent->PushBackThermDaughterOfInterest(tParticleEntry);
    if(tThermEvent->IsParticleOfInterest(tParticleEntry)) tThermEvent->PushBackThermParticleOfInterest(tParticleEntry);
  }
  tThermEvent->MatchDaughtersWithFathers();
  tThermEvent->FindAllV0sFathers();
  tThermEvent->SetEventID(tEventID);
  fEventsCollection.push_back(tThermEvent);

cout << "aFileLocation = " << aFileLocation << endl;
cout << "tNEvents = " << tNEvents << endl;
cout << "fEventsCollection.size() = " << fEventsCollection.size() << endl << endl;

  tFile->Close();
  delete tFile;

//  delete tParticleEntry;
//  delete tThermEvent;
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractFromAllRootFiles(const char *aDirName)
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


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildTransformMatrices()
{
  fSigLamKchPTransform = new TH2D("fSigLamKchPTransform","fSigLamKchPTransform",200,0.,1.,200,0.,1.);

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    fEventsCollection[iEv]->FillTransformMatrix(fSigLamKchPTransform);
  }

  fSigLamKchPTransform->Draw("colz");
}










