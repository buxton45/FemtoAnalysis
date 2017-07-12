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
  fNFiles(0),
  fNEvents(0),
  fFileNameCollection(0),
  fEventsCollection(0),

  fMixEvents(false),
  fNEventsToMix(5),
  fMixingEventsCollection(0),

  fKStarMin(0.),
  fKStarMax(1.),
  fNBinsKStar(200),

  fBuildUniqueParents(false),
  fUniqueV0Parents(0),
  fUniqueTrackParents(0),

  //LamKchP
  fSigToLamKchPTransform(0),
  fXiCToLamKchPTransform(0),
  fXi0ToLamKchPTransform(0),
  fOmegaToLamKchPTransform(0),
  fPairFractionsLamKchP(0),
  fParentsMatrixLamKchP(0),

  //ALamKchP
  fASigToALamKchPTransform(0),
  fAXiCToALamKchPTransform(0),
  fAXi0ToALamKchPTransform(0),
  fAOmegaToALamKchPTransform(0),
  fPairFractionsALamKchP(0),
  fParentsMatrixALamKchP(0),

  //LamKchM
  fSigToLamKchMTransform(0),
  fXiCToLamKchMTransform(0),
  fXi0ToLamKchMTransform(0),
  fOmegaToLamKchMTransform(0),
  fPairFractionsLamKchM(0),
  fParentsMatrixLamKchM(0),

  //ALamKchM
  fASigToALamKchMTransform(0),
  fAXiCToALamKchMTransform(0),
  fAXi0ToALamKchMTransform(0),
  fAOmegaToALamKchMTransform(0),
  fPairFractionsALamKchM(0),
  fParentsMatrixALamKchM(0),

  fSigToLamLamTransform(0)

{
  //LamKchP
  fSigToLamKchPTransform = new TH2D("fSigToLamKchPTransform","fSigToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXiCToLamKchPTransform = new TH2D("fXiCToLamKchPTransform","fXiCToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXi0ToLamKchPTransform = new TH2D("fXi0ToLamKchPTransform","fXi0ToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fOmegaToLamKchPTransform = new TH2D("fOmegaToLamKchPTransform","fOmegaToLamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fPairFractionsLamKchP = new TH1D("fPairFractionsLamKchP", "fPairFractionsLamKchP", 6, 0, 6);
  fParentsMatrixLamKchP = new TH2D("fParentsMatrixLamKchP", "fParentsMatrixLamKchP", 100, 0, 100, 100, 0, 100);

  //ALamKchP
  fASigToALamKchPTransform = new TH2D("fASigToALamKchPTransform","fASigToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXiCToALamKchPTransform = new TH2D("fAXiCToALamKchPTransform","fAXiCToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXi0ToALamKchPTransform = new TH2D("fAXi0ToALamKchPTransform","fAXi0ToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAOmegaToALamKchPTransform = new TH2D("fAOmegaToALamKchPTransform","fAOmegaToALamKchPTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fPairFractionsALamKchP = new TH1D("fPairFractionsALamKchP", "fPairFractionsALamKchP", 6, 0, 6);
  fParentsMatrixALamKchP = new TH2D("fParentsMatrixALamKchP", "fParentsMatrixALamKchP", 100, 0, 100, 100, 0, 100);

  //LamKchM
  fSigToLamKchMTransform = new TH2D("fSigToLamKchMTransform","fSigToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXiCToLamKchMTransform = new TH2D("fXiCToLamKchMTransform","fXiCToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fXi0ToLamKchMTransform = new TH2D("fXi0ToLamKchMTransform","fXi0ToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fOmegaToLamKchMTransform = new TH2D("fOmegaToLamKchMTransform","fOmegaToLamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fPairFractionsLamKchM = new TH1D("fPairFractionsLamKchM", "fPairFractionsLamKchM", 6, 0, 6);
  fParentsMatrixLamKchM = new TH2D("fParentsMatrixLamKchM", "fParentsMatrixLamKchM", 100, 0, 100, 100, 0, 100);

  //ALamKchM
  fASigToALamKchMTransform = new TH2D("fASigToALamKchMTransform","fASigToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXiCToALamKchMTransform = new TH2D("fAXiCToALamKchMTransform","fAXiCToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAXi0ToALamKchMTransform = new TH2D("fAXi0ToALamKchMTransform","fAXi0ToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fAOmegaToALamKchMTransform = new TH2D("fAOmegaToALamKchMTransform","fAOmegaToALamKchMTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
  fPairFractionsALamKchM = new TH1D("fPairFractionsALamKchM", "fPairFractionsALamKchM", 6, 0, 6);
  fParentsMatrixALamKchM = new TH2D("fParentsMatrixALamKchM", "fParentsMatrixALamKchM", 100, 0, 100, 100, 0, 100);

  //LamLam
  fSigToLamLamTransform = new TH2D("fSigToLamLamTransform","fSigToLamLamTransform",fNBinsKStar,fKStarMin,fKStarMax,fNBinsKStar,fKStarMin,fKStarMax);
}


//________________________________________________________________________________________________________________
ThermEventsCollection::~ThermEventsCollection()
{
  cout << "ThermEventsCollection object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
int ThermEventsCollection::ReturnEventIndex(unsigned int aEventID)
{
  int tEventIndex = -1;
  for(unsigned int i=0; i<fEventsCollection.size(); i++)
  {
    if(fEventsCollection[i].GetEventID() == aEventID)
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
void ThermEventsCollection::WriteThermEventV0s(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent aThermEvent)
{
  vector<ThermV0Particle> tV0ParticleVec = aThermEvent.GetV0ParticleCollection(aParticleType);
  vector<double> tTempVec;

  aOutput << aThermEvent.GetEventID() << " " << aParticleType << " " << tV0ParticleVec.size() << endl;
  for(unsigned int i=0; i<tV0ParticleVec.size(); i++)
  {
    tTempVec = PackageV0ParticleForWriting(tV0ParticleVec[i]);
    WriteRow(aOutput,tTempVec);
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::WriteThermEventParticles(ostream &aOutput, ParticlePDGType aParticleType, ThermEvent aThermEvent)
{
  vector<ThermParticle> tParticleVec = aThermEvent.GetParticleCollection(aParticleType);
  vector<double> tTempVec;

  aOutput << aThermEvent.GetEventID() << " " << aParticleType << " " << tParticleVec.size() << endl;
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

  unsigned int tEventID = 0;
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
          ThermEvent tThermEvent;
          tThermEvent.SetEventID(tEventID);
          tThermEvent.SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
          fEventsCollection.push_back(tThermEvent);
        }
        else  //Event already exists in fEventsCollection, so simply add particle collection to it
        {
          fEventsCollection[tEventIndex].SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
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
    ThermEvent tThermEvent;
    tThermEvent.SetEventID(tEventID);
    tThermEvent.SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
    fEventsCollection.push_back(tThermEvent);
  }
  else
  {
    fEventsCollection[tEventIndex].SetV0ParticleCollection(tEventID,aPDGType,tTempV0Collection);
  }
  tTempV0Collection.clear();
  tFileIn.close();
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractParticleCollectionsFromTxtFile(TString aFileName, ParticlePDGType aPDGType)
{
  assert(aPDGType == kPDGKchP || aPDGType == kPDGKchM);

  ifstream tFileIn(aFileName);

  vector<ThermParticle> tTempCollection;
  vector<double> tTempParticle1dVec;

  unsigned int tEventID = 0;
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
          ThermEvent tThermEvent;
          tThermEvent.SetEventID(tEventID);
          tThermEvent.SetParticleCollection(tEventID,aPDGType,tTempCollection);
          fEventsCollection.push_back(tThermEvent);
        }
        else  //Event already exists in fEventsCollection, so simply add particle collection to it
        {
          fEventsCollection[tEventIndex].SetParticleCollection(tEventID,aPDGType,tTempCollection);
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
    ThermEvent tThermEvent;
    tThermEvent.SetEventID(tEventID);
    tThermEvent.SetParticleCollection(tEventID,aPDGType,tTempCollection);
    fEventsCollection.push_back(tThermEvent);
  }
  else
  {
    fEventsCollection[tEventIndex].SetParticleCollection(tEventID,aPDGType,tTempCollection);
  }
  tTempCollection.clear();
  tFileIn.close();
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
  ThermEvent tThermEvent;

  for(int i=0; i<tParticleBranch->GetEntries(); i++)
  {
    tParticleBranch->GetEntry(i);

    if(i==0) tEventID = tParticleEntry->eventid;

    if(tParticleEntry->eventid != tEventID)
    {
      tThermEvent.MatchDaughtersWithFathers();
      tThermEvent.FindAllV0sFathers();
      tThermEvent.SetEventID(tEventID);
      fEventsCollection.push_back(tThermEvent);
      tThermEvent.ClearThermEvent();

      tNEvents++;
      tEventID = tParticleEntry->eventid;
    }

    tThermEvent.PushBackThermParticle(tParticleEntry);
    if(tThermEvent.IsDaughterOfInterest(tParticleEntry)) tThermEvent.PushBackThermDaughterOfInterest(tParticleEntry);
    if(tThermEvent.IsParticleOfInterest(tParticleEntry)) tThermEvent.PushBackThermParticleOfInterest(tParticleEntry);
  }
  tThermEvent.MatchDaughtersWithFathers();
  tThermEvent.FindAllV0sFathers();
  tThermEvent.SetEventID(tEventID);
  fEventsCollection.push_back(tThermEvent);

  fNEvents += tNEvents;

cout << "aFileLocation = " << aFileLocation << endl;
cout << "fEventsCollection.size() = " << fEventsCollection.size() << endl;
cout << "fNEvents = " << fNEvents << endl;


  tFile->Close();
  delete tFile;

  delete tParticleEntry;
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::ExtractFromAllRootFiles(const char *aDirName, bool bBuildUniqueParents)
{
  fBuildUniqueParents = bBuildUniqueParents;

  fFileNameCollection.clear();
  TString tCompleteFilePath;

  TSystemDirectory tDir(aDirName,aDirName);
  TList* tFiles = tDir.GetListOfFiles();

  const char* tBeginningText = "event";
  const char* tEndingText = ".root";

  fNFiles = 0;

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
        fNFiles++;
        fFileNameCollection.push_back(tName);
        tCompleteFilePath = TString(aDirName) + tName;
        ExtractEventsFromRootFile(tCompleteFilePath);

        BuildAllTransformMatrices();
        BuildAllPairFractionHistograms();
        fEventsCollection.clear();
        fEventsCollection.shrink_to_fit();

        cout << "fNFiles = " << fNFiles << endl << endl;
      }
    }
  }
  cout << "Total number of files = " << fNFiles << endl;
  cout << "Total number of events = " << fNEvents << endl << endl;
}


//________________________________________________________________________________________________________________
bool ThermEventsCollection::DoubleCheckLamAttributes(ThermV0Particle &aV0)
{
  if(!aV0.GoodV0()) {cout << "DoubleCheckLamAttributes Fail 1" << endl; return false;}
  if(aV0.GetPID()!= kPDGLam) {cout << "DoubleCheckLamAttributes Fail 2" << endl; return false;}
  if(!aV0.BothDaughtersFound()) {cout << "DoubleCheckLamAttributes Fail 3" << endl; return false;}
  if(aV0.GetDaughter1PID() != kPDGProt) {cout << "DoubleCheckLamAttributes Fail 4" << endl; return false;}
  if(aV0.GetDaughter2PID() != kPDGPiM) {cout << "DoubleCheckLamAttributes Fail 5" << endl; return false;}

  return true;
}

//________________________________________________________________________________________________________________
bool ThermEventsCollection::DoubleCheckALamAttributes(ThermV0Particle &aV0)
{
  if(!aV0.GoodV0()) {cout << "DoubleCheckALamAttributes Fail 1" << endl; return false;}
  if(aV0.GetPID()!= kPDGALam) {cout << "DoubleCheckALamAttributes Fail 2" << endl; return false;}
  if(!aV0.BothDaughtersFound()) {cout << "DoubleCheckALamAttributes Fail 3" << endl; return false;}
  if(aV0.GetDaughter1PID() != kPDGPiP) {cout << "DoubleCheckALamAttributes Fail 4" << endl; return false;}
  if(aV0.GetDaughter2PID() != kPDGAntiProt) {cout << "DoubleCheckALamAttributes Fail 5" << endl; return false;}

  return true;
}

//________________________________________________________________________________________________________________
bool ThermEventsCollection::DoubleCheckK0Attributes(ThermV0Particle &aV0)
{
  if(!aV0.GoodV0()) {cout << "DoubleCheckK0Attributes Fail 1" << endl; return false;}
  if(aV0.GetPID()!= kPDGK0) {cout << "DoubleCheckK0Attributes Fail 2" << endl; return false;}
  if(!aV0.BothDaughtersFound()) {cout << "DoubleCheckK0Attributes Fail 3" << endl; return false;}
  if(aV0.GetDaughter1PID() != kPDGPiP) {cout << "DoubleCheckK0Attributes Fail 4" << endl; return false;}
  if(aV0.GetDaughter2PID() != kPDGPiM) {cout << "DoubleCheckK0Attributes Fail 5" << endl; return false;}

  return true;
}



//________________________________________________________________________________________________________________
bool ThermEventsCollection::DoubleCheckV0Attributes(ThermV0Particle &aV0)
{
  //------------------------------
  if(aV0.GetDaughter1Mass()==0) {cout << "DoubleCheckV0Attributes Fail 1" << endl; return false;}
  if(aV0.GetDaughter1T()==0 || aV0.GetDaughter1X()==0 || aV0.GetDaughter1Y()==0 ||aV0.GetDaughter1Z()==0) {cout << "DoubleCheckV0Attributes Fail 2" << endl; return false;}
  if(aV0.GetDaughter1E()==0 || aV0.GetDaughter1Px()==0 || aV0.GetDaughter1Py()==0 ||aV0.GetDaughter1Pz()==0) {cout << "DoubleCheckV0Attributes Fail 3" << endl; return false;}

  if(aV0.GetDaughter2Mass()==0) {cout << "DoubleCheckV0Attributes Fail 4" << endl; return false;}
  if(aV0.GetDaughter2T()==0 || aV0.GetDaughter2X()==0 || aV0.GetDaughter2Y()==0 ||aV0.GetDaughter2Z()==0) {cout << "DoubleCheckV0Attributes Fail 5" << endl; return false;}
  if(aV0.GetDaughter2E()==0 || aV0.GetDaughter2Px()==0 || aV0.GetDaughter2Py()==0 ||aV0.GetDaughter2Pz()==0) {cout << "DoubleCheckV0Attributes Fail 6" << endl; return false;}

  if(aV0.GetFatherT()==0 || aV0.GetFatherX()==0 || aV0.GetFatherY()==0 ||aV0.GetFatherZ()==0) {cout << "DoubleCheckV0Attributes Fail 7" << endl; return false;}
  if(aV0.GetFatherE()==0 || aV0.GetFatherPx()==0 || aV0.GetFatherPy()==0 ||aV0.GetFatherPz()==0) {cout << "DoubleCheckV0Attributes Fail 8" << endl; return false;}

  //------------------------------
  if(aV0.GetPID() == kPDGLam) return DoubleCheckLamAttributes(aV0);
  else if(aV0.GetPID() == kPDGALam) return DoubleCheckALamAttributes(aV0);
  else if(aV0.GetPID() == kPDGK0) return DoubleCheckK0Attributes(aV0);
  else assert(0);

}

//________________________________________________________________________________________________________________
double ThermEventsCollection::GetKStar(ThermParticle &aParticle, ThermV0Particle &aV0)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  px1 = aParticle.GetPx();
  py1 = aParticle.GetPy();
  pz1 = aParticle.GetPz();
  mass1 = aParticle.GetMass();
  E1 = aParticle.GetE();

  px2 = aV0.GetPx();
  py2 = aV0.GetPy();
  pz2 = aV0.GetPz();
  mass2 = aV0.GetMass();
  E2 = aV0.GetE();

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
double ThermEventsCollection::GetKStar(ThermV0Particle &aV01, ThermV0Particle &aV02)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  px1 = aV01.GetPx();
  py1 = aV01.GetPy();
  pz1 = aV01.GetPz();
  mass1 = aV01.GetMass();
  E1 = aV01.GetE();

  px2 = aV02.GetPx();
  py2 = aV02.GetPy();
  pz2 = aV02.GetPz();
  mass2 = aV02.GetMass();
  E2 = aV02.GetE();

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
double ThermEventsCollection::GetFatherKStar(ThermParticle &aParticle, ThermV0Particle &aV0)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  px1 = aParticle.GetPx();
  py1 = aParticle.GetPy();
  pz1 = aParticle.GetPz();
  mass1 = aParticle.GetMass();
  E1 = aParticle.GetE();

  px2 = aV0.GetFatherPx();
  py2 = aV0.GetFatherPy();
  pz2 = aV0.GetFatherPz();
  mass2 = aV0.GetFatherMass();
  E2 = aV0.GetFatherE();

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
double ThermEventsCollection::GetFatherKStar(ThermV0Particle &aV01, ThermV0Particle &aV02)
{
  double px1, py1, pz1, mass1, E1;
  double px2, py2, pz2, mass2, E2;

  px1 = aV01.GetFatherPx();
  py1 = aV01.GetFatherPy();
  pz1 = aV01.GetFatherPz();
  mass1 = aV01.GetFatherMass();
  E1 = aV01.GetFatherE();

  px2 = aV02.GetPx();
  py2 = aV02.GetPy();
  pz2 = aV02.GetPz();
  mass2 = aV02.GetMass();
  E2 = aV02.GetE();

  double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);
  double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);
  double tKStar = 0.5*sqrt(tQinvSq);

  return tKStar;
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::FillTransformMatrixParticleV0(vector<ThermParticle> &aParticleCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aFatherType, TH2* aMatrix)
{
  ThermParticle tParticle;
  ThermV0Particle tV0;
  double tKStar, tFatherKStar;

  for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
  {
    tV0 = aV0Collection[iV0];
    if(tV0.GetFatherPID() == aFatherType && tV0.GoodV0())
    {
      for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
      {
        tParticle = aParticleCollection[iPar];

        tKStar = GetKStar(tParticle,tV0);
        tFatherKStar = GetFatherKStar(tParticle,tV0);

        assert(DoubleCheckV0Attributes(tV0));
        aMatrix->Fill(tKStar,tFatherKStar);
      }
    }
  }
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::FillTransformMatrixV0V0(vector<ThermV0Particle> &aV0wFatherCollection, vector<ThermV0Particle> &aV0Collection, ParticlePDGType aFatherType, TH2* aMatrix)
{
  ThermV0Particle tV01, tV02;
  double tKStar, tFatherKStar;

  for(unsigned int iV01=0; iV01<aV0wFatherCollection.size(); iV01++)
  {
    tV01 = aV0wFatherCollection[iV01];
    if(tV01.GetFatherPID() == aFatherType && tV01.GoodV0())
    {
      for(unsigned int iV02=0; iV02<aV0Collection.size(); iV02++)
      {
        tV02 = aV0Collection[iV02];
        if(tV02.GoodV0() && !(tV02.GetEID()==tV01.GetEID() && tV02.GetEventID()==tV02.GetEventID()) ) //For instance, if I am doing LamLam w/o mixing events, I do not want to pair a Lam with itself
        {
          tKStar = GetKStar(tV01,tV02);
          tFatherKStar = GetFatherKStar(tV01,tV02);

          assert(DoubleCheckV0Attributes(tV01) /*&& DoubleCheckV0Attributes(tV02)*/);  //TODO many tV02 will fail because they are primordial and do not have fathers
											//  However, this DoubleCheckV0Attributes was mainly for debugging, and can probably be removed
          aMatrix->Fill(tKStar,tFatherKStar);
        }
      }
    }
  }
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildTransformMatrixParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, ParticlePDGType aFatherType, TH2* aMatrix)
{
  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aV0Collection =  fEventsCollection[iEv].GetV0ParticleCollection(aV0Type);
    if(!fMixEvents)
    {
      fMixingEventsCollection.clear();
      fMixingEventsCollection.push_back(fEventsCollection[iEv]);
    }
    for(unsigned int iMixEv=0; iMixEv < fMixingEventsCollection.size(); iMixEv++)
    {
      aParticleCollection = fMixingEventsCollection[iMixEv].GetParticleCollection(aParticleType);
      FillTransformMatrixParticleV0(aParticleCollection,aV0Collection,aFatherType,aMatrix);
    }

    if(fMixEvents)
    {
      assert(fMixingEventsCollection.size() <= fNEventsToMix);
      if(fMixingEventsCollection.size() == fNEventsToMix)
      {
        //delete fMixingEventsCollection.back();
        fMixingEventsCollection.pop_back();
      }

      fMixingEventsCollection.insert(fMixingEventsCollection.begin(), fEventsCollection[iEv]);
    }
  }

}



//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildTransformMatrixV0V0(ParticlePDGType aV0wFatherType, ParticlePDGType aV0Type, ParticlePDGType aFatherType, TH2* aMatrix)
{
  vector<ThermV0Particle> aV0wFatherCollection;
  vector<ThermV0Particle> aV0Collection;

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aV0wFatherCollection =  fEventsCollection[iEv].GetV0ParticleCollection(aV0wFatherType);
    if(!fMixEvents)
    {
      fMixingEventsCollection.clear();
      fMixingEventsCollection.push_back(fEventsCollection[iEv]);
    }
    for(unsigned int iMixEv=0; iMixEv < fMixingEventsCollection.size(); iMixEv++)
    {
      aV0Collection = fMixingEventsCollection[iMixEv].GetV0ParticleCollection(aV0Type);
      FillTransformMatrixV0V0(aV0wFatherCollection,aV0Collection,aFatherType,aMatrix);
    }

    if(fMixEvents)
    {
      assert(fMixingEventsCollection.size() <= fNEventsToMix);
      if(fMixingEventsCollection.size() == fNEventsToMix)
      {
        //delete fMixingEventsCollection.back();
        fMixingEventsCollection.pop_back();
      }

      fMixingEventsCollection.insert(fMixingEventsCollection.begin(), fEventsCollection[iEv]);
    }
  }
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildAllTransformMatrices()
{
  //LamKchP
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGSigma, fSigToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGXiC, fXiCToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGXi0, fXi0ToLamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGLam, kPDGOmega, fOmegaToLamKchPTransform);

  //ALamKchP
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGASigma, fASigToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGAXiC, fAXiCToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGAXi0, fAXi0ToALamKchPTransform);
  BuildTransformMatrixParticleV0(kPDGKchP, kPDGALam, kPDGAOmega, fAOmegaToALamKchPTransform);

  //LamKchM
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGSigma, fSigToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGXiC, fXiCToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGXi0, fXi0ToLamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGLam, kPDGOmega, fOmegaToLamKchMTransform);

  //ALamKchM
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGASigma, fASigToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGAXiC, fAXiCToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGAXi0, fAXi0ToALamKchMTransform);
  BuildTransformMatrixParticleV0(kPDGKchM, kPDGALam, kPDGAOmega, fAOmegaToALamKchMTransform);

  //LamLam
  BuildTransformMatrixV0V0(kPDGLam, kPDGLam, kPDGSigma, fSigToLamLamTransform);
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::SaveAllTransformMatrices(TString aSaveFileLocation)
{
  TFile *tFile = new TFile(aSaveFileLocation, "RECREATE");
  assert(tFile->IsOpen());


  //LamKchP
  fSigToLamKchPTransform->Write();
  fXiCToLamKchPTransform->Write();
  fXi0ToLamKchPTransform->Write();
  fOmegaToLamKchPTransform->Write();

  //ALamKchP
  fASigToALamKchPTransform->Write();
  fAXiCToALamKchPTransform->Write();
  fAXi0ToALamKchPTransform->Write();
  fAOmegaToALamKchPTransform->Write();

  //LamKchM
  fSigToLamKchMTransform->Write();
  fXiCToLamKchMTransform->Write();
  fXi0ToLamKchMTransform->Write();
  fOmegaToLamKchMTransform->Write();

  //ALamKchM
  fASigToALamKchMTransform->Write();
  fAXiCToALamKchMTransform->Write();
  fAXi0ToALamKchMTransform->Write();
  fAOmegaToALamKchMTransform->Write();

  //LamLam to check with Jai
  fSigToLamLamTransform->Write();

  tFile->Close();
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::MapAndFillParentsMatrix(TH2* aMatrix, int aV0FatherType, int aTrackFatherType)
{
  //Note: List of parent PIDs found by turning on bBuildUniqueParents switch in ThermEventsCollection::ExtractFromAllRootFiles
  vector<int> tV0Fathers {3114, 3116, 3118, 3122, 3124, 3212, 3214, 3216, 3218, 3224, 3226, 3228, 3312, 3322, 3334, 4028, 4128, 4228, 8116, 8117, 8118, 8900, 8901, 13112, 13114, 13116, 13124, 13212, 13214, 13216, 13222, 13224, 13226, 13314, 13324, 23114, 23214, 23224, 31214, 32112, 32124, 32212, 33122, 42112, 42212, 67000, 67001, 67718, 67719};

  vector<int> tTrackFathers {115, 119, 215, 219, 313, 317, 321, 323, 327, 333, 335, 337, 3118, 3124, 3126, 3128, 3216, 3218, 3226, 3228, 3334, 4128, 4228, 8117, 8118, 8900, 8901, 9000, 10111, 10115, 10211, 10215, 10221, 10311, 10313, 10321, 10323, 10331, 13124, 13126, 13212, 13214, 13216, 13222, 13224, 13226, 13314, 13324, 20223, 20313, 20315, 20323, 20325, 20333, 23114, 23122, 23124, 23214, 23224, 30313, 30323, 31214, 32112, 33122, 42112, 43122, 53122, 67001, 67718, 67719, 100313, 100323, 100331, 100333, 9000223};

  int tBinV0Father=-1, tBinTrackFather=-1;
  for(unsigned int i=0; i<tV0Fathers.size(); i++) if(aV0FatherType==tV0Fathers[i]) tBinV0Father=i;
  for(unsigned int i=0; i<tTrackFathers.size(); i++) if(aTrackFatherType==tTrackFathers[i]) tBinTrackFather=i;

  if(!fBuildUniqueParents)  //If true, I am likely looking for new parents, so I don't want these asserts to be tripped
  {
    if(tBinV0Father==-1) cout << "FAILURE IMMINENT: aV0FatherType = " << aV0FatherType << endl;
    if(tBinTrackFather==-1) cout << "FAILURE IMMINENT: aTrackFatherType = " << aTrackFatherType << endl;
    assert(tBinV0Father>-1);
    assert(tBinTrackFather>-1);
  }
  aMatrix->Fill(tBinV0Father,tBinTrackFather);
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildPairFractionHistogramsParticleV0(ParticlePDGType aParticleType, ParticlePDGType aV0Type, TH1* aHistogram, TH2* aMatrix)
{
  vector<ThermParticle> aParticleCollection;
  vector<ThermV0Particle> aV0Collection;

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    aV0Collection =  fEventsCollection[iEv].GetV0ParticleCollection(aV0Type);
    aParticleCollection = fEventsCollection[iEv].GetParticleCollection(aParticleType);

    ThermV0Particle tV0;

    for(unsigned int iV0=0; iV0<aV0Collection.size(); iV0++)
    {
      tV0 = aV0Collection[iV0];
      int tV0FatherType = tV0.GetFatherPID();

      double tBin = 0.;
      if(tV0FatherType == aV0Type) tBin = 0.;
      else if(tV0FatherType==kPDGSigma || tV0FatherType==kPDGASigma) tBin = 1.;
      else if(tV0FatherType==kPDGXi0 || tV0FatherType==kPDGAXi0) tBin = 2.;
      else if(tV0FatherType==kPDGXiC || tV0FatherType==kPDGAXiC) tBin = 3.;
      else if(tV0FatherType==kPDGOmega || tV0FatherType==kPDGAOmega) tBin = 4.;
      else tBin = -1.;
      tBin += 0.1;

      if(tV0.GoodV0())
      {
        ThermParticle tParticle;
        for(unsigned int iPar=0; iPar<aParticleCollection.size(); iPar++)
        {
          tParticle = aParticleCollection[iPar];
          int tParticleFatherType = tParticle.GetFatherPID();

          aHistogram->Fill(tBin);
          if(fBuildUniqueParents) BuildUniqueParents(TMath::Abs(tV0FatherType), TMath::Abs(tParticleFatherType));
          MapAndFillParentsMatrix(aMatrix, TMath::Abs(tV0FatherType), TMath::Abs(tParticleFatherType));
        }
      }
    }

  }

}


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildAllPairFractionHistograms()
{
  BuildPairFractionHistogramsParticleV0(kPDGKchP, kPDGLam, fPairFractionsLamKchP, fParentsMatrixLamKchP);
  BuildPairFractionHistogramsParticleV0(kPDGKchM, kPDGALam, fPairFractionsALamKchM, fParentsMatrixALamKchM);

  BuildPairFractionHistogramsParticleV0(kPDGKchM, kPDGLam, fPairFractionsLamKchM, fParentsMatrixLamKchM);
  BuildPairFractionHistogramsParticleV0(kPDGKchP, kPDGALam, fPairFractionsALamKchP, fParentsMatrixALamKchP);
}


//________________________________________________________________________________________________________________
void ThermEventsCollection::BuildUniqueParents(int aV0FatherType, int aTrackFatherType)
{
  bool bV0ParentAlreadyIncluded = false;
  bool bTrackParentAlreadyIncluded = false;
  for(unsigned int a=0; a<fUniqueV0Parents.size(); a++)
  {
    if(fUniqueV0Parents[a] == aV0FatherType) bV0ParentAlreadyIncluded = true;
  }
  for(unsigned int b=0; b<fUniqueTrackParents.size(); b++)
  {
    if(fUniqueTrackParents[b] == aTrackFatherType) bTrackParentAlreadyIncluded = true;
  }
  if(!bV0ParentAlreadyIncluded) fUniqueV0Parents.push_back(aV0FatherType);
  if(!bTrackParentAlreadyIncluded) fUniqueTrackParents.push_back(aTrackFatherType);

}

//________________________________________________________________________________________________________________
void ThermEventsCollection::PrintUniqueParents()
{
  std::sort(fUniqueV0Parents.begin(), fUniqueV0Parents.end());
  cout << "fUniqueV0Parents.size() = " << fUniqueV0Parents.size() << endl;
  for(unsigned int a=0; a<fUniqueV0Parents.size(); a++) cout << fUniqueV0Parents[a] << ", ";
  cout << endl;

  std::sort(fUniqueTrackParents.begin(), fUniqueTrackParents.end());
  cout << "fUniqueTrackParents.size() = " << fUniqueTrackParents.size() << endl;
  for(unsigned int b=0; b<fUniqueTrackParents.size(); b++) cout << fUniqueTrackParents[b] << ", ";
  cout << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void ThermEventsCollection::SaveAllPairFractionHistograms(TString aSaveFileLocation)
{
  TFile *tFile = new TFile(aSaveFileLocation, "RECREATE");
  assert(tFile->IsOpen());

  fPairFractionsLamKchP->Write();
  fParentsMatrixLamKchP->Write();

  fPairFractionsALamKchM->Write();
  fParentsMatrixALamKchM->Write();

  fPairFractionsLamKchM->Write();
  fParentsMatrixLamKchM->Write();

  fPairFractionsALamKchP->Write();
  fParentsMatrixALamKchP->Write();

  tFile->Close();
}

//________________________________________________________________________________________________________________
TCanvas* ThermEventsCollection::DrawAllPairFractionHistograms()
{
  TCanvas* tReturnCan = new TCanvas("tReturnCan","tReturnCan");
  tReturnCan->Divide(2,2);
  
  tReturnCan->cd(1);
  fPairFractionsLamKchP->Draw();

  tReturnCan->cd(2);
  fPairFractionsALamKchM->Draw();

  tReturnCan->cd(3);
  fPairFractionsLamKchM->Draw();

  tReturnCan->cd(4);
  fPairFractionsALamKchP->Draw();

  return tReturnCan;
}



