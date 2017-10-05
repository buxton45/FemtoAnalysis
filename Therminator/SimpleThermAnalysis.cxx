/* SimpleThermAnalysis.cxx */

#include "SimpleThermAnalysis.h"

#ifdef __ROOT__
ClassImp(SimpleThermAnalysis)
#endif

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
SimpleThermAnalysis::SimpleThermAnalysis() :
  fNFiles(0),
  fNEvents(0),

  fEventsDirectory("/home/jesse/Analysis/Therminator2/events/TestEvents/"),
  fPairFractionsSaveName("/home/jesse/Analysis/ReducedTherminator2Events/test/testPairFractionsv2.root"),
  fTransformMatricesSaveName("/home/jesse/Analysis/ReducedTherminator2Events/test/testTransformMatricesv2.root"),
  fCorrelationFunctionsSaveName("/home/jesse/Analysis/ReducedTherminator2Events/test/testCorrelationFunctions.root"),
  fSingleParticlesSaveName("/home/jesse/Analysis/ReducedTherminator2Events/test/testSingleParticleAnalysesv2.root"),

  fFileNameCollection(0),
  fEventsCollection(0),

  fMixEvents(false),
  fNEventsToMix(5),
  fMixingEventsCollection(0),

  fKStarMin(0.),
  fKStarMax(1.),
  fNBinsKStar(200),

  fMaxPrimaryDecayLength(-1.),

  fBuildUniqueParents(false),

  fBuildPairFractions(true),
  fBuildTransformMatrices(true),
  fBuildCorrelationFunctions(true),
  fBuildMixedEventNumerators(false),
  fWeightCfsWithParentInteraction(false),
  fBuildSingleParticleAnalyses(true),

  fAnalysisLamKchP(nullptr),
  fAnalysisALamKchM(nullptr),
  fAnalysisLamKchM(nullptr),
  fAnalysisALamKchP(nullptr),
  fAnalysisLamK0(nullptr),
  fAnalysisALamK0(nullptr),

  fSPAnalysisLam(nullptr),
  fSPAnalysisALam(nullptr),
  fSPAnalysisKchP(nullptr), 
  fSPAnalysisKchM(nullptr),
  fSPAnalysisProt(nullptr), 
  fSPAnalysisAProt(nullptr),
  fSPAnalysisK0(nullptr)

{
  fAnalysisLamKchP = new ThermPairAnalysis(kLamKchP);
  fAnalysisALamKchM = new ThermPairAnalysis(kALamKchM);
  fAnalysisLamKchM = new ThermPairAnalysis(kLamKchM);
  fAnalysisALamKchP = new ThermPairAnalysis(kALamKchP);
  fAnalysisLamK0 = new ThermPairAnalysis(kLamK0);
  fAnalysisALamK0 = new ThermPairAnalysis(kALamK0);


  fSPAnalysisLam = new ThermSingleParticleAnalysis(kPDGLam);
  fSPAnalysisALam = new ThermSingleParticleAnalysis(kPDGALam);
  fSPAnalysisKchP = new ThermSingleParticleAnalysis(kPDGKchP);
  fSPAnalysisKchM = new ThermSingleParticleAnalysis(kPDGKchM);
  fSPAnalysisProt = new ThermSingleParticleAnalysis(kPDGProt);
  fSPAnalysisAProt = new ThermSingleParticleAnalysis(kPDGAntiProt);
  fSPAnalysisK0 = new ThermSingleParticleAnalysis(kPDGK0);

}



//________________________________________________________________________________________________________________
SimpleThermAnalysis::~SimpleThermAnalysis()
{
/*no-op*/
}


//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetUseMixedEvents(bool aUse)
{
  fMixEvents = aUse;

  fAnalysisLamKchP->SetUseMixedEvents(aUse);
  fAnalysisALamKchM->SetUseMixedEvents(aUse);
  fAnalysisLamKchM->SetUseMixedEvents(aUse);
  fAnalysisALamKchP->SetUseMixedEvents(aUse);
  fAnalysisLamK0->SetUseMixedEvents(aUse);
  fAnalysisALamK0->SetUseMixedEvents(aUse);
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildUniqueParents(bool aBuild)
{
  fBuildUniqueParents = aBuild;

  fAnalysisLamKchP->SetBuildUniqueParents(aBuild);
  fAnalysisALamKchM->SetBuildUniqueParents(aBuild);
  fAnalysisLamKchM->SetBuildUniqueParents(aBuild);
  fAnalysisALamKchP->SetBuildUniqueParents(aBuild);
  fAnalysisLamK0->SetBuildUniqueParents(aBuild);
  fAnalysisALamK0->SetBuildUniqueParents(aBuild);

  fSPAnalysisLam->SetBuildUniqueParents(aBuild);
  fSPAnalysisALam->SetBuildUniqueParents(aBuild);
  fSPAnalysisKchP->SetBuildUniqueParents(aBuild);
  fSPAnalysisKchM->SetBuildUniqueParents(aBuild);
  fSPAnalysisProt->SetBuildUniqueParents(aBuild);
  fSPAnalysisAProt->SetBuildUniqueParents(aBuild);
  fSPAnalysisK0->SetBuildUniqueParents(aBuild);
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SaveAll()
{
  if(fBuildPairFractions)
  {
    fPairFractionsSaveName += TString(".root");
    TFile *tFilePairFractions = new TFile(fPairFractionsSaveName, "RECREATE");

    fAnalysisLamKchP->SavePairFractionsAndParentsMatrix(tFilePairFractions);
    fAnalysisALamKchM->SavePairFractionsAndParentsMatrix(tFilePairFractions);
    fAnalysisLamKchM->SavePairFractionsAndParentsMatrix(tFilePairFractions);
    fAnalysisALamKchP->SavePairFractionsAndParentsMatrix(tFilePairFractions);
    fAnalysisLamK0->SavePairFractionsAndParentsMatrix(tFilePairFractions);
    fAnalysisALamK0->SavePairFractionsAndParentsMatrix(tFilePairFractions);

    tFilePairFractions->Close();
  }
  //---------------------------------------------
  if(fBuildTransformMatrices)
  {
    fTransformMatricesSaveName += TString(".root");
    TFile* tFileTransformMatrices = new TFile(fTransformMatricesSaveName, "RECREATE");

    fAnalysisLamKchP->SaveAllTransformMatrices(tFileTransformMatrices);
    fAnalysisALamKchM->SaveAllTransformMatrices(tFileTransformMatrices);
    fAnalysisLamKchM->SaveAllTransformMatrices(tFileTransformMatrices);
    fAnalysisALamKchP->SaveAllTransformMatrices(tFileTransformMatrices);
    fAnalysisLamK0->SaveAllTransformMatrices(tFileTransformMatrices);
    fAnalysisALamK0->SaveAllTransformMatrices(tFileTransformMatrices);

    tFileTransformMatrices->Close();
  }
  //---------------------------------------------
  if(fBuildCorrelationFunctions)
  {
    if(fBuildMixedEventNumerators) fCorrelationFunctionsSaveName += TString::Format("_%iMixedEvNum", fNEventsToMix);
    if(fWeightCfsWithParentInteraction) fCorrelationFunctionsSaveName += TString("_WeightParentsInteraction");
    fCorrelationFunctionsSaveName += TString(".root");
    TFile* tFileCorrelationFunctions = new TFile(fCorrelationFunctionsSaveName, "RECREATE");

    fAnalysisLamKchP->SaveAllCorrelationFunctions(tFileCorrelationFunctions);
    fAnalysisALamKchM->SaveAllCorrelationFunctions(tFileCorrelationFunctions);
    fAnalysisLamKchM->SaveAllCorrelationFunctions(tFileCorrelationFunctions);
    fAnalysisALamKchP->SaveAllCorrelationFunctions(tFileCorrelationFunctions);
    fAnalysisLamK0->SaveAllCorrelationFunctions(tFileCorrelationFunctions);
    fAnalysisALamK0->SaveAllCorrelationFunctions(tFileCorrelationFunctions);

    tFileCorrelationFunctions->Close();
  }
  //---------------------------------------------
  if(fBuildSingleParticleAnalyses)
  {
    fSingleParticlesSaveName += TString(".root");
    TFile* tFileSingleParticleAnalyses = new TFile(fSingleParticlesSaveName, "RECREATE");

    fSPAnalysisLam->SaveAll(tFileSingleParticleAnalyses);
    fSPAnalysisALam->SaveAll(tFileSingleParticleAnalyses);
    fSPAnalysisKchP->SaveAll(tFileSingleParticleAnalyses);
    fSPAnalysisKchM->SaveAll(tFileSingleParticleAnalyses);
    fSPAnalysisProt->SaveAll(tFileSingleParticleAnalyses);
    fSPAnalysisAProt->SaveAll(tFileSingleParticleAnalyses);
    fSPAnalysisK0->SaveAll(tFileSingleParticleAnalyses);

    tFileSingleParticleAnalyses->Close();
  }
}

//________________________________________________________________________________________________________________
vector<ThermEvent> SimpleThermAnalysis::ExtractEventsFromRootFile(TString aFileLocation)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TTree *tTree = (TTree*)tFile->Get("particles");

  ParticleCoor *tParticleEntry = new ParticleCoor();
  TBranch *tParticleBranch = tTree->GetBranch("particle");
  tParticleBranch->SetAddress(tParticleEntry);

  int tNEvents = 1;
  unsigned int tEventID;
  ThermEvent tThermEvent;

  vector<ThermEvent> tEventsCollection(0);

  for(int i=0; i<tParticleBranch->GetEntries(); i++)
  {
    tParticleBranch->GetEntry(i);

    if(i==0) tEventID = tParticleEntry->eventid;

    if(tParticleEntry->eventid != tEventID)
    {
      tThermEvent.MatchDaughtersWithFathers();
      tThermEvent.FindAllFathers();
      tThermEvent.SetEventID(tEventID);
      tEventsCollection.push_back(tThermEvent);
      tThermEvent.ClearThermEvent();

      tNEvents++;
      tEventID = tParticleEntry->eventid;
    }

    tThermEvent.PushBackThermParticle(tParticleEntry);
    if(tThermEvent.IsDaughterOfInterest(tParticleEntry)) tThermEvent.PushBackThermDaughterOfInterest(tParticleEntry);
    if(tThermEvent.IsParticleOfInterest(tParticleEntry)) tThermEvent.PushBackThermParticleOfInterest(tParticleEntry);
  }
  tThermEvent.MatchDaughtersWithFathers();
  tThermEvent.FindAllFathers();
  tThermEvent.SetEventID(tEventID);
  tEventsCollection.push_back(tThermEvent);

  fNEvents += tNEvents;

cout << "aFileLocation = " << aFileLocation << endl;
cout << "tEventsCollection.size() = " << tEventsCollection.size() << endl;
cout << "fNEvents = " << fNEvents << endl;


  tFile->Close();
  delete tFile;

  delete tParticleEntry;

  return tEventsCollection;
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::ProcessAll()
{
  TString tCompleteFilePath;

  TSystemDirectory tDir(fEventsDirectory.Data(), fEventsDirectory.Data());
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
        tCompleteFilePath = TString(fEventsDirectory.Data()) + tName;
        fEventsCollection = ExtractEventsFromRootFile(tCompleteFilePath);
        ProcessEventByEvent(fEventsCollection);

        cout << "fNFiles = " << fNFiles << endl << endl;
      }
    }
  }
  cout << "Total number of files = " << fNFiles << endl;

  if(fBuildUniqueParents)
  {
    fAnalysisLamKchP->PrintUniqueParents();
    fAnalysisALamKchM->PrintUniqueParents();
    fAnalysisLamKchM->PrintUniqueParents();
    fAnalysisALamKchP->PrintUniqueParents();
    fAnalysisLamK0->PrintUniqueParents();
    fAnalysisALamK0->PrintUniqueParents();
  }


  fAnalysisLamKchP->PrintPrimaryAndOtherPairInfo();
  fAnalysisALamKchM->PrintPrimaryAndOtherPairInfo();
  fAnalysisLamKchM->PrintPrimaryAndOtherPairInfo();
  fAnalysisALamKchP->PrintPrimaryAndOtherPairInfo();
  fAnalysisLamK0->PrintPrimaryAndOtherPairInfo();
  fAnalysisALamK0->PrintPrimaryAndOtherPairInfo();



  SaveAll();
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::ProcessEventByEvent(vector<ThermEvent> &aEventsCollection)
{
  if(!fMixEvents) fMixingEventsCollection.clear();

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    fAnalysisLamKchP->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection, fMaxPrimaryDecayLength);
    fAnalysisALamKchM->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection, fMaxPrimaryDecayLength);

    fAnalysisLamKchM->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection, fMaxPrimaryDecayLength);
    fAnalysisALamKchP->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection, fMaxPrimaryDecayLength);

    fAnalysisLamK0->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection, fMaxPrimaryDecayLength);
    fAnalysisALamK0->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection, fMaxPrimaryDecayLength);

    //--------------------------------------------
    if(fBuildSingleParticleAnalyses)
    {
      fSPAnalysisLam->ProcessEvent(fEventsCollection[iEv]);
      fSPAnalysisALam->ProcessEvent(fEventsCollection[iEv]);
      fSPAnalysisKchP->ProcessEvent(fEventsCollection[iEv]);
      fSPAnalysisKchM->ProcessEvent(fEventsCollection[iEv]);
      fSPAnalysisProt->ProcessEvent(fEventsCollection[iEv]); 
      fSPAnalysisAProt->ProcessEvent(fEventsCollection[iEv]);
      fSPAnalysisK0->ProcessEvent(fEventsCollection[iEv]);
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
void SimpleThermAnalysis::SetBuildPairFractions(bool aBuild) 
{
  fBuildPairFractions = aBuild;

  fAnalysisLamKchP->SetBuildPairFractions(aBuild);
  fAnalysisALamKchM->SetBuildPairFractions(aBuild);

  fAnalysisLamKchM->SetBuildPairFractions(aBuild);
  fAnalysisALamKchP->SetBuildPairFractions(aBuild);

  fAnalysisLamK0->SetBuildPairFractions(aBuild);
  fAnalysisALamK0->SetBuildPairFractions(aBuild);
}


//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildTransformMatrices(bool aBuild)
{
  fBuildTransformMatrices = aBuild;

  fAnalysisLamKchP->SetBuildTransformMatrices(aBuild);
  fAnalysisALamKchM->SetBuildTransformMatrices(aBuild);

  fAnalysisLamKchM->SetBuildTransformMatrices(aBuild);
  fAnalysisALamKchP->SetBuildTransformMatrices(aBuild);

  fAnalysisLamK0->SetBuildTransformMatrices(aBuild);
  fAnalysisALamK0->SetBuildTransformMatrices(aBuild);
}


//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildCorrelationFunctions(bool aBuild)
{
  fBuildCorrelationFunctions = aBuild;

  fAnalysisLamKchP->SetBuildCorrelationFunctions(aBuild);
  fAnalysisALamKchM->SetBuildCorrelationFunctions(aBuild);

  fAnalysisLamKchM->SetBuildCorrelationFunctions(aBuild);
  fAnalysisALamKchP->SetBuildCorrelationFunctions(aBuild);

  fAnalysisLamK0->SetBuildCorrelationFunctions(aBuild);
  fAnalysisALamK0->SetBuildCorrelationFunctions(aBuild);
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildMixedEventNumerators(bool aBuild)
{
  fBuildMixedEventNumerators = aBuild;

  fAnalysisLamKchP->SetBuildMixedEventNumerators(aBuild);
  fAnalysisALamKchM->SetBuildMixedEventNumerators(aBuild);

  fAnalysisLamKchM->SetBuildMixedEventNumerators(aBuild);
  fAnalysisALamKchP->SetBuildMixedEventNumerators(aBuild);

  fAnalysisLamK0->SetBuildMixedEventNumerators(aBuild);
  fAnalysisALamK0->SetBuildMixedEventNumerators(aBuild);
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetWeightCfsWithParentInteraction(bool aSet)
{
  fWeightCfsWithParentInteraction = aSet;

  fAnalysisLamKchP->SetWeightCfsWithParentInteraction(aSet);
  fAnalysisALamKchM->SetWeightCfsWithParentInteraction(aSet);

  fAnalysisLamKchM->SetWeightCfsWithParentInteraction(aSet);
  fAnalysisALamKchP->SetWeightCfsWithParentInteraction(aSet);

  fAnalysisLamK0->SetWeightCfsWithParentInteraction(aSet);
  fAnalysisALamK0->SetWeightCfsWithParentInteraction(aSet);
}


//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildSingleParticleAnalyses(bool aBuild)
{
  fBuildSingleParticleAnalyses = aBuild;
}







