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
  fMixEventsForTransforms(false),
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
  fBuild3dHists(false),
  fBuildMixedEventNumerators(false),
  fUnitWeightCfNums(false),
  fWeightCfsWithParentInteraction(false),
  fOnlyWeightLongDecayParents(false),
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
  fSPAnalysisK0(nullptr),

  fFlowAnalysis(nullptr),

  fCheckCoECoM(false),
  fRotateEventsByRandAzAngles(false),
  fPerformFlowAnalysis(false),
  fBuildArtificialV3Signal(false)

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

  fFlowAnalysis = new ThermFlowAnalysis();
}



//________________________________________________________________________________________________________________
SimpleThermAnalysis::~SimpleThermAnalysis()
{
/*no-op*/
}


//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetUseMixedEventsForTransforms(bool aUse)
{
  fMixEventsForTransforms = aUse;
  if(fMixEventsForTransforms) fMixEvents = true;

  fAnalysisLamKchP->SetUseMixedEventsForTransforms(aUse);
  fAnalysisALamKchM->SetUseMixedEventsForTransforms(aUse);
  fAnalysisLamKchM->SetUseMixedEventsForTransforms(aUse);
  fAnalysisALamKchP->SetUseMixedEventsForTransforms(aUse);
  fAnalysisLamK0->SetUseMixedEventsForTransforms(aUse);
  fAnalysisALamK0->SetUseMixedEventsForTransforms(aUse);
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
    if(fOnlyWeightLongDecayParents) fCorrelationFunctionsSaveName += TString("_OnlyWeightLongDecayParents");
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
  vector<ThermEvent> tEventsCollection(0);

  TFile *tFile = TFile::Open(aFileLocation);
  if(!tFile || tFile->IsZombie() || !tFile->IsOpen()) return tEventsCollection;

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
      if(fBuildArtificialV3Signal) tThermEvent.BuildArtificialV3Signal();
      if(fPerformFlowAnalysis) fFlowAnalysis->BuildVnEPIngredients(tThermEvent); //Want to do this BEFORE EnforceKinematicCuts is called because need large eta 
                                                                                 //for calculation (shouldn't matter if before or after event rotation)
      if(fRotateEventsByRandAzAngles) tThermEvent.RotateAllParticlesByRandomAzimuthalAngle(false);
      tThermEvent.MatchDaughtersWithFathers();
      if(fCheckCoECoM) tThermEvent.CheckCoECoM();
      tThermEvent.FindAllFathers();
      tThermEvent.EnforceKinematicCuts();
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
  if(fBuildArtificialV3Signal) tThermEvent.BuildArtificialV3Signal();
  if(fPerformFlowAnalysis) fFlowAnalysis->BuildVnEPIngredients(tThermEvent);
  if(fRotateEventsByRandAzAngles) tThermEvent.RotateAllParticlesByRandomAzimuthalAngle(false);
  tThermEvent.MatchDaughtersWithFathers();
  if(fCheckCoECoM) tThermEvent.CheckCoECoM();
  tThermEvent.FindAllFathers();
  tThermEvent.EnforceKinematicCuts();
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
void SimpleThermAnalysis::ProcessAllInDirectory(TSystemDirectory* aEventsDirectory)
{
  TString tCompleteDirectoryPath = aEventsDirectory->GetTitle();
  if(!tCompleteDirectoryPath.EndsWith("/")) tCompleteDirectoryPath += TString("/");

  cout << "Analyzing events in directory: " << tCompleteDirectoryPath << endl;

  TString tCompleteFilePath;

  TList* tFiles = aEventsDirectory->GetListOfFiles();
  int tNFilesInDir = 0;

  const char* tBeginningText = "event";
  const char* tEndingText = ".root";

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
        tNFilesInDir++;
        tCompleteFilePath = TString(tCompleteDirectoryPath.Data()) + tName;
        fEventsCollection = ExtractEventsFromRootFile(tCompleteFilePath);
        ProcessEventByEvent(fEventsCollection);

        cout << "tNFilesInDir = " << tNFilesInDir << endl;
        cout << "fNFiles = " << fNFiles << endl << endl;
      }
    }
  }
  cout << "Total number of files in subdirectory = " << tNFilesInDir << endl;
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::ProcessAll()
{
  TSystemDirectory *tDir = new TSystemDirectory(fEventsDirectory.Data(), fEventsDirectory.Data());
  TList* tSubDirectories = tDir->GetListOfFiles(); 

  const char* tBeginningText = "events";

  fNFiles = 0;
  int tNSubDirectories = 0;

  if(tSubDirectories)
  {
    TSystemDirectory* tSubDirectory;
    TString tName;
    TIter tIterNext(tSubDirectories);

    while((tSubDirectory=(TSystemDirectory*)tIterNext()))
    {
      tName = tSubDirectory->GetName();
      if(tSubDirectory->IsDirectory() && tName.BeginsWith(tBeginningText))
      {
        tNSubDirectories++;
        cout << "tNSubDirectories = " << tNSubDirectories << endl << endl;
        ProcessAllInDirectory(tSubDirectory);
      }
    }
  }

  if(tNSubDirectories==0) ProcessAllInDirectory(tDir);  //i.e. no events subdirectories; all events in 
                                                        //single directory located at fEventsDirectory

  cout << "Total number of files = " << fNFiles << endl;
  //----------------------------------------------------

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

  fFlowAnalysis->Finalize();

  SaveAll();
  tDir->Delete();
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::ProcessEventByEvent(vector<ThermEvent> &aEventsCollection)
{
  if(!fMixEvents) fMixingEventsCollection.clear();

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    fAnalysisLamKchP->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection);
    fAnalysisALamKchM->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection);

    fAnalysisLamKchM->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection);
    fAnalysisALamKchP->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection);

    fAnalysisLamK0->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection);
    fAnalysisALamK0->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection);

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
void SimpleThermAnalysis::SetBuildCorrelationFunctions(bool aBuild, bool aBuild3dHists)
{
  fBuildCorrelationFunctions = aBuild;
  fBuild3dHists = aBuild3dHists;

  if(fBuildCorrelationFunctions) fMixEvents = true;

  fAnalysisLamKchP->SetBuildCorrelationFunctions(fBuildCorrelationFunctions, fBuild3dHists);
  fAnalysisALamKchM->SetBuildCorrelationFunctions(fBuildCorrelationFunctions, fBuild3dHists);

  fAnalysisLamKchM->SetBuildCorrelationFunctions(fBuildCorrelationFunctions, fBuild3dHists);
  fAnalysisALamKchP->SetBuildCorrelationFunctions(fBuildCorrelationFunctions, fBuild3dHists);

  fAnalysisLamK0->SetBuildCorrelationFunctions(fBuildCorrelationFunctions, fBuild3dHists);
  fAnalysisALamK0->SetBuildCorrelationFunctions(fBuildCorrelationFunctions, fBuild3dHists);
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
void SimpleThermAnalysis::SetUnitWeightCfNums(bool aSet)
{
  fUnitWeightCfNums = aSet;

  fAnalysisLamKchP->SetUnitWeightCfNums(aSet);
  fAnalysisALamKchM->SetUnitWeightCfNums(aSet);

  fAnalysisLamKchM->SetUnitWeightCfNums(aSet);
  fAnalysisALamKchP->SetUnitWeightCfNums(aSet);

  fAnalysisLamK0->SetUnitWeightCfNums(aSet);
  fAnalysisALamK0->SetUnitWeightCfNums(aSet);
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
void SimpleThermAnalysis::SetOnlyWeightLongDecayParents(bool aSet)
{
  fOnlyWeightLongDecayParents = aSet;

  fAnalysisLamKchP->SetOnlyWeightLongDecayParents(aSet);
  fAnalysisALamKchM->SetOnlyWeightLongDecayParents(aSet);

  fAnalysisLamKchM->SetOnlyWeightLongDecayParents(aSet);
  fAnalysisALamKchP->SetOnlyWeightLongDecayParents(aSet);

  fAnalysisLamK0->SetOnlyWeightLongDecayParents(aSet);
  fAnalysisALamK0->SetOnlyWeightLongDecayParents(aSet);
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildSingleParticleAnalyses(bool aBuild)
{
  fBuildSingleParticleAnalyses = aBuild;
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetMaxPrimaryDecayLength(double aMax)
{
  fMaxPrimaryDecayLength = aMax;

  fAnalysisLamKchP->SetMaxPrimaryDecayLength(fMaxPrimaryDecayLength);
  fAnalysisALamKchM->SetMaxPrimaryDecayLength(fMaxPrimaryDecayLength);

  fAnalysisLamKchM->SetMaxPrimaryDecayLength(fMaxPrimaryDecayLength);
  fAnalysisALamKchP->SetMaxPrimaryDecayLength(fMaxPrimaryDecayLength);

  fAnalysisLamK0->SetMaxPrimaryDecayLength(fMaxPrimaryDecayLength);
  fAnalysisALamK0->SetMaxPrimaryDecayLength(fMaxPrimaryDecayLength);
}





