/* SimpleThermAnalysis.cxx */

#include "SimpleThermAnalysis.h"

#ifdef __ROOT__
ClassImp(SimpleThermAnalysis)
#endif

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
SimpleThermAnalysis::SimpleThermAnalysis(FitGeneratorType aFitGenType, bool aBuildOtherPairs, bool aBuildSingleParticleAnalyses, bool aPerformFlowAnalysis) :
  fFitGenType(aFitGenType),
  fBuildOtherPairs(aBuildOtherPairs),
  fBuildSingleParticleAnalyses(aBuildSingleParticleAnalyses),
  fPerformFlowAnalysis(aPerformFlowAnalysis),

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
  fBuildPairSourcewmTInfo(false),
  fBuildCfYlm(false),
  fBuildAliFemtoCfYlm(false),
  fBuildMixedEventNumerators(false),
  fUnitWeightCfNums(false),
  fWeightCfsWithParentInteraction(false),
  fOnlyWeightLongDecayParents(false),
  fDrawRStarFromGaussian(false),

  fPairAnalysisVec(0),
  fOtherPairAnalysisVec(0),
  fSPAnalysisVec(0),

  fFlowCollection(nullptr),

  fCheckCoECoM(false),
  fRotateEventsByRandAzAngles(false),
  fArtificialV3Info(0,-1)

{
  if(fFitGenType==kPair || fFitGenType==kPairwConj)
  {
    fPairAnalysisVec.push_back(new ThermPairAnalysis(kLamKchP));
    fPairAnalysisVec.push_back(new ThermPairAnalysis(kLamKchM));
    fPairAnalysisVec.push_back(new ThermPairAnalysis(kLamK0));
  }
  if(fFitGenType==kConjPair || fFitGenType==kPairwConj)
  {
    fPairAnalysisVec.push_back(new ThermPairAnalysis(kALamKchM));
    fPairAnalysisVec.push_back(new ThermPairAnalysis(kALamKchP));
    fPairAnalysisVec.push_back(new ThermPairAnalysis(kALamK0));
  }

  if(fBuildOtherPairs)
  {
    fOtherPairAnalysisVec.push_back(new ThermPairAnalysis(kKchPKchP));
    fOtherPairAnalysisVec.push_back(new ThermPairAnalysis(kK0K0));
    fOtherPairAnalysisVec.push_back(new ThermPairAnalysis(kLamLam));
  }

  if(fBuildSingleParticleAnalyses)
  {
    fSPAnalysisVec.push_back(new ThermSingleParticleAnalysis(kPDGLam));
    fSPAnalysisVec.push_back(new ThermSingleParticleAnalysis(kPDGALam));
    fSPAnalysisVec.push_back(new ThermSingleParticleAnalysis(kPDGKchP));
    fSPAnalysisVec.push_back(new ThermSingleParticleAnalysis(kPDGKchM));
    fSPAnalysisVec.push_back(new ThermSingleParticleAnalysis(kPDGProt));
    fSPAnalysisVec.push_back(new ThermSingleParticleAnalysis(kPDGAntiProt));
    fSPAnalysisVec.push_back(new ThermSingleParticleAnalysis(kPDGK0));
  }

  if(fPerformFlowAnalysis) fFlowCollection = new ThermFlowCollection();
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

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetUseMixedEventsForTransforms(aUse);
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildUniqueParents(bool aBuild)
{
  fBuildUniqueParents = aBuild;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetBuildUniqueParents(aBuild);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetBuildUniqueParents(false);
  }
  if(fBuildSingleParticleAnalyses)
  {
    for(unsigned int iSP=0; iSP<fSPAnalysisVec.size(); iSP++) fSPAnalysisVec[iSP]->SetBuildUniqueParents(aBuild);
  }
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SaveAll()
{
  if(fBuildPairFractions)
  {
    if(fFitGenType==kPair)     fPairFractionsSaveName += TString("_PairOnly");
    if(fFitGenType==kConjPair) fPairFractionsSaveName += TString("_ConjPairOnly");

    fPairFractionsSaveName += TString(".root");
    TFile *tFilePairFractions = new TFile(fPairFractionsSaveName, "RECREATE");

    for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SavePairFractionsAndParentsMatrix(tFilePairFractions);
    tFilePairFractions->Close();
  }
  //---------------------------------------------
  if(fBuildTransformMatrices)
  {
    if(fFitGenType==kPair)     fTransformMatricesSaveName += TString("_PairOnly");
    if(fFitGenType==kConjPair) fTransformMatricesSaveName += TString("_ConjPairOnly");

    fTransformMatricesSaveName += TString(".root");
    TFile* tFileTransformMatrices = new TFile(fTransformMatricesSaveName, "RECREATE");

    for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SaveAllTransformMatrices(tFileTransformMatrices);
    tFileTransformMatrices->Close();
  }
  //---------------------------------------------
  if(fBuildCorrelationFunctions)
  {
    if(fBuildMixedEventNumerators) fCorrelationFunctionsSaveName += TString::Format("_%iMixedEvNum", fNEventsToMix);
    if(fWeightCfsWithParentInteraction) fCorrelationFunctionsSaveName += TString("_WeightParentsInteraction");
    if(fOnlyWeightLongDecayParents) fCorrelationFunctionsSaveName += TString("_OnlyWeightLongDecayParents");
    if(fDrawRStarFromGaussian) fCorrelationFunctionsSaveName += TString("_DrawRStarFromGaussian");
    if(fBuildPairSourcewmTInfo) fCorrelationFunctionsSaveName += TString("_BuildPairSourcewmTInfo");
    if(fBuildCfYlm) fCorrelationFunctionsSaveName += TString("_BuildCfYlm");
    if(fBuildAliFemtoCfYlm) fCorrelationFunctionsSaveName += TString("_BuildAliFemtoCfYlm");

    if(fFitGenType==kPair)     fCorrelationFunctionsSaveName += TString("_PairOnly");
    if(fFitGenType==kConjPair) fCorrelationFunctionsSaveName += TString("_ConjPairOnly");

    fCorrelationFunctionsSaveName += TString(".root");
    TFile* tFileCorrelationFunctions = new TFile(fCorrelationFunctionsSaveName, "RECREATE");

    for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SaveAllCorrelationFunctions(tFileCorrelationFunctions);

    if(fBuildOtherPairs)
    {
      for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SaveAllCorrelationFunctions(tFileCorrelationFunctions);
    }

    tFileCorrelationFunctions->Close();
  }
  //---------------------------------------------
  if(fBuildSingleParticleAnalyses)
  {
    fSingleParticlesSaveName += TString(".root");
    TFile* tFileSingleParticleAnalyses = new TFile(fSingleParticlesSaveName, "RECREATE");

    for(unsigned int iSP=0; iSP<fSPAnalysisVec.size(); iSP++) fSPAnalysisVec[iSP]->SaveAll(tFileSingleParticleAnalyses);
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
      if(fArtificialV3Info.first) tThermEvent.BuildArtificialV3Signal(fArtificialV3Info.second, fRotateEventsByRandAzAngles);
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
  if(fArtificialV3Info.first) tThermEvent.BuildArtificialV3Signal(fArtificialV3Info.second, fRotateEventsByRandAzAngles);
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
    for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->PrintUniqueParents();
  }
  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->PrintPrimaryAndOtherPairInfo();

  SaveAll();
  if(fPerformFlowAnalysis) fFlowCollection->Finalize();
  tDir->Delete();
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::ProcessEventByEvent(vector<ThermEvent> &aEventsCollection)
{
  if(!fMixEvents) fMixingEventsCollection.clear();

  for(unsigned int iEv=0; iEv < fEventsCollection.size(); iEv++)
  {
    for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection);
    if(fBuildOtherPairs)
    {
      for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->ProcessEvent(fEventsCollection[iEv], fMixingEventsCollection);
    }

    //--------------------------------------------
    if(fBuildSingleParticleAnalyses)
    {
      for(unsigned int iSP=0; iSP<fSPAnalysisVec.size(); iSP++) fSPAnalysisVec[iSP]->ProcessEvent(fEventsCollection[iEv]);
    }

    if(fPerformFlowAnalysis) fFlowCollection->BuildVnEPIngredients(fEventsCollection[iEv]);

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

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetBuildPairFractions(aBuild);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetBuildPairFractions(false);
  }
}


//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildTransformMatrices(bool aBuild)
{
  fBuildTransformMatrices = aBuild;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetBuildTransformMatrices(aBuild);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetBuildTransformMatrices(false);
  }
}


//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildCorrelationFunctions(bool aBuild, bool aBuild3dHists, bool aBuildPairSourcewmTInfo)
{
  fBuildCorrelationFunctions = aBuild;
  fBuild3dHists = aBuild3dHists;
  fBuildPairSourcewmTInfo = aBuildPairSourcewmTInfo;

  if(fBuildCorrelationFunctions) fMixEvents = true;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetBuildCorrelationFunctions(fBuildCorrelationFunctions, fBuild3dHists, fBuildPairSourcewmTInfo);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetBuildCorrelationFunctions(fBuildCorrelationFunctions, fBuild3dHists, fBuildPairSourcewmTInfo);
  }
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildCfYlm(bool aSet)
{
  fBuildCfYlm = aSet;
  if(fBuildCfYlm) assert(fBuildCorrelationFunctions);

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetBuildCfYlm(aSet);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetBuildCfYlm(aSet);
  }
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildAliFemtoCfYlm(bool aSet)
{
  fBuildAliFemtoCfYlm = aSet;
  if(fBuildAliFemtoCfYlm) assert(fBuildCorrelationFunctions);

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetBuildAliFemtoCfYlm(aSet);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetBuildAliFemtoCfYlm(aSet);
  }
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetBuildMixedEventNumerators(bool aBuild)
{
  fBuildMixedEventNumerators = aBuild;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetBuildMixedEventNumerators(aBuild);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetBuildMixedEventNumerators(aBuild);
  }
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetUnitWeightCfNums(bool aSet)
{
  fUnitWeightCfNums = aSet;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetUnitWeightCfNums(aSet);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetUnitWeightCfNums(aSet);
  }
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetWeightCfsWithParentInteraction(bool aSet)
{
  fWeightCfsWithParentInteraction = aSet;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetWeightCfsWithParentInteraction(aSet);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetWeightCfsWithParentInteraction(aSet);
  }
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetOnlyWeightLongDecayParents(bool aSet)
{
  fOnlyWeightLongDecayParents = aSet;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetOnlyWeightLongDecayParents(aSet);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetOnlyWeightLongDecayParents(aSet);
  }
}

//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetDrawRStarFromGaussian(bool aSet)
{
  fDrawRStarFromGaussian = aSet;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetDrawRStarFromGaussian(aSet);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetDrawRStarFromGaussian(aSet);
  }
}


//________________________________________________________________________________________________________________
void SimpleThermAnalysis::SetMaxPrimaryDecayLength(double aMax)
{
  fMaxPrimaryDecayLength = aMax;

  for(unsigned int iP=0; iP<fPairAnalysisVec.size(); iP++) fPairAnalysisVec[iP]->SetMaxPrimaryDecayLength(fMaxPrimaryDecayLength);
  if(fBuildOtherPairs)
  {
    for(unsigned int iOP=0; iOP<fOtherPairAnalysisVec.size(); iOP++) fOtherPairAnalysisVec[iOP]->SetMaxPrimaryDecayLength(fMaxPrimaryDecayLength);
  }
}





