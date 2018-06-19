#include "SimpleThermAnalysis.h"
#include "TApplication.h"

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kMin);
  tFullTimer.Start();

//-----------------------------------------------------------------------------
  bool bRunFull = true;
  bool bUseMixedEventsForTransforms = true;
  bool bPrintUniqueParents = false;

  bool bBuildPairFractions = false;
  bool bBuildTransformMatrices = false;
  bool bBuildSingleParticleAnalyses = false;

  bool bBuildCorrelationFunctions = true;
  bool bBuild3dHists = false;
  bool bBuildMixedEventNumerators = false;
  int tNEventsToMix = 5;

  bool bUnitWeightCfNums = true;
  bool bWeightCfsWithParentInteraction = false;
  bool bOnlyWeightLongDecayParents = false;

  bool bCheckCoECoM = false;
  bool bRotateEventsByRandomAzimuthalAngles = false;
  bool bOnlyRunOverJaiEvents = false;

  bool bPerformFlowAnalysis = false;
  bool bBuildArtificialV3Signal = false;
  int tV3InclusionProb1 = 25;  //NOTE: A value of -1 turns entire V2 signal into V3

  double tMaxPrimaryDecayLength = -1; 
//  double tMaxPrimaryDecayLength = 4.01; 

  int tImpactParam = 5;
  //-----------------------------------------
  TString tEventsDirectory, tMatricesSaveFileName, tPairFractionSaveName, tSingleParticlesSaveName, tCorrelationFunctionsSaveName, tFlowSaveName;

  if(bRunFull)
  {
    tEventsDirectory = TString::Format("/home/jesse/Analysis/Therminator2/events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
    tMatricesSaveFileName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/TransformMatricesv2", tImpactParam);
    tPairFractionSaveName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/PairFractionsv2", tImpactParam);
    tSingleParticlesSaveName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/SingleParticleAnalysesv2", tImpactParam);
    tCorrelationFunctionsSaveName  = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/CorrelationFunctions", tImpactParam);
    tFlowSaveName  = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/FlowGraphs", tImpactParam);
  }
  else
  {
    tEventsDirectory = "/home/jesse/Analysis/Therminator2/events/TestEvents/";
    tMatricesSaveFileName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testTransformMatricesv2";
    tPairFractionSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testPairFractionsv2";
    tSingleParticlesSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testSingleParticleAnalysesv2";
    tCorrelationFunctionsSaveName  = "/home/jesse/Analysis/ReducedTherminator2Events/test/testCorrelationFunctions";
    tFlowSaveName  = "/home/jesse/Analysis/ReducedTherminator2Events/test/testFlowGraphs";
  }

  if(bOnlyRunOverJaiEvents) 
  {
    tEventsDirectory += TString("events0/");

    tMatricesSaveFileName += TString("_JaiEventsOnly");
    tPairFractionSaveName += TString("_JaiEventsOnly");
    tSingleParticlesSaveName += TString("_JaiEventsOnly");
    tCorrelationFunctionsSaveName += TString("_JaiEventsOnly");
    tFlowSaveName += TString("_JaiEventsOnly");
  }

  if(bRotateEventsByRandomAzimuthalAngles)
  {
    tMatricesSaveFileName += TString("_RandomEPs");
    tPairFractionSaveName += TString("_RandomEPs");
    tSingleParticlesSaveName += TString("_RandomEPs");
    tCorrelationFunctionsSaveName += TString("_RandomEPs");
    tFlowSaveName += TString("_RandomEPs");
  }

  if(bBuildArtificialV3Signal)
  {
    tMatricesSaveFileName += TString::Format("_ArtificialV3Signal%d", tV3InclusionProb1);
    tPairFractionSaveName += TString::Format("_ArtificialV3Signal%d", tV3InclusionProb1);
    tSingleParticlesSaveName += TString::Format("_ArtificialV3Signal%d", tV3InclusionProb1);
    tCorrelationFunctionsSaveName += TString::Format("_ArtificialV3Signal%d", tV3InclusionProb1);
    tFlowSaveName += TString::Format("_ArtificialV3Signal%d", tV3InclusionProb1);
cout << "tFlowSaveName = " << tFlowSaveName << endl << endl << endl;
  }

  if(bUnitWeightCfNums) tCorrelationFunctionsSaveName += TString("_NumWeight1");
  //-----------------------------------------

  SimpleThermAnalysis *tSimpleThermAnalysis = new SimpleThermAnalysis();
  tSimpleThermAnalysis->SetUseMixedEventsForTransforms(bUseMixedEventsForTransforms);
  tSimpleThermAnalysis->SetBuildUniqueParents(bPrintUniqueParents);

  tSimpleThermAnalysis->SetBuildPairFractions(bBuildPairFractions);
  tSimpleThermAnalysis->SetBuildTransformMatrices(bBuildTransformMatrices);

  tSimpleThermAnalysis->SetBuildCorrelationFunctions(bBuildCorrelationFunctions, bBuild3dHists);
  tSimpleThermAnalysis->SetBuildMixedEventNumerators(bBuildMixedEventNumerators);
  tSimpleThermAnalysis->SetUnitWeightCfNums(bUnitWeightCfNums);
  tSimpleThermAnalysis->SetWeightCfsWithParentInteraction(bWeightCfsWithParentInteraction);
  tSimpleThermAnalysis->SetOnlyWeightLongDecayParents(bOnlyWeightLongDecayParents);
  tSimpleThermAnalysis->SetNEventsToMix(tNEventsToMix);

  tSimpleThermAnalysis->SetBuildSingleParticleAnalyses(bBuildSingleParticleAnalyses);

  tSimpleThermAnalysis->SetEventsDirectory(tEventsDirectory);
  tSimpleThermAnalysis->SetPairFractionsSaveName(tPairFractionSaveName);
  tSimpleThermAnalysis->SetTransformMatricesSaveName(tMatricesSaveFileName);
  tSimpleThermAnalysis->SetSingleParticlesSaveName(tSingleParticlesSaveName);
  tSimpleThermAnalysis->SetCorrelationFunctionsSaveName(tCorrelationFunctionsSaveName);
  tSimpleThermAnalysis->SetFlowAnalysisSaveName(tFlowSaveName);
  if(tMaxPrimaryDecayLength > 0.) tSimpleThermAnalysis->SetMaxPrimaryDecayLength(tMaxPrimaryDecayLength);

  tSimpleThermAnalysis->SetCheckCoECoM(bCheckCoECoM);
  tSimpleThermAnalysis->SetRotateEventsByRandomAzimuthalAngles(bRotateEventsByRandomAzimuthalAngles);
  tSimpleThermAnalysis->SetPerformFlowAnalysis(bPerformFlowAnalysis);
  tSimpleThermAnalysis->SetBuildArtificialV3Signal(bBuildArtificialV3Signal, tV3InclusionProb1);

  tSimpleThermAnalysis->ProcessAll();

//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
