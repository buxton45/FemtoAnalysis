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
  FitGeneratorType tFitGenType = kPairwConj;
  bool bBuildOtherPairs=false;
  bool bBuildSingleParticleAnalyses = false;
  bool bPerformFlowAnalysis = false;

  bool bRunFull = true;
  bool bUseMixedEventsForTransforms = true;
  bool bPrintUniqueParents = false;

  bool bBuildPairFractions = false;
  bool bBuildTransformMatrices = false;

  bool bBuildCorrelationFunctions = true;
  bool bBuild3dHists = false;
  bool bBuildPairSourcewmTInfo = false;
  bool bBuildCfYlm = true;

  bool bBuildMixedEventNumerators = false;
  int tNEventsToMix = 5;

  bool bUnitWeightCfNums = false;
  bool bWeightCfsWithParentInteraction = false;
  bool bOnlyWeightLongDecayParents = false;

  bool bDrawRStarFromGaussian = true;
  td1dVec tGaussSourceInfoAllLamK{5.0, 5.0, 5.0,
                                  3.0, 0.0, 0.0};

  bool bCheckCoECoM = false;
  bool bRotateEventsByRandomAzimuthalAngles = true;
  bool bOnlyRunOverJaiEvents = false;

  bool bBuildArtificialV3Signal = false;
  int tV3InclusionProb1 = 25;  //NOTE: A value of -1 turns entire V2 signal into V3

  bool bBuildArtificialV2Signal = false;
  int tV2InclusionProb1 = -1;  //NOTE: A value of -1 means 100%

  bool bKillFlowSignals=false;

  double tMaxPrimaryDecayLength = -1; 
//  double tMaxPrimaryDecayLength = 4.01; 

  int tImpactParam = 2;
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
  }
  if(bBuildArtificialV2Signal)
  {
    tMatricesSaveFileName += TString::Format("_ArtificialV2Signal%d", tV2InclusionProb1);
    tPairFractionSaveName += TString::Format("_ArtificialV2Signal%d", tV2InclusionProb1);
    tSingleParticlesSaveName += TString::Format("_ArtificialV2Signal%d", tV2InclusionProb1);
    tCorrelationFunctionsSaveName += TString::Format("_ArtificialV2Signal%d", tV2InclusionProb1);
    tFlowSaveName += TString::Format("_ArtificialV2Signal%d", tV2InclusionProb1);
  }
  if(bKillFlowSignals)
  {
    tMatricesSaveFileName += TString("_KillFlowSignals");
    tPairFractionSaveName += TString("_KillFlowSignals");
    tSingleParticlesSaveName += TString("_KillFlowSignals");
    tCorrelationFunctionsSaveName += TString("_KillFlowSignals");
    tFlowSaveName += TString("_KillFlowSignals");
  }

  if(bUnitWeightCfNums) tCorrelationFunctionsSaveName += TString("_NumWeight1");
  if(bBuildOtherPairs) tCorrelationFunctionsSaveName += TString("_wOtherPairs");
  //-----------------------------------------

  SimpleThermAnalysis *tSimpleThermAnalysis = new SimpleThermAnalysis(tFitGenType, bBuildOtherPairs, bBuildSingleParticleAnalyses, bPerformFlowAnalysis);
  tSimpleThermAnalysis->SetUseMixedEventsForTransforms(bUseMixedEventsForTransforms);
  tSimpleThermAnalysis->SetBuildUniqueParents(bPrintUniqueParents);

  tSimpleThermAnalysis->SetBuildPairFractions(bBuildPairFractions);
  tSimpleThermAnalysis->SetBuildTransformMatrices(bBuildTransformMatrices);

  tSimpleThermAnalysis->SetBuildCorrelationFunctions(bBuildCorrelationFunctions, bBuild3dHists, bBuildPairSourcewmTInfo);
  tSimpleThermAnalysis->SetBuildCfYlm(bBuildCfYlm);
  tSimpleThermAnalysis->SetBuildMixedEventNumerators(bBuildMixedEventNumerators);
  tSimpleThermAnalysis->SetUnitWeightCfNums(bUnitWeightCfNums);
  tSimpleThermAnalysis->SetWeightCfsWithParentInteraction(bWeightCfsWithParentInteraction);
  tSimpleThermAnalysis->SetOnlyWeightLongDecayParents(bOnlyWeightLongDecayParents);
  tSimpleThermAnalysis->SetDrawRStarFromGaussian(bDrawRStarFromGaussian);
  tSimpleThermAnalysis->SetGaussSourceInfoAllLamK(tGaussSourceInfoAllLamK[0], tGaussSourceInfoAllLamK[1], tGaussSourceInfoAllLamK[2], 
                                                  tGaussSourceInfoAllLamK[3], tGaussSourceInfoAllLamK[4], tGaussSourceInfoAllLamK[5]);
  tSimpleThermAnalysis->SetNEventsToMix(tNEventsToMix);

  tSimpleThermAnalysis->SetEventsDirectory(tEventsDirectory);
  tSimpleThermAnalysis->SetPairFractionsSaveName(tPairFractionSaveName);
  tSimpleThermAnalysis->SetTransformMatricesSaveName(tMatricesSaveFileName);
  tSimpleThermAnalysis->SetSingleParticlesSaveName(tSingleParticlesSaveName);
  tSimpleThermAnalysis->SetCorrelationFunctionsSaveName(tCorrelationFunctionsSaveName);
  tSimpleThermAnalysis->SetFlowAnalysisSaveName(tFlowSaveName);
  if(tMaxPrimaryDecayLength > 0.) tSimpleThermAnalysis->SetMaxPrimaryDecayLength(tMaxPrimaryDecayLength);

  tSimpleThermAnalysis->SetCheckCoECoM(bCheckCoECoM);
  tSimpleThermAnalysis->SetRotateEventsByRandomAzimuthalAngles(bRotateEventsByRandomAzimuthalAngles);
  tSimpleThermAnalysis->SetBuildArtificialV3Signal(bBuildArtificialV3Signal, tV3InclusionProb1);
  tSimpleThermAnalysis->SetBuildArtificialV2Signal(bBuildArtificialV2Signal, tV2InclusionProb1);
  tSimpleThermAnalysis->SetKillFlowSignals(bKillFlowSignals);

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
