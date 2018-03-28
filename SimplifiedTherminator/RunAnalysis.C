#include "SimpleThermAnalysis.h"
//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  bool bRunFull = true;
  bool bUseMixedEvents = true;
  bool bPrintUniqueParents = false;

  bool bBuildPairFractions = false;
  bool bBuildTransformMatrices = false;
  bool bBuildSingleParticleAnalyses = false;

  bool bBuildCorrelationFunctions = true;
  bool bBuildMixedEventNumerators = false;
  int tNEventsToMix = 5;

  bool bUnitWeightCfNums = false;
  bool bWeightCfsWithParentInteraction = false;
  bool bOnlyWeightLongDecayParents = false;

  bool bCheckCoECoM = false;
  bool bRotateEventsByRandomAzimuthalAngles = true;
  bool bOnlyRunOverJaiEvents = false;

  double tMaxPrimaryDecayLength = -1.; 
//  double tMaxPrimaryDecayLength = 3.01; 

  int tImpactParam = 8;
  //-----------------------------------------
  TString tEventsDirectory, tMatricesSaveFileName, tPairFractionSaveName, tSingleParticlesSaveName, tCorrelationFunctionsSaveName;

  if(bRunFull)
  {
    tEventsDirectory = TString::Format("/home/jesse/Analysis/Therminator2/events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
    tMatricesSaveFileName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/TransformMatricesv2", tImpactParam);
    tPairFractionSaveName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/PairFractionsv2", tImpactParam);
    tSingleParticlesSaveName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/SingleParticleAnalysesv2", tImpactParam);
    tCorrelationFunctionsSaveName  = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/CorrelationFunctions", tImpactParam);
  }
  else
  {
    tEventsDirectory = "/home/jesse/Analysis/Therminator2/events/TestEvents/";
    tMatricesSaveFileName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testTransformMatricesv2";
    tPairFractionSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testPairFractionsv2";
    tSingleParticlesSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testSingleParticleAnalysesv2";
    tCorrelationFunctionsSaveName  = "/home/jesse/Analysis/ReducedTherminator2Events/test/testCorrelationFunctions";
  }

  if(bOnlyRunOverJaiEvents) 
  {
    tEventsDirectory += TString("events0/");

    tMatricesSaveFileName += TString("_JaiEventsOnly");
    tPairFractionSaveName += TString("_JaiEventsOnly");
    tSingleParticlesSaveName += TString("_JaiEventsOnly");
    tCorrelationFunctionsSaveName += TString("_JaiEventsOnly");
  }

  if(bRotateEventsByRandomAzimuthalAngles)
  {
    tMatricesSaveFileName += TString("_RandomEPs");
    tPairFractionSaveName += TString("_RandomEPs");
    tSingleParticlesSaveName += TString("_RandomEPs");
    tCorrelationFunctionsSaveName += TString("_RandomEPs");
  }

  if(bUnitWeightCfNums) tCorrelationFunctionsSaveName += TString("_NumWeight1");
  //-----------------------------------------

  SimpleThermAnalysis *tSimpleThermAnalysis = new SimpleThermAnalysis();
  tSimpleThermAnalysis->SetUseMixedEvents(bUseMixedEvents);
  tSimpleThermAnalysis->SetBuildUniqueParents(bPrintUniqueParents);

  tSimpleThermAnalysis->SetBuildPairFractions(bBuildPairFractions);
  tSimpleThermAnalysis->SetBuildTransformMatrices(bBuildTransformMatrices);

  tSimpleThermAnalysis->SetBuildCorrelationFunctions(bBuildCorrelationFunctions);
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
  if(tMaxPrimaryDecayLength > 0.) tSimpleThermAnalysis->SetMaxPrimaryDecayLength(tMaxPrimaryDecayLength);

  tSimpleThermAnalysis->SetCheckCoECoM(bCheckCoECoM);
  tSimpleThermAnalysis->SetRotateEventsByRandomAzimuthalAngles(bRotateEventsByRandomAzimuthalAngles);

  tSimpleThermAnalysis->ProcessAll();

//-------------------------------------------------------------------------------
  cout << "Finished program: ";
  return 0;
}
