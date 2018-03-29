#include "SimpleThermAnalysis.h"
//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  bool bRunLocal = true;
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

  bool bUnitWeightCfNums = false;
  bool bWeightCfsWithParentInteraction = false;
  bool bOnlyWeightLongDecayParents = false;

  bool bCheckCoECoM = false;
  bool bRotateEventsByRandomAzimuthalAngles = true;
  bool bOnlyRunOverJaiEvents = false;

  double tMaxPrimaryDecayLength = -1.; 
//  double tMaxPrimaryDecayLength = 4.01; 

  double tImpactParam = 3.0; /*3., 5.7, 7.4, 8.7, 9.9 */
  //-----------------------------------------
  TString tEventsBase, tSaveBase;
  if(bRunLocal)
  {
    tEventsBase = "/home/jesse/Analysis/Therminator2/events/";
    tSaveBase = "/home/jesse/Analysis/ReducedTherminator2Events/";
  }
  else
  {
    tEventsBase = "/data/alicesrv1d1/Models/";
    tSaveBase = "/home/jbuxton/Results/";
  }

  TString tLhyqid3vDir;
  if(tImpactParam==3.) tLhyqid3vDir = TString::Format("lhyqid3v_LHCPbPb_2760_b%0.0f/", tImpactParam);
  else                 tLhyqid3vDir = TString::Format("lhyqid3v_LHCPbPb_2760_b%0.1f/", tImpactParam);

  TString tEventsDirectory, tMatricesSaveFileName, tPairFractionSaveName, tSingleParticlesSaveName, tCorrelationFunctionsSaveName;
  if(bRunFull)
  {
    tEventsDirectory = TString::Format("%s%s", tEventsBase.Data(), tLhyqid3vDir.Data());
    tMatricesSaveFileName = TString::Format("%s%sTransformMatricesv2", tSaveBase.Data(), tLhyqid3vDir.Data());
    tPairFractionSaveName = TString::Format("%s%sPairFractionsv2", tSaveBase.Data(), tLhyqid3vDir.Data());
    tSingleParticlesSaveName = TString::Format("%s%sSingleParticleAnalysesv2", tSaveBase.Data(), tLhyqid3vDir.Data());
    tCorrelationFunctionsSaveName  = TString::Format("%s%sCorrelationFunctions", tSaveBase.Data(), tLhyqid3vDir.Data());
  }
  else
  {
    tEventsBase = "/home/jbuxton/";
    tEventsDirectory = TString::Format("%sTestEvents/", tEventsBase.Data());
    tMatricesSaveFileName = TString::Format("%stest/testTransformMatricesv2", tSaveBase.Data());
    tPairFractionSaveName = TString::Format("%stest/testPairFractionsv2", tSaveBase.Data());
    tSingleParticlesSaveName = TString::Format("%stest/testSingleParticleAnalysesv2", tSaveBase.Data());
    tCorrelationFunctionsSaveName  = TString::Format("%stest/testCorrelationFunctions", tSaveBase.Data());
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
  if(tMaxPrimaryDecayLength > 0.) tSimpleThermAnalysis->SetMaxPrimaryDecayLength(tMaxPrimaryDecayLength);

  tSimpleThermAnalysis->SetCheckCoECoM(bCheckCoECoM);
  tSimpleThermAnalysis->SetRotateEventsByRandomAzimuthalAngles(bRotateEventsByRandomAzimuthalAngles);

  tSimpleThermAnalysis->ProcessAll();

//-------------------------------------------------------------------------------
  cout << "Finished program: " << endl;
  return 0;
}
