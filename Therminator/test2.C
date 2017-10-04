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
  bool bRunFull = false;
  bool bUseMixedEvents = true;
  bool bPrintUniqueParents = false;

  bool bBuildPairFractions = false;
  bool bBuildTransformMatrices = false;
  bool bBuildSingleParticleAnalyses = false;

  bool bBuildCorrelationFunctions = true;
  bool bBuildMixedEventNumerators = true;
  int tNEventsToMix = 5;
  bool bWeightCfsWithParentInteraction = false;

//  double tMaxPrimaryDecayLength = -1.; 
  double tMaxPrimaryDecayLength = 3.01; 

  //-----------------------------------------
  TString tEventsDirectory, tMatricesSaveFileName, tPairFractionSaveName, tSingleParticlesSaveName, tCorrelationFunctionsSaveName;

  if(bRunFull)
  {
    tEventsDirectory = "/home/jesse/Analysis/Therminator2/events/lhyqid3v_LHCPbPb_2760_b2/";
    tMatricesSaveFileName = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/TransformMatricesv2";
    tPairFractionSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/PairFractionsv2";
    tSingleParticlesSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/SingleParticleAnalysesv2";
    tCorrelationFunctionsSaveName  = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions";
  }
  else
  {
    tEventsDirectory = "/home/jesse/Analysis/Therminator2/events/TestEvents/";
    tMatricesSaveFileName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testTransformMatricesv2";
    tPairFractionSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testPairFractionsv2";
    tSingleParticlesSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testSingleParticleAnalysesv2";
    tCorrelationFunctionsSaveName  = "/home/jesse/Analysis/ReducedTherminator2Events/test/testCorrelationFunctions";
  }
  //-----------------------------------------

  SimpleThermAnalysis *tSimpleThermAnalysis = new SimpleThermAnalysis();
  tSimpleThermAnalysis->SetUseMixedEvents(bUseMixedEvents);
  tSimpleThermAnalysis->SetBuildUniqueParents(bPrintUniqueParents);

  tSimpleThermAnalysis->SetBuildPairFractions(bBuildPairFractions);
  tSimpleThermAnalysis->SetBuildTransformMatrices(bBuildTransformMatrices);

  tSimpleThermAnalysis->SetBuildCorrelationFunctions(bBuildCorrelationFunctions);
  tSimpleThermAnalysis->SetBuildMixedEventNumerators(bBuildMixedEventNumerators);
  tSimpleThermAnalysis->SetWeightCfsWithParentInteraction(bWeightCfsWithParentInteraction);
  tSimpleThermAnalysis->SetNEventsToMix(tNEventsToMix);

  tSimpleThermAnalysis->SetBuildSingleParticleAnalyses(bBuildSingleParticleAnalyses);

  tSimpleThermAnalysis->SetEventsDirectory(tEventsDirectory);
  tSimpleThermAnalysis->SetPairFractionsSaveName(tPairFractionSaveName);
  tSimpleThermAnalysis->SetTransformMatricesSaveName(tMatricesSaveFileName);
  tSimpleThermAnalysis->SetSingleParticlesSaveName(tSingleParticlesSaveName);
  tSimpleThermAnalysis->SetCorrelationFunctionsSaveName(tCorrelationFunctionsSaveName);
  if(tMaxPrimaryDecayLength > 0.) tSimpleThermAnalysis->SetMaxPrimaryDecayLength(tMaxPrimaryDecayLength);

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
