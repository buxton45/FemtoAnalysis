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

  //-----------------------------------------
  TString tEventsDirectory, tMatricesSaveFileName, tPairFractionSaveName;

  if(bRunFull)
  {
    tEventsDirectory = "/home/jesse/Analysis/Therminator2/events/lhyqid3v_LHCPbPb_2760_b2/";
    tMatricesSaveFileName = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/TransformMatricesv2.root";
    tPairFractionSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/PairFractionsv2.root";
  }
  else
  {
    tEventsDirectory = "/home/jesse/Analysis/Therminator2/events/TestEvents/";
    tMatricesSaveFileName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testTransformMatricesv2.root";
    tPairFractionSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testPairFractionsv2.root";
  }
  //-----------------------------------------

  SimpleThermAnalysis *tSimpleThermAnalysis = new SimpleThermAnalysis();
  tSimpleThermAnalysis->SetUseMixedEvents(bUseMixedEvents);
  tSimpleThermAnalysis->SetBuildUniqueParents(bPrintUniqueParents);

  tSimpleThermAnalysis->SetEventsDirectory(tEventsDirectory);
  tSimpleThermAnalysis->SetPairFractionsSaveName(tPairFractionSaveName);
  tSimpleThermAnalysis->SetTransformMatricesSaveName(tMatricesSaveFileName);

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
