#include "ThermEventsCollection.h"
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

  ThermEventsCollection *tEventsCollection = new ThermEventsCollection();
  tEventsCollection->SetUseMixedEvents(true);

  bool bRunFull = false;
  bool bWriteEvents = false;
  bool bReadFromRoot = true;
  bool bReadFromTxt = false;
  bool bPrintUniqueParents = false;
  assert(!(bReadFromRoot && bReadFromTxt));

  int tImpactParam = 2;

  TString tEventsDirectory, tEventsSaveFileNameBase, tMatricesSaveFileName, tPairFractionSaveName;
  //-----------------------------------------
  if(bRunFull)
  {
    tEventsDirectory = TString::Format("/home/jesse/Analysis/Therminator2/events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
    tEventsSaveFileNameBase = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
    tMatricesSaveFileName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/TransformMatrices.root", tImpactParam);
    tPairFractionSaveName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/PairFractions.root", tImpactParam);
  }
  else
  {
    tEventsDirectory = "/home/jesse/Analysis/Therminator2/events/TestEvents/";
    tEventsSaveFileNameBase = "/home/jesse/Analysis/ReducedTherminator2Events/test/test";
    tMatricesSaveFileName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testTransformMatrices.root";
    tPairFractionSaveName = "/home/jesse/Analysis/ReducedTherminator2Events/test/testPairFractions.root";
  }
  //-----------------------------------------
  if(bWriteEvents)
  {
    tEventsCollection->ExtractFromAllRootFiles(tEventsDirectory, bPrintUniqueParents);
    tEventsCollection->WriteAllEvents(tEventsSaveFileNameBase);
  }

  //-----------------------------------------
  if(bReadFromRoot) tEventsCollection->ExtractFromAllRootFiles(tEventsDirectory, bPrintUniqueParents);
  if(bReadFromTxt) tEventsCollection->ExtractEventsFromAllTxtFiles(tEventsSaveFileNameBase);

  //-----------------------------------------


  tEventsCollection->SaveAllTransformMatrices(tMatricesSaveFileName);
  tEventsCollection->SaveAllPairFractionHistograms(tPairFractionSaveName);
  TCanvas* tCan = tEventsCollection->DrawAllPairFractionHistograms();
  if(bPrintUniqueParents) tEventsCollection->PrintUniqueParents();
  tEventsCollection->PrintAllPrimaryAndOtherPairInfo();
//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
