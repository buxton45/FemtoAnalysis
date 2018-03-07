#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cassert>
#include <stdio.h>

#include "TFile.h"
#include "TSystemDirectory.h"
#include "TSystemFile.h"
#include "TApplication.h"
#include "TRegexp.h"
#include "TSystem.h"

using namespace std;
//________________________________________________________________________________________________________________
void IncrementAllNamesInDirectory(TString aEventsDirLocation, int aIncrement)
{
  TSystemDirectory *tEventsDirectory = new TSystemDirectory(aEventsDirLocation.Data(), aEventsDirLocation.Data());


  TString tCompleteDirectoryPath = tEventsDirectory->GetTitle();
  if(!tCompleteDirectoryPath.EndsWith("/")) tCompleteDirectoryPath += TString("/");

  cout << "Incrementing event names in directory: " << tCompleteDirectoryPath << " by " << aIncrement << endl;
  cout << "Will be placed in ShiftedEvents subdirectory" << endl;
  //NOTE: Important to place the renamed files in a different directory to prevent any overwriting to happen
  //      If, for instance, aIncrement is less than the total number of files, files will be overwritten
  //      ex. if aIncrement=100 and there are 150 files, event000->event100, which already exists!


  TString tSubDirName = "ShiftedEvents/";
  gSystem->mkdir(tCompleteDirectoryPath+tSubDirName, true);

  TList* tFiles = tEventsDirectory->GetListOfFiles();
  int tNFilesInDir = 0;

  const char* tBeginningText = "event";
  const char* tEndingText = ".root";

  if(tFiles)
  {
    TSystemFile* tFile;
    TString tName;
    TIter tIterNext(tFiles);

    int tIndexLocationBeg=-1, tIndexLocationEnd=-1;
    int tCurrentIndex=-1, tNewIndex=-1;
    int tSuccess=-1;

    TString tCompleteOldName, tCompleteNewName;

    while((tFile=(TSystemFile*)tIterNext()))
    {
      tName = tFile->GetName();
      if(!tFile->IsDirectory() && tName.BeginsWith(tBeginningText) && tName.EndsWith(tEndingText))
      {
        tNFilesInDir++;

        tIndexLocationBeg = tName.Index(TRegexp(tBeginningText));
          tIndexLocationBeg += 5; //to account for length of "event"
        tIndexLocationEnd = tName.Index(TRegexp(tEndingText));

        assert(tIndexLocationBeg > -1 && tIndexLocationEnd > -1);

        tCurrentIndex = ((TString)tName(tIndexLocationBeg, (tIndexLocationEnd-tIndexLocationBeg))).Atoi();

        tNewIndex = tCurrentIndex + aIncrement;
        tCompleteOldName = tCompleteDirectoryPath + tName;
        tCompleteNewName = TString::Format("%s%s%s%03i%s", tCompleteDirectoryPath.Data(), tSubDirName.Data(), tBeginningText, tNewIndex, tEndingText);

        cout << "tCompleteOldName = " << tCompleteOldName << endl;
        cout << "tCompleteNewName = " << tCompleteNewName << endl << endl;

        tSuccess = rename(tCompleteOldName.Data(), tCompleteNewName.Data());
        assert(tSuccess==0);
      }
    }
  }
  cout << endl << "Total number of files in subdirectory = " << tNFilesInDir << endl;
}





//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  int tImpactParam = 2;
  const char* tSubDir = "events7a";

  TString tDirectory = TString::Format("/home/jesse/Analysis/Therminator2/events/lhyqid3v_LHCPbPb_2760_b%d/%s", tImpactParam, tSubDir);
  int tIncrement = 100;


  IncrementAllNamesInDirectory(tDirectory, tIncrement);

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}




