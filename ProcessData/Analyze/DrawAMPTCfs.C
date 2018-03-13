#include "DrawAMPTCfs.h"


//_________________________________________________________________________________________
//*****************************************************************************************
//_________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

//  bool bSaveFigures = false;

  TString tResultsDate = "20180312";
  CentralityType tAnDirCentType = kMB;    //If data stored in separate directories for each centrality, 0010, 1030, 3050
                                          //    choose corresponding CentralityType
                                          //If data stored in one, large, 0100 directory, choose kMB
  if(tResultsDate.EqualTo("20180312")) tAnDirCentType = kMB;

  bool aCombineAllcLamcKch = true;
  bool aCombineConjugates = true;
  bool aCombineCentFiles = true;

  AnalysisType tAnType = kLamKchP;
  TString tCentralityFile = "3040";
  //----------
  int aRebin = 4;
  double aMinNorm=0.32, aMaxNorm=0.40;

  //---------------------------------------------------
/*
  AnalysisType tConjAnType;
  if     (aAnType==kLamK0)   {tConjAnType=kALamK0;}
  else if(aAnType==kALamK0)  {tConjAnType=kLamK0;}
  else if(aAnType==kLamKchP) {tConjAnType=kALamKchM;}
  else if(aAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(aAnType==kLamKchM) {tConjAnType=kALamKchP;}
  else if(aAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);


  vector<AnalysisType> tAnTypesVec;
  if     (aCombineAllcLamcKch) tAnTypesVec = {kLamKchP, kALamKchM, kLamKchM, kALamKchP};
  else if(aCombineConjugates)  tAnTypesVec = {tAnType, tConjAnType};
  else                         tAnTypesVec = {tAnType};

  vector<TString> tCentralityFilesVec;
  if(!aCombineCentFiles) tCentralityFilesVec = {tCentralityFile};
  else
  {
    if     (tCentralityFile.EqualTo("0005") || tCentralityFile.EqualTo("0510")) tCentralityFilesVec = vector<TString>{"0005", "0510"};
    else if(tCentralityFile.EqualTo("1020") || tCentralityFile.EqualTo("2030")) tCentralityFilesVec = vector<TString>{"1020", "2030"};
    else if(tCentralityFile.EqualTo("3040") || tCentralityFile.EqualTo("4050")) tCentralityFilesVec = vector<TString>{"3040", "4050"};
    else if(tCentralityFile.EqualTo("5060") || tCentralityFile.EqualTo("6070") || tCentralityFile.EqualTo("7080")) tCentralityFilesVec = vector<TString>{"5060", "6070", "7080"};
    else assert(0);
  }

//-----------------------------------------------------------------------------

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/AMPT/%s/",tResultsDate.Data());
  vector<TString> tFileLocationBasesVec(0);

  for(unsigned int i=0; i<tCentralityFilesVec.size(); i++) tFileLocationBasesVec.push_back(TString::Format("%sAMPT_%s.root",tDirectoryBase.Data(), tCentralityFilesVec[i].Data()));

  TString tHeavyCfName = "CfHeavy";
  for(unsigned int i=0; i<tAnTypesVec.size(); i++) tHeavyCfName += TString::Format("_%s", cAnalysisBaseTags[tAnTypesVec[i]]);
  for(unsigned int iCentFile=0; iCentFile<tCentralityFilesVec.size(); iCentFile++) tHeavyCfName += TString::Format("_%s", tCentralityFilesVec[iCentFile].Data());
  CfHeavy* tCfHeavy = BuildCfHeavy(tHeavyCfName, tFileLocationBasesVec, tCentralityFilesVec, tAnTypesVec, tAnDirCentType, aRebin, aMinNorm, aMaxNorm);
*/

  TH1* tCf = TypicalGetAMPTCf(tResultsDate, tAnDirCentType, tAnType, tCentralityFile, aCombineAllcLamcKch, aCombineConjugates, aCombineCentFiles, aRebin, aMinNorm, aMaxNorm);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();

  tCf->Draw();

//  tCfHeavy->GetHeavyCf()->Draw();

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
