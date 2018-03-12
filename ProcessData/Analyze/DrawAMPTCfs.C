#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TList.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "CanvasPartition.h"
#include "CfLite.h"
#include "CfHeavy.h"

//_________________________________________________________________________________________
TObjArray* GetAnDir(AnalysisType aAnType, CentralityType aCentType, TString aFileLocation)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TDirectoryFile *tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  TString tFemtoListName;
  if(aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP) tFemtoListName = TString("cLamcKch");
  else if(aAnType==kLamK0 || aAnType==kALamK0) tFemtoListName = TString("cLamK0");
  else if(aAnType==kXiKchP || aAnType==kAXiKchM || aAnType==kXiKchM || aAnType==kAXiKchP) tFemtoListName = TString("cXicKch");
  else assert(0);
  tFemtoListName += TString("_femtolist");

  TList *tFemtolist = (TList*)tDirFile->Get(tFemtoListName);
  tFemtolist->SetOwner();
  //----------------------------------
  TString tAnDirName = TString(cAnalysisBaseTags[aAnType]) + TString(cCentralityTags[aCentType]);
  tAnDirName.ReplaceAll("0010","010");

  TObjArray *tAnDir = (TObjArray*)tFemtolist->FindObject(tAnDirName)->Clone();
  tAnDir->SetOwner();

  //----------------------------------
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtolist object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtolist);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }

  tFemtolist->Delete();
  delete tFemtolist;

  tDirFile->Close();
  delete tDirFile;

  tFile->Close();
  delete tFile;


  //----------------------------------

  return tAnDir;
}

//_________________________________________________________________________________________
TH1* Get1dHist(TObjArray* aAnDir, TString aHistName, TString aNewName="")
{
  TH1* tHist = (TH1*)aAnDir->FindObject(aHistName);
  tHist->SetDirectory(0);

  //-----make sure tHisto is retrieved
  if(!tHist) {cout << "1dHist NOT FOUND!!!:  Name:  " << aHistName << endl;}
  assert(tHist);
  //----------------------------------

  TH1 *ReturnHist = (TH1*)tHist->Clone(aNewName);
    ReturnHist->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHist->GetSumw2N()) {ReturnHist->Sumw2();}

  return (TH1*)ReturnHist;
}




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

  bool aCombineAllcLamcKch = true;

  TString tResultsDate = "20180308";

  AnalysisType tAnType = kLamKchP;

  CentralityType tCentType = k3050;
  TString tCentralityFile = "3040";

  int aRebin = 4;
  double aMinNorm=0.32, aMaxNorm=0.40;

  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;

  AnalysisType tConjAnType;
  if(tAnType==kLamK0) {tConjAnType=kALamK0;}
  else if(tAnType==kLamKchP) {tConjAnType=kALamKchM;}
  else if(tAnType==kLamKchM) {tConjAnType=kALamKchP;}

//-----------------------------------------------------------------------------
  bool bSaveFigures = false;

//-----------------------------------------------------------------------------



  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/AMPT/%s/",tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sAMPT_%s.root",tDirectoryBase.Data(), tCentralityFile.Data());


  if(!aCombineAllcLamcKch)
  {
    TObjArray* tAnDir = GetAnDir(tAnType, tCentType, tFileLocationBase);

    TH1* tNum = Get1dHist(tAnDir, TString::Format("NumKStarCf_%s", cAnalysisBaseTags[tAnType]));
    TH1* tDen = Get1dHist(tAnDir, TString::Format("DenKStarCf_%s", cAnalysisBaseTags[tAnType]));

    CfLite* tCfLite = new CfLite(TString::Format("CfLite_%s_%s", cAnalysisBaseTags[tAnType], tCentralityFile.Data()), 
                                 TString::Format("CfLite_%s_%s", cAnalysisBaseTags[tAnType], tCentralityFile.Data()), 
                                 tNum, tDen, aMinNorm, aMaxNorm);
    tCfLite->Rebin(aRebin);


    TCanvas* tCan = new TCanvas("tCan", "tCan");
    tCan->cd();

    tCfLite->Cf()->Draw();
  }
  else
  {
    TObjArray* tAnDir_LamKchP = GetAnDir(kLamKchP, tCentType, tFileLocationBase);

    TH1* tNum_LamKchP = Get1dHist(tAnDir_LamKchP, TString::Format("NumKStarCf_%s", cAnalysisBaseTags[kLamKchP]));
    TH1* tDen_LamKchP = Get1dHist(tAnDir_LamKchP, TString::Format("DenKStarCf_%s", cAnalysisBaseTags[kLamKchP]));

    CfLite* tCfLite_LamKchP = new CfLite(TString::Format("CfLite_%s_%s", cAnalysisBaseTags[kLamKchP], tCentralityFile.Data()), 
                                 TString::Format("CfLite_%s_%s", cAnalysisBaseTags[kLamKchP], tCentralityFile.Data()), 
                                 tNum_LamKchP, tDen_LamKchP, aMinNorm, aMaxNorm);
    tCfLite_LamKchP->Rebin(aRebin);
    //---------------------------------
    TObjArray* tAnDir_ALamKchM = GetAnDir(kALamKchM, tCentType, tFileLocationBase);

    TH1* tNum_ALamKchM = Get1dHist(tAnDir_ALamKchM, TString::Format("NumKStarCf_%s", cAnalysisBaseTags[kALamKchM]));
    TH1* tDen_ALamKchM = Get1dHist(tAnDir_ALamKchM, TString::Format("DenKStarCf_%s", cAnalysisBaseTags[kALamKchM]));

    CfLite* tCfLite_ALamKchM = new CfLite(TString::Format("CfLite_%s_%s", cAnalysisBaseTags[kALamKchM], tCentralityFile.Data()), 
                                 TString::Format("CfLite_%s_%s", cAnalysisBaseTags[kALamKchM], tCentralityFile.Data()), 
                                 tNum_ALamKchM, tDen_ALamKchM, aMinNorm, aMaxNorm);
    tCfLite_ALamKchM->Rebin(aRebin);
    //---------------------------------
    TObjArray* tAnDir_LamKchM = GetAnDir(kLamKchM, tCentType, tFileLocationBase);

    TH1* tNum_LamKchM = Get1dHist(tAnDir_LamKchM, TString::Format("NumKStarCf_%s", cAnalysisBaseTags[kLamKchM]));
    TH1* tDen_LamKchM = Get1dHist(tAnDir_LamKchM, TString::Format("DenKStarCf_%s", cAnalysisBaseTags[kLamKchM]));

    CfLite* tCfLite_LamKchM = new CfLite(TString::Format("CfLite_%s_%s", cAnalysisBaseTags[kLamKchM], tCentralityFile.Data()), 
                                 TString::Format("CfLite_%s_%s", cAnalysisBaseTags[kLamKchM], tCentralityFile.Data()), 
                                 tNum_LamKchM, tDen_LamKchM, aMinNorm, aMaxNorm);
    tCfLite_LamKchM->Rebin(aRebin);
    //---------------------------------
    TObjArray* tAnDir_ALamKchP = GetAnDir(kALamKchP, tCentType, tFileLocationBase);

    TH1* tNum_ALamKchP = Get1dHist(tAnDir_ALamKchP, TString::Format("NumKStarCf_%s", cAnalysisBaseTags[kALamKchP]));
    TH1* tDen_ALamKchP = Get1dHist(tAnDir_ALamKchP, TString::Format("DenKStarCf_%s", cAnalysisBaseTags[kALamKchP]));

    CfLite* tCfLite_ALamKchP = new CfLite(TString::Format("CfLite_%s_%s", cAnalysisBaseTags[kALamKchP], tCentralityFile.Data()), 
                                 TString::Format("CfLite_%s_%s", cAnalysisBaseTags[kALamKchP], tCentralityFile.Data()), 
                                 tNum_ALamKchP, tDen_ALamKchP, aMinNorm, aMaxNorm);
    tCfLite_ALamKchP->Rebin(aRebin);
    //---------------------------------

    TString tHeavyCfName = TString::Format("CfHeavy_cLamcKch_%s",tCentralityFile.Data());
    vector<CfLite*> tCfLiteVec = {tCfLite_LamKchP, tCfLite_ALamKchM, tCfLite_LamKchM, tCfLite_ALamKchP};
    CfHeavy* tCfHeavy = new CfHeavy(tHeavyCfName, tHeavyCfName, tCfLiteVec, aMinNorm, aMaxNorm);

    TCanvas* tCan = new TCanvas("tCan", "tCan");
    tCan->cd();

    tCfHeavy->GetHeavyCf()->Draw();
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
