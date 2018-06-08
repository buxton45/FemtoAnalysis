#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1D.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TList.h"

using std::cout;
using std::endl;
using std::vector;

#include "CfLite.h"
class CfLite;

#include "CfHeavy.h"
class CfHeavy;

#include "Types.h"

//_________________________________________ Taken from PartialAnalysis method _________________________________
TObjArray* ConnectAnalysisDirectory(TString aFileLocation, TString aDirectoryName)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TList *tFemtolist;
  TString tFemtoListName;
  TDirectoryFile *tDirFile;

  tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  if(aDirectoryName.Contains("LamKch")) tFemtoListName = "cLamcKch";
  else if(aDirectoryName.Contains("LamK0")) tFemtoListName = "cLamK0";
  else if(aDirectoryName.Contains("XiKch")) tFemtoListName = "cXicKch";
  else if(aDirectoryName.Contains("XiK0")) tFemtoListName = "cXiK0";
  else assert(0);

  tFemtoListName += TString("_femtolist");
  tFemtolist = (TList*)tDirFile->Get(tFemtoListName);
  aDirectoryName.ReplaceAll("0010","010");

  tFemtolist->SetOwner();

  TObjArray *ReturnArray = (TObjArray*)tFemtolist->FindObject(aDirectoryName)->Clone();
    ReturnArray->SetOwner();


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
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  tFemtolist->Delete();
  delete tFemtolist;

  tDirFile->Close();
  delete tDirFile;

  tFile->Close();
  delete tFile;


  return ReturnArray;
}

//___________________________________ Taken from FitPartialAnalysis method _______________________________________
TH1D* Get1dHisto(TString aFileLocation, TString aDirectoryName, TString aHistoName, TString aNewName)
{
  TObjArray* tDir = ConnectAnalysisDirectory(aFileLocation,aDirectoryName);
    tDir->SetOwner();

  TH1D *tHisto = (TH1D*)tDir->FindObject(aHistoName);
    tHisto->SetDirectory(0);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "1dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH1D *ReturnHisto = (TH1D*)tHisto->Clone(aNewName);
    ReturnHisto->SetDirectory(0);

  delete tDir;

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  return (TH1D*)ReturnHisto;
}

//________________________________________________________________________________________________________________
CfLite* BuildCfLite(TH1D* aNum, TH1D* aDen, TString aName, int aRebin=1)
{
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;

  CfLite* tCfLite = new CfLite(aName, aName, aNum, aDen, tMinNorm, tMaxNorm);
  tCfLite->Rebin(aRebin);
  return tCfLite;
}

//________________________________________________________________________________________________________________
CfHeavy* BuildCfHeavy(TString aName, vector<CfLite*> &aCfLiteCollection, double aMinNorm, double aMaxNorm)
{
  CfHeavy* tCfHeavy = new CfHeavy(aName, aName, aCfLiteCollection, aMinNorm, aMaxNorm);
  return tCfHeavy;
}

//_____________________________ Taken from ThermCf.cxx __________________________________________________________
CfHeavy* CombineTwoCfHeavy(TString aName, CfHeavy* aCfHeavy1, CfHeavy* aCfHeavy2)
{
  vector<CfLite*> tCfLiteVec1 = aCfHeavy1->GetCfLiteCollection();
  vector<CfLite*> tCfLiteVec2 = aCfHeavy2->GetCfLiteCollection();

  vector<CfLite*> tCfLiteVec(0);
  for(unsigned int i=0; i<tCfLiteVec1.size(); i++) tCfLiteVec.push_back(tCfLiteVec1[i]);
  for(unsigned int i=0; i<tCfLiteVec2.size(); i++) tCfLiteVec.push_back(tCfLiteVec2[i]);

  double tMinNorm = aCfHeavy1->GetMinNorm();
  assert(tMinNorm == aCfHeavy2->GetMinNorm());
  double tMaxNorm = aCfHeavy1->GetMaxNorm();
  assert(tMaxNorm == aCfHeavy2->GetMaxNorm());

  CfHeavy* tCfHeavy = new CfHeavy(aName, aName, tCfLiteVec, tMinNorm, tMaxNorm);
  return tCfHeavy;
}

//________________________________________________________________________________________________________________
//TODO NOTE: For this function, aCentType==kMB means 50-90%
CfHeavy* BuildMinusPlusCfHeavy(TString aFileLocationBase, AnalysisType aAnType, CentralityType aCentType)
{
  int aRebin=2;

  //---------------------------------------------------------------
  TString tCentTag = cCentralityTags[aCentType];
  if(aCentType==kMB) tCentTag = TString("_5090");
  TString tDirectoryName = TString::Format("%s%s", cAnalysisBaseTags[aAnType], tCentTag.Data());

  TString tNumName = TString::Format("%s%s", cKStarCfBaseTagNum, cAnalysisBaseTags[aAnType]);
  TString tDenName = TString::Format("%s%s", cKStarCfBaseTagDen, cAnalysisBaseTags[aAnType]);
  TString tCfName = TString::Format("Cf_%s", cAnalysisBaseTags[aAnType]);

  //---------------------------------------------------------------
  TString tFileLocation_Minus = TString::Format("%s_FemtoMinus.root", aFileLocationBase.Data());

  TH1D* tNum_Minus = Get1dHisto(tFileLocation_Minus, tDirectoryName, tNumName, TString::Format("%s_Minus", tNumName.Data()));
  TH1D* tDen_Minus = Get1dHisto(tFileLocation_Minus, tDirectoryName, tDenName, TString::Format("%s_Minus", tDenName.Data()));
  CfLite* tCfLite_Minus = BuildCfLite(tNum_Minus, tDen_Minus, TString::Format("%s_Minus", tCfName.Data()), 1);
  //-----------------
  TString tFileLocation_Plus = TString::Format("%s_FemtoPlus.root", aFileLocationBase.Data());

  TH1D* tNum_Plus = Get1dHisto(tFileLocation_Plus, tDirectoryName, tNumName, TString::Format("%s_Plus", tNumName.Data()));
  TH1D* tDen_Plus = Get1dHisto(tFileLocation_Plus, tDirectoryName, tDenName, TString::Format("%s_Plus", tDenName.Data()));
  CfLite* tCfLite_Plus = BuildCfLite(tNum_Plus, tDen_Plus, TString::Format("%s_Plus", tCfName.Data()), 1);
  //---------------------------------------------------------------
  vector<CfLite*> tCfLiteCollection {tCfLite_Minus, tCfLite_Plus};
  CfHeavy* tCfHeavy = BuildCfHeavy(tCfName, tCfLiteCollection, 0.32, 0.40);
    tCfHeavy->Rebin(aRebin);

  return tCfHeavy;
}


//________________________________________________________________________________________________________________
//TODO NOTE: For this function, aCentType==kMB means 50-90%
void DrawCf(TPad* aPad, TString aFileLocationBase, AnalysisType aAnType, CentralityType aCentType)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //---------------------------------------------------------------
  int tColor = 1;
  int tMarkerStyle = 20;

  //---------------------------------------------------------------
  CfHeavy* tCfHeavy = BuildMinusPlusCfHeavy(aFileLocationBase, aAnType, aCentType);
  TH1D* tCf = (TH1D*)tCfHeavy->GetHeavyCfClone();
    tCf->SetLineColor(tColor);
    tCf->SetMarkerColor(tColor);
    tCf->SetMarkerStyle(tMarkerStyle);


  tCf->GetXaxis()->SetTitle("k* (GeV/c)");
  tCf->GetYaxis()->SetTitle("C(k*)");

//  tCf->GetXaxis()->SetRangeUser(0.,0.329);
  tCf->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf->Draw();


  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  TString tCentTag = cCentralityTags[aCentType];
  if(aCentType==kMB) tCentTag = TString("_5090");
  tLeg->AddEntry(tCf, TString::Format("%s%s", cAnalysisBaseTags[aAnType], tCentTag.Data()));


  tLeg->Draw();
}

//________________________________________________________________________________________________________________
//TODO NOTE: For this function, aCentType==kMB means 50-90%
void DrawCf_CombineConj(TPad* aPad, TString aFileLocationBase, AnalysisType aAnType, CentralityType aCentType)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  AnalysisType aConjAnType;
  if     (aAnType==kLamK0) aConjAnType=kALamK0;
  else if(aAnType==kLamKchP) aConjAnType=kALamKchM;
  else if(aAnType==kLamKchM) aConjAnType=kALamKchP;
  else assert(0);


  //---------------------------------------------------------------
  int tColor = 1;
  int tMarkerStyle = 20;

  //---------------------------------------------------------------
  CfHeavy* tAnCfHeavy = BuildMinusPlusCfHeavy(aFileLocationBase, aAnType, aCentType);
  CfHeavy* tConjCfHeavy = BuildMinusPlusCfHeavy(aFileLocationBase, aConjAnType, aCentType);

  CfHeavy* tCfHeavy = CombineTwoCfHeavy(TString::Format("%swConj", tAnCfHeavy->GetHeavyCfName().Data()), tAnCfHeavy, tConjCfHeavy);
  TH1D* tCf = (TH1D*)tCfHeavy->GetHeavyCfClone();
    tCf->SetLineColor(tColor);
    tCf->SetMarkerColor(tColor);
    tCf->SetMarkerStyle(tMarkerStyle);


  tCf->GetXaxis()->SetTitle("k* (GeV/c)");
  tCf->GetYaxis()->SetTitle("C(k*)");

//  tCf->GetXaxis()->SetRangeUser(0.,0.329);
  tCf->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf->Draw();


  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  TString tCentTag = cCentralityTags[aCentType];
  if(aCentType==kMB) tCentTag = TString("_5090");
  tLeg->AddEntry(tCf, TString::Format("%s%s", cAnalysisBaseTags[aAnType], tCentTag.Data()));


  tLeg->Draw();
}


//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  bool bCombineConj = true;
//  bool bSaveFigures = false;

  TString tFileBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180606/Results_cLamcKch_20180606";
  TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20180607/Figures/";

  //TODO NOTE: For this function, aCentType==kMB means 50-90%
  CentralityType tCentType = kMB;
  //--------------------------------------------

  if(!bCombineConj)
  {
    TCanvas* tCanLamKchP = new TCanvas("Cfs_LamKchP", "Cfs_LamKchP");
    tCanLamKchP->Divide(2,1);
    DrawCf((TPad*)tCanLamKchP->cd(1), tFileBase, kLamKchP, tCentType);
    DrawCf((TPad*)tCanLamKchP->cd(2), tFileBase, kALamKchM, tCentType);

    TCanvas* tCanLamKchM = new TCanvas("Cfs_LamKchM", "Cfs_LamKchM");
    tCanLamKchM->Divide(2,1);
    DrawCf((TPad*)tCanLamKchM->cd(1), tFileBase, kLamKchM, tCentType);
    DrawCf((TPad*)tCanLamKchM->cd(2), tFileBase, kALamKchP, tCentType);
  }
  else
  {
    TCanvas* tCanLamKchPwConj = new TCanvas("Cfs_LamKchPwConj", "Cfs_LamKchPwConj");
    DrawCf_CombineConj((TPad*)tCanLamKchPwConj, tFileBase, kLamKchP, tCentType);

    TCanvas* tCanLamKchMwConj = new TCanvas("Cfs_LamKchMwConj", "Cfs_LamKchMwConj");
    DrawCf_CombineConj((TPad*)tCanLamKchMwConj, tFileBase, kLamKchM, tCentType);
  }


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
