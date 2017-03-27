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

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;


TH1* GetPurityNumerator(TString aFileLocation, CentralityType aCentType = k0010)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TDirectoryFile *tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  TString tDirectoryName;
  TString tFemtoListName;
  if(aFileLocation.Contains("AntiLambda"))
  {
    tFemtoListName = "AntiLambdaPurityBgdEstimator";
    tDirectoryName = TString("AProtPiP");
  }
  else if(aFileLocation.Contains("Lambda"))
  {
    tFemtoListName = "LambdaPurityBgdEstimator";
    tDirectoryName = TString("ProtPiM");
  }
  else if(aFileLocation.Contains("K0Short"))
  {
    tFemtoListName = "K0ShortPurityBgdEstimator";
    tDirectoryName = TString("PiPPiM");
  }
  else
  {
    cout << "ERROR in GetPurityNumerators!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    cout << "Invalid aFileLocation = " << aFileLocation << endl;
    assert(0);
  }
  tFemtoListName += TString("_femtolist");
  tDirectoryName += cCentralityTags[aCentType];
    tDirectoryName.ReplaceAll("0010","010");

  TList *tFemtoList;
  tFemtoList = (TList*)tDirFile->Get(tFemtoListName);
  TObjArray *tContents = (TObjArray*)tFemtoList->FindObject(tDirectoryName)->Clone();
    tContents->SetOwner();


  TString tHistoName = "NumV0PurityBgdEstimator_";
  if(aFileLocation.Contains("AntiLambda")) tHistoName += TString("AntiLambda");
  else if(aFileLocation.Contains("Lambda")) tHistoName += TString("Lambda");
  else if(aFileLocation.Contains("K0Short")) tHistoName += TString("K0Short");

  else
  {
    cout << "ERROR2 in GetPurityNumerators!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    cout << "Invalid aFileLocation = " << aFileLocation << endl;
    assert(0);
  }

  TH1* tReturnHist = (TH1*)tContents->FindObject(tHistoName);

  //-----make sure tReturnHist is retrieved
  if(!tReturnHist) {cout << "1dHisto NOT FOUND!!!:  Name:  " << tHistoName << endl;}
  assert(tReturnHist);
  //----------------------------------

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtolist object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtoList);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  tFemtoList->Delete();
  delete tFemtoList;

  tFile->Close();
  delete tFile;

  return (TH1*)tReturnHist;
}


int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  //-----Analysis
  TString AnalysisFileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cLamK0_20161027/Results_cLamK0_20161027";
  Analysis* LamK0 = new Analysis(AnalysisFileLocationBase,kLamK0,k0010);
  Analysis* ALamK0 = new Analysis(AnalysisFileLocationBase,kALamK0,k0010);

  TString SaveFileName = "~/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/0010/Results_cLamK0_AsRc_20150923_0010TEST.root";

  //-----V0PurityBgdEstimator
  TString V0PurityBgdFileLocationBase = "~/Analysis/FemtoAnalysis/Results/V0PurityBgdEstimator/LambdaPurityBgdEstimator";
  TString V0PurityBgdMinus = V0PurityBgdFileLocationBase + TString("_FemtoMinus.root");
  TString V0PurityBgdPlus = V0PurityBgdFileLocationBase + TString("_FemtoPlus.root");

  TH1* tNumMinus = GetPurityNumerator(V0PurityBgdMinus,k0010);
  TH1* tNumPlus = GetPurityNumerator(V0PurityBgdPlus,k0010);

  vector<TH1*> tVecTH1 = {tNumMinus,tNumPlus};
  Purity *tV0BgdEstNum = new Purity("tV0BgdEstNum", kLam, tVecTH1);

  TCanvas* canPurity2 = new TCanvas("canPurity2","canPurity2");
  tV0BgdEstNum->DrawPurity((TPad*)canPurity2->cd(1));

//-----------------------------------------------------------------------------
//  bool bSaveFigures = false;
//  TString tSaveFiguresLocation = "~/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/0010/";
//-------------------------------------------------------------------

  LamK0->BuildPurityCollection();
  ALamK0->BuildPurityCollection();

  TCanvas* canPurity = new TCanvas("canPurity","canPurity");
  canPurity->Divide(2,1);

  LamK0->DrawAllPurityHistos((TPad*)canPurity->cd(1));
  ALamK0->DrawAllPurityHistos((TPad*)canPurity->cd(2));

/*
  if(bSaveFigures)
  {
    TString aName = "cLamK0Purity.eps";
    canPurity->SaveAs(tSaveFiguresLocation+aName);

    TString aName2 = "LamPurity_LamK0.eps";
    canPurity->cd(1)->cd(1)->SaveAs(tSaveFiguresLocation+aName2);

    TString aName3 = "K0Purity_LamK0.eps";
    canPurity->cd(1)->cd(2)->SaveAs(tSaveFiguresLocation+aName3);
  }
*/

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
