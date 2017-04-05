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
#include "TLegend.h"
#include "TLegendEntry.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;


TH1* GetPurityEstimator(TString aFileLocation, ParticleType aV0Type, bool aGetNum=true, CentralityType aCentType = k0010)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TDirectoryFile *tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  TString tDirectoryName;
  TString tFemtoListName;
  if(aV0Type==kALam)
  {
    tFemtoListName = "AntiLambdaPurityBgdEstimator";
    tDirectoryName = TString("AProtPiP");
  }
  else if(aV0Type==kLam)
  {
    tFemtoListName = "LambdaPurityBgdEstimator";
    tDirectoryName = TString("ProtPiM");
  }
  else if(aV0Type==kK0)
  {
    tFemtoListName = "K0ShortPurityBgdEstimator";
    tDirectoryName = TString("PiPPiM");
  }
  else
  {
    cout << "ERROR in GetPurityEstimator!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    cout << "Invalid aV0Type = " << aV0Type << endl;
    assert(0);
  }
  tFemtoListName += TString("_femtolist");
  tDirectoryName += cCentralityTags[aCentType];
    tDirectoryName.ReplaceAll("0010","010");

  TList *tFemtoList;
  tFemtoList = (TList*)tDirFile->Get(tFemtoListName);
  TObjArray *tContents = (TObjArray*)tFemtoList->FindObject(tDirectoryName)->Clone();
    tContents->SetOwner();


  TString tHistoName;
  if(aGetNum) tHistoName = "NumV0PurityBgdEstimator_";
  else tHistoName = "DenV0PurityBgdEstimator_";
  if(aV0Type==kALam) tHistoName += TString("AntiLambda");
  else if(aV0Type==kLam) tHistoName += TString("Lambda");
  else if(aV0Type==kK0) tHistoName += TString("K0Short");

  else
  {
    cout << "ERROR2 in GetPurityEstimator!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    cout << "Invalid aV0Type = " << aV0Type << endl;
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
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

//  ParticleType tV0Type = kLam;
  ParticleType tV0Type = kK0;

  //-----Analysis
  TString AnalysisFileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cLamK0_20161027/Results_cLamK0_20161027";
  Analysis* LamK0 = new Analysis(AnalysisFileLocationBase,kLamK0,k0010);
  Analysis* ALamK0 = new Analysis(AnalysisFileLocationBase,kALamK0,k0010);

  TString SaveFileName = "~/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/0010/Results_cLamK0_AsRc_20150923_0010TEST.root";

  //-----V0PurityBgdEstimator
  TString V0PurityBgdFileLocationBase = "~/Analysis/FemtoAnalysis/Results/V0PurityBgdEstimator/";
  if(tV0Type==kLam) V0PurityBgdFileLocationBase += TString("Lambda");
  else if(tV0Type==kALam) V0PurityBgdFileLocationBase += TString("AntiLambda");
  else if(tV0Type==kK0) V0PurityBgdFileLocationBase += TString("K0Short");
  else V0PurityBgdFileLocationBase += TString("");
  V0PurityBgdFileLocationBase += TString("PurityBgdEstimator");

  TString V0PurityBgdMinus = V0PurityBgdFileLocationBase + TString("_FemtoMinus.root");
  TString V0PurityBgdPlus = V0PurityBgdFileLocationBase + TString("_FemtoPlus.root");

  TH1* tNumMinus = GetPurityEstimator(V0PurityBgdMinus,tV0Type,true,k0010);
  TH1* tNumPlus = GetPurityEstimator(V0PurityBgdPlus,tV0Type,true,k0010);

  TH1* tDenMinus = GetPurityEstimator(V0PurityBgdMinus,tV0Type,false,k0010);
  TH1* tDenPlus = GetPurityEstimator(V0PurityBgdPlus,tV0Type,false,k0010);

  vector<TH1*> tVecTH1Num = {tNumMinus,tNumPlus};
  Purity *tV0BgdEstNum = new Purity("tV0BgdEstNum", tV0Type, tVecTH1Num);
  double tV0BgdEstNumSignalPlusBgd = tV0BgdEstNum->GetSignalPlusBgd();

  vector<TH1*> tVecTH1Den = {tDenMinus,tDenPlus};
  Purity *tV0BgdEstDen = new Purity("tV0BgdEsDen", tV0Type, tVecTH1Den);



//-----------------------------------------------------------------------------
  bool bSaveFigures = false;
  TString tSaveFiguresLocation = "~/Analysis/Presentations/";
//-------------------------------------------------------------------

  LamK0->BuildPurityCollection();
  ALamK0->BuildPurityCollection();

  TH1* tDataPurity = LamK0->GetCombinedPurityHisto(tV0Type);
  tDataPurity->SetMarkerStyle(20);

  double tTestSignalPlusBgd = LamK0->GetPurityObject(tV0Type)->GetSignalPlusBgd();

  //-------------------------------------------------------------

  bool bZoomBgd = true;
  TCanvas* canPurity = new TCanvas("canPurity","canPurity");
  TH1* tV0BgdEstNumHist = tV0BgdEstNum->GetCombinedPurity();
  cout << "tTestSignalPlusBgd/tV0BgdEstNumSignalPlusBgd = scale = " << tTestSignalPlusBgd/tV0BgdEstNumSignalPlusBgd << endl;
  tV0BgdEstNumHist->Scale(tTestSignalPlusBgd/tV0BgdEstNumSignalPlusBgd);
  tV0BgdEstNumHist->SetMarkerStyle(20);
  tV0BgdEstNumHist->SetMarkerColor(2);
  canPurity->cd();
  if(bZoomBgd)
  {
//TODO
    if(tV0BgdEstNumHist->GetMinimum() > tDataPurity->GetMinimum()) tV0BgdEstNumHist->GetYaxis()->SetRangeUser(0.25*tV0BgdEstNumHist->GetMinimum(),3.0*tV0BgdEstNumHist->GetMinimum());
    else tV0BgdEstNumHist->GetYaxis()->SetRangeUser(0.8*tDataPurity->GetMinimum(),2.5*tDataPurity->GetMinimum());
  }
  tV0BgdEstNumHist->Draw();
  tDataPurity->Draw("same");


  TLegend *tLeg1 = new TLegend(0.65,0.70,0.89,0.89);
    tLeg1->SetFillColor(0);
    tLeg1->SetHeader(cRootParticleTags[tV0Type]);
    tLeg1->AddEntry(tDataPurity, "Data", "p");
    tLeg1->AddEntry(tV0BgdEstNumHist, "Same Event Pairs", "p");
    TLegendEntry* tHeader = (TLegendEntry*)tLeg1->GetListOfPrimitives()->First();
    tHeader->SetTextAlign(22);
    tLeg1->Draw();

  //-------------------------------------------------------------


  TCanvas* canPurity2 = new TCanvas("canPurity2", "canPurity2");
  TH1* tV0BgdEstDenHist = tV0BgdEstDen->GetCombinedPurity();
  tV0BgdEstDenHist->SetMarkerStyle(20);
  tV0BgdEstDenHist->SetMarkerColor(4);
  canPurity2->cd();
  double tScaleNum=0, tScaleDen=0;
  if(tV0Type==kLam || tV0Type==kALam)
  {
    tScaleNum = tV0BgdEstNumHist->Integral(1,tV0BgdEstNumHist->FindBin(1.095));
    tScaleDen = tV0BgdEstDenHist->Integral(1,tV0BgdEstDenHist->FindBin(1.095));
  }
  else if(tV0Type==kK0)
  {
    tScaleNum = tV0BgdEstNumHist->Integral(1,tV0BgdEstNumHist->FindBin(0.46));
    tScaleDen = tV0BgdEstDenHist->Integral(1,tV0BgdEstDenHist->FindBin(0.46));
  }


  tV0BgdEstDenHist->Scale(tScaleNum/tScaleDen);
  tV0BgdEstDenHist->GetYaxis()->SetRangeUser(0.9*tV0BgdEstDenHist->GetMinimum(), 1.1*tV0BgdEstDenHist->GetMaximum());
  tV0BgdEstDenHist->Draw();
  tV0BgdEstNumHist->Draw("same");

  TLegend *tLeg2 = new TLegend(0.65,0.70,0.89,0.89);
    tLeg2->SetFillColor(0);
    tLeg2->SetHeader(cRootParticleTags[tV0Type]);
    tLeg2->AddEntry(tV0BgdEstNumHist, "Same Event Pairs", "p");
    tLeg2->AddEntry(tV0BgdEstDenHist, "Mixed Event Pairs", "p");
    TLegendEntry* tHeader2 = (TLegendEntry*)tLeg2->GetListOfPrimitives()->First();
    tHeader2->SetTextAlign(22);
    tLeg2->Draw();


  if(bSaveFigures)
  {
    TString tName1 = "DataVsNum";
      tName1 += cParticleTags[tV0Type];
      if(bZoomBgd) tName1 += TString("_Zoomed");
      tName1 += TString(".eps");
      canPurity->SaveAs(tSaveFiguresLocation+tName1);

    TString tName2 = "NumVsDen";
      tName2 += cParticleTags[tV0Type];
      tName2 += TString(".eps");
      canPurity2->SaveAs(tSaveFiguresLocation+tName2);
  }


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
