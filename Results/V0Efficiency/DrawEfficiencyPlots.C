#include <iostream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TDirectoryFile.h"
#include "TLegend.h"



//________________________________________________________________________________________________________________
TString GetBinLabels(int aBin)
{
  if     (aBin==1)  return TString("Unassigned");
  else if(aBin==2)  return TString("Primary");
  else if(aBin==3)  return TString("#Sigma^{0}");
  else if(aBin==4)  return TString("#Xi^{0}");
  else if(aBin==5)  return TString("#Xi^{-}");
  else if(aBin==6)  return TString("#Sigma^{*0}");
  else if(aBin==7)  return TString("#Sigma^{*+}");
  else if(aBin==8)  return TString("#Sigma^{*-}");
  else if(aBin==9)  return TString("K^{*0}");
  else if(aBin==10) return TString("K^{*ch}");
  else if(aBin==11) return TString("Other");
  else if(aBin==12) return TString("Fake");
  else if(aBin==12) return TString("W.E.");
  else assert(0);

  return TString("");
}


//________________________________________________________________________________________________________________
TList* GetAllHistograms(TString aFileLocation)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TList *tReturnList;
  TDirectoryFile *tDirFile;

  tDirFile = (TDirectoryFile*)tFile->Get("Results");
  tReturnList = (TList*)tDirFile->Get("MyListStudy");
    tReturnList->SetOwner();

  //------------------------
  tDirFile->Close();
  delete tDirFile;

  tFile->Close();
  delete tFile;
  //------------------------

  return tReturnList;
}

//________________________________________________________________________________________________________________
TList* SimpleCombineBFields(TList* aListMinus, TList* aListPlus)
{
  assert(aListMinus->GetEntries() == aListPlus->GetEntries());
  for(int i=0; i<aListMinus->GetEntries(); i++) assert(aListMinus->At(i)->GetName() == aListPlus->At(i)->GetName());

  TList* tReturnList = new TList();
  TH1F *tHistMinus, *tHistPlus, *tHistCombined;
  for(int i=0; i<aListMinus->GetEntries(); i++)
  {
/*
    TH1F* tHist = (TH1F*)aListMinus->At(i)->Clone();
    tHist->Add((TH1F*)aListPlus->At(i));
    tReturnList->Add(tHist);
*/
    tHistMinus = (TH1F*)aListMinus->At(i);
    tHistPlus = (TH1F*)aListPlus->At(i);

    //Make sure everything appears in order
    assert(tHistMinus->GetNbinsX() == tHistPlus->GetNbinsX());
    assert(tHistPlus->GetBinWidth(1) == tHistPlus->GetBinWidth(1));
    for(int j=1; j<=tHistMinus->GetNbinsX(); j++) assert(tHistMinus->GetXaxis()->GetName()==tHistPlus->GetXaxis()->GetName());

    tHistCombined = (TH1F*)tHistMinus->Clone();
    tHistCombined->Add(tHistPlus);
    tReturnList->Add(tHistCombined);
  }

  return tReturnList;
}


//________________________________________________________________________________________________________________
TH1F* GetHist(TList* aHistList, TString aHistBaseName="fMCTruthOfOriginalParticles", TString aV0Name="Lam")
{
  TH1F* tReturnHist = nullptr;
  TString tHistName = TString::Format("%s_%s", aHistBaseName.Data(), aV0Name.Data());
  tReturnHist = (TH1F*)aHistList->FindObject(tHistName)->Clone();
  assert(tReturnHist);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
void SetAttributes(TH1F* aHist, int aColor, int aMarkerStyle)
{
  aHist->SetMarkerStyle(aMarkerStyle);
  aHist->SetMarkerSize(1.0);
  aHist->SetMarkerColor(aColor);
  aHist->SetLineColor(aColor);
}

//________________________________________________________________________________________________________________
void SetAllAttributesInList(TList* aHistList)
{
  int tColor=1, tMarkerStyle=20;
/*
  for(int i=0; i<aHistList->GetEntries(); i++)
  {
    if     (TString(aHistList->At(i)->GetName()).Contains("PurityAid")) continue;
    else if(TString(aHistList->At(i)->GetName()).Contains("OriginalParticles"))      tColor=kBlack;
    else if(TString(aHistList->At(i)->GetName()).Contains("V0FinderParticles"))      tColor=kRed;
    else if(TString(aHistList->At(i)->GetName()).Contains("ReconstructedParticles")) tColor=kBlue;
    else assert(0);
    //----------
    if     (TString(aHistList->At(i)->GetName()).Contains("PurityAid")) continue;
    else if(TString(aHistList->At(i)->GetName()).Contains("ALam")) tMarkerStyle = 22;
    else if(TString(aHistList->At(i)->GetName()).Contains("Lam"))  tMarkerStyle = 20;
    else if(TString(aHistList->At(i)->GetName()).Contains("K0s"))  tMarkerStyle = 21;
    else assert(0);
    //----------
    SetAttributes((TH1F*)aHistList->At(i), tColor, tMarkerStyle);
  }
*/
  for(int i=0; i<aHistList->GetEntries(); i++)
  {
    if     (TString(aHistList->At(i)->GetName()).Contains("PurityAid")) continue;
    else if(TString(aHistList->At(i)->GetName()).Contains("OriginalParticles"))      tMarkerStyle = 20;
    else if(TString(aHistList->At(i)->GetName()).Contains("V0FinderParticles"))      tMarkerStyle = 21;
    else if(TString(aHistList->At(i)->GetName()).Contains("ReconstructedParticles")) tMarkerStyle = 22;
    else assert(0);
    //----------
    if     (TString(aHistList->At(i)->GetName()).Contains("PurityAid")) continue;
    else if(TString(aHistList->At(i)->GetName()).Contains("ALam")) tColor=kRed;
    else if(TString(aHistList->At(i)->GetName()).Contains("Lam"))  tColor=kBlack;
    else if(TString(aHistList->At(i)->GetName()).Contains("K0s"))  tColor=kBlue;
    else assert(0);
    //----------
    SetAttributes((TH1F*)aHistList->At(i), tColor, tMarkerStyle);
  }

}



//________________________________________________________________________________________________________________
double GetRecoEff(TList* aHistList, TString aV0Name="Lam", TString aMotherName="Primary")
{
  TH1F* tHistOG = GetHist(aHistList, "fMCTruthOfOriginalParticles", aV0Name);
  TH1F* tHistPost = GetHist(aHistList, "fMCTruthOfReconstructedParticles", aV0Name);
  assert(tHistOG->GetNbinsX() == tHistPost->GetNbinsX());

  double tCountsOG=-1., tCountsPost=-1.;
  for(int i=1; i<=tHistOG->GetNbinsX(); i++)
  {
    if(aMotherName.EqualTo(tHistOG->GetXaxis()->GetBinLabel(i)))
    {
      assert(aMotherName.EqualTo(tHistPost->GetXaxis()->GetBinLabel(i)));
      tCountsOG = tHistOG->GetBinContent(i);
      tCountsPost = tHistPost->GetBinContent(i);
    }
  }
  double tRecoEff=0.;
  if(tCountsOG==0. && tCountsPost != 0. && !aMotherName.EqualTo("Fake"))
  {
    cout << "tCountsOG==0. && tCountsPost != 0. && !aMotherName.EqualTo(Fake)" << endl << "WEIRD....." << endl;
    cout << "tHistOG->GetName() = " << tHistOG->GetName() << endl;
    cout << "tHistPost->GetName() = " << tHistPost->GetName() << endl;
    cout << "aV0Name = " << aV0Name << endl;
    cout << "aMotherName = " << aMotherName << endl;
    assert(0);
  }
  if(tCountsOG==0. || tCountsPost == 0.) return tRecoEff;

  tRecoEff = tCountsPost/tCountsOG;
  return tRecoEff;
}

//________________________________________________________________________________________________________________
vector<double> GetAllRecoEff(TList* aHistList, TString aV0Name="Lam")
{
  TString tBinName="";
  double tRecoEff=-1.;
  vector<double> tReturnVec(0);
  for(int i=1; i<=12; i++)
  {
    tBinName = GetBinLabels(i);
    tRecoEff = GetRecoEff(aHistList, aV0Name, tBinName);
//    cout << tBinName << " : " << tRecoEff << endl;
    tReturnVec.push_back(tRecoEff);
  }
  return tReturnVec;
}


//________________________________________________________________________________________________________________
void DrawMCTruths(TPad* aPad, TList* aHistList, bool aDrawV0Finder=false, bool aDrawK0s=false, TString aTitle="")
{
  aPad->cd();
  aPad->SetLogy();

  gStyle->SetOptStat(0);
//  gStyle->SetOptTitle(0);

  SetAllAttributesInList(aHistList);
  ((TH1F*)aHistList->At(0))->Draw("AXIS");
  //---------------------
  TString tBaseNameOG = "fMCTruthOfOriginalParticles";
  TString tBaseNameV0 = "fMCTruthOfV0FinderParticles";
  TString tBaseNamePost = "fMCTruthOfReconstructedParticles";

  TH1F* tHistOG_Lam = GetHist(aHistList, tBaseNameOG, "Lam");
  TH1F* tHistOG_ALam = GetHist(aHistList, tBaseNameOG, "ALam");
  TH1F* tHistOG_K0s = GetHist(aHistList, tBaseNameOG, "K0s");

  TH1F* tHistV0_Lam = GetHist(aHistList, tBaseNameV0, "Lam");
  TH1F* tHistV0_ALam = GetHist(aHistList, tBaseNameV0, "ALam");
  TH1F* tHistV0_K0s = GetHist(aHistList, tBaseNameV0, "K0s");

  TH1F* tHistPost_Lam = GetHist(aHistList, tBaseNamePost, "Lam");
  TH1F* tHistPost_ALam = GetHist(aHistList, tBaseNamePost, "ALam");
  TH1F* tHistPost_K0s = GetHist(aHistList, tBaseNamePost, "K0s");
  //---------------------

  tHistOG_Lam->SetTitle(TString::Format("%s_MCTruths", aTitle.Data()));
  tHistOG_Lam->GetYaxis()->SetRangeUser(100, 1000000000);
  tHistOG_Lam->GetXaxis()->SetLabelSize(0.05);

  //-----Lam
  tHistOG_Lam->Draw("p");
  if(aDrawV0Finder) tHistV0_Lam->Draw("psame");
  tHistPost_Lam->Draw("psame");

  //-----ALam
  tHistOG_ALam->Draw("psame");
  if(aDrawV0Finder) tHistV0_ALam->Draw("psame");
  tHistPost_ALam->Draw("psame");

  //-----K0s
  if(aDrawK0s)
  {
    tHistOG_K0s->Draw("psame");
    if(aDrawV0Finder) tHistV0_K0s->Draw("psame");
    tHistPost_K0s->Draw("psame");
  }

  //----------------------------------
  TLegend* tLeg = new TLegend(0.15, 0.70, 0.55, 0.89, "", "NDC");
  tLeg->SetNColumns(2);

  tLeg->AddEntry((TObject*)0, "Rec. Stage", "");
  tLeg->AddEntry((TObject*)0, "Particle Type", "");

  tLeg->AddEntry((TObject*)0, "", "");
  tLeg->AddEntry((TObject*)0, "", "");

  tLeg->AddEntry(tHistOG_Lam, "Pre. Reco", "p");
  tLeg->AddEntry(tHistOG_Lam, "#Lambda", "p");

  if(!aDrawV0Finder && !aDrawK0s)
  {
    tLeg->AddEntry(tHistPost_Lam, "Post Reco.", "p");
    tLeg->AddEntry(tHistOG_ALam, "#bar{#Lambda}", "p");
  }
  else
  {
    if(aDrawV0Finder) tLeg->AddEntry(tHistV0_Lam, "V0Finder", "p");
    else tLeg->AddEntry((TObject*)0, "", "");
    tLeg->AddEntry(tHistOG_ALam, "#bar{#Lambda}", "p");

    tLeg->AddEntry(tHistPost_Lam, "Post Reco.", "p");
    if(aDrawK0s) tLeg->AddEntry(tHistOG_K0s, "K^{0}_{S}", "p");
  }

  tLeg->Draw();
  //----------------------------------
}


//________________________________________________________________________________________________________________
void DrawRecoEffs(TPad* aPad, TList* aHistList, bool aDrawK0s=false, bool aDrawLegend=true, TString aTitle="")
{
  aPad->cd();

  gStyle->SetOptStat(0);
//  gStyle->SetOptTitle(0);
  gStyle->SetPaintTextFormat("0.3f");  //For precision of values printed with ->Draw("TEXT")

  //---------------------------
 
  vector<double> tAllReco_Lam = GetAllRecoEff(aHistList, "Lam");
  vector<double> tAllReco_ALam = GetAllRecoEff(aHistList, "ALam");
  vector<double> tAllReco_K0s = GetAllRecoEff(aHistList, "K0s");

  assert(tAllReco_Lam.size() == tAllReco_ALam.size());
  assert(tAllReco_Lam.size() == tAllReco_K0s.size());
  //---------------------------
  TH1F* tRecoHist_Lam = (TH1F*)aHistList->At(0)->Clone("tRecoHist_Lam");
  TH1F* tRecoHist_ALam = (TH1F*)aHistList->At(0)->Clone("tRecoHist_ALam");
  TH1F* tRecoHist_K0s = (TH1F*)aHistList->At(0)->Clone("tRecoHist_K0s");

  SetAttributes(tRecoHist_Lam, kBlack, 20);
  SetAttributes(tRecoHist_ALam, kRed, 20);
  SetAttributes(tRecoHist_K0s, kBlue, 20);

  //---------------------------
  for(int i=1; i<=tRecoHist_Lam->GetNbinsX(); i++)
  {
    tRecoHist_Lam->SetBinContent(i, tAllReco_Lam[i-1]);
    tRecoHist_Lam->SetBinError(i, 0.);

    tRecoHist_ALam->SetBinContent(i, tAllReco_ALam[i-1]);
    tRecoHist_ALam->SetBinError(i, 0.);

    tRecoHist_K0s->SetBinContent(i, tAllReco_K0s[i-1]);
    tRecoHist_K0s->SetBinError(i, 0.);
  }
  //---------------------------
  tRecoHist_Lam->SetTitle(TString::Format("%s_RecoEffs", aTitle.Data()));
  tRecoHist_Lam->GetYaxis()->SetRangeUser(0.0, 0.3);
  tRecoHist_Lam->GetXaxis()->SetLabelSize(0.05);

  tRecoHist_Lam->Draw("HTEXT25");
  tRecoHist_ALam->Draw("HTEXT25same");
  if(aDrawK0s) tRecoHist_K0s->Draw("TEXT25Hsame");

  //----------------------------------
  if(aDrawLegend)
  {
    TLegend* tLeg = new TLegend(0.15, 0.70, 0.55, 0.89, "", "NDC");
    tLeg->AddEntry(tRecoHist_Lam, "#Lambda", "l");
    tLeg->AddEntry(tRecoHist_ALam, "#bar{#Lambda}", "l");
    if(aDrawK0s) tLeg->AddEntry(tRecoHist_K0s, "K^{0}_{S}", "l");

    tLeg->Draw();
  }
}


//________________________________________________________________________________________________________________
TCanvas* DrawMCTruthsAndRecoEffs(TList* aHistList, bool aDrawV0Finder=false, bool aDrawK0s=false, bool aVertical=true, TString aTitle="")
{
  TString tCanName = TString::Format("tCanTruthsAndEffs_%s", aTitle.Data());
  TCanvas* tCan;
  TPad *tPad1, *tPad2;

  if(aVertical)
  {
    tCan = new TCanvas(tCanName, tCanName, 700, 1000);
    tCan->cd();

    tPad1 = new TPad("tPad1","tPad1",0.0,0.5,1.0,1.0);
    tPad1->Draw();
    tPad2 = new TPad("tPad2","tPad2",0.0,0.0,1.0,0.5);
    tPad2->Draw();
  }
  else
  {
    tCan = new TCanvas(tCanName, tCanName, 1400, 500);
    tCan->cd();

    tPad1 = new TPad("tPad1","tPad1",0.0,0.0,0.5,1.0);
    tPad1->Draw();
    tPad2 = new TPad("tPad2","tPad2",0.5,0.0,1.0,1.0);
    tPad2->Draw();
  }

  //-----------------------------

  DrawMCTruths(tPad1, aHistList, aDrawV0Finder, aDrawK0s, aTitle);
  DrawRecoEffs(tPad2, aHistList, aDrawK0s, true, aTitle);

  return tCan;
}


//________________________________________________________________________________________________________________
TCanvas* SetupAndDraw(TString aBaseDirLocation, TString aResultDate, bool aWithInjected, bool a12a17a, 
                      bool aDrawV0Finder, bool aDrawK0s, bool aVertical,
                      bool aSave=false, TString aSaveDir="/home/jesse/")
{
  vector<TString> tInjTags = {"woInjected", "wInjected"};
  vector<TString> tDatasetTags = {"_Lhc12a17d", ""};

  TString tTitle = TString::Format("%s%s", tInjTags[aWithInjected].Data(), tDatasetTags[a12a17a].Data());

  TString tFileLocationMinus = TString::Format("%s%s/AnalysisResults_FemtoMinus_%s%s.root", aBaseDirLocation.Data(), aResultDate.Data(), tInjTags[aWithInjected].Data(), tDatasetTags[a12a17a].Data());
  TString tFileLocationPlus = TString::Format("%s%s/AnalysisResults_FemtoPlus_%s%s.root", aBaseDirLocation.Data(), aResultDate.Data(), tInjTags[aWithInjected].Data(), tDatasetTags[a12a17a].Data());

  TList* tHistsMinus = GetAllHistograms(tFileLocationMinus);
  TList* tHistsPlus = GetAllHistograms(tFileLocationPlus);

  TList* tHistsCombined = SimpleCombineBFields(tHistsMinus, tHistsPlus);

//-----------------------------------------------------------------------------

  TCanvas* tCan = DrawMCTruthsAndRecoEffs(tHistsCombined, aDrawV0Finder, aDrawK0s, aVertical, tTitle);

  if(aSave)
  {
    tCan->SaveAs(TString::Format("%s%s.pdf", aSaveDir.Data(), tCan->GetName()));
  }

  return tCan;
}


//________________________________________________________________________________________________________________
void SetupAndDrawAll(TString aBaseDirLocation, TString aResultDate, 
                         bool aDrawV0Finder, bool aDrawK0s, bool aVertical,
                         bool aSave=false, TString aSaveDir="/home/jesse/")
{
  //LHC12a17a with injected
  TCanvas* tCan1 = SetupAndDraw(aBaseDirLocation, aResultDate, true, true, 
                                aDrawV0Finder, aDrawK0s, aVertical, 
                                aSave, aSaveDir);

  //LHC12a17a without injected
  TCanvas* tCan2 = SetupAndDraw(aBaseDirLocation, aResultDate, false, true, 
                                aDrawV0Finder, aDrawK0s, aVertical, 
                                aSave, aSaveDir);

  //LHC12a17d with injected
  TCanvas* tCan3 = SetupAndDraw(aBaseDirLocation, aResultDate, true, false, 
                                aDrawV0Finder, aDrawK0s, aVertical, 
                                aSave, aSaveDir);

  //LHC12a17d without injected
  TCanvas* tCan4 = SetupAndDraw(aBaseDirLocation, aResultDate, false, false, 
                                aDrawV0Finder, aDrawK0s, aVertical, 
                                aSave, aSaveDir);

}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int DrawEfficiencyPlots() 
{
//-----------------------------------------------------------------------------
  
  TString tBaseDirLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/V0Efficiency/";

  TString tResultDate = "20181016";
  bool tWithInjected = true;
  bool t12a17a = true;

  bool tDrawV0Finder=false;
  bool tDrawK0s=false;
  bool tVertical=false;

  bool tDrawAll=false;

  bool tSave = false;
  TString tSaveDir = TString::Format("%s%s/", tBaseDirLocation.Data(), tResultDate.Data());

//-----------------------------------------------------------------------------
  if(tDrawAll)
  {
    SetupAndDrawAll(tBaseDirLocation, tResultDate, 
                    tDrawV0Finder, tDrawK0s, tVertical, 
                    tSave, tSaveDir);
    return 0;
  }


  vector<TString> tInjTags = {"woInjected", "wInjected"};
  vector<TString> tDatasetTags = {"_Lhc12a17d", ""};

  TString tTitle = TString::Format("%s%s", tInjTags[tWithInjected].Data(), tDatasetTags[t12a17a].Data());

  TString tFileLocationMinus = TString::Format("%s%s/AnalysisResults_FemtoMinus_%s%s.root", tBaseDirLocation.Data(), tResultDate.Data(), tInjTags[tWithInjected].Data(), tDatasetTags[t12a17a].Data());
  TString tFileLocationPlus = TString::Format("%s%s/AnalysisResults_FemtoPlus_%s%s.root", tBaseDirLocation.Data(), tResultDate.Data(), tInjTags[tWithInjected].Data(), tDatasetTags[t12a17a].Data());

  TList* tHistsMinus = GetAllHistograms(tFileLocationMinus);
  TList* tHistsPlus = GetAllHistograms(tFileLocationPlus);

  TList* tHistsCombined = SimpleCombineBFields(tHistsMinus, tHistsPlus);

//-----------------------------------------------------------------------------

  TCanvas* tTestCan3 = SetupAndDraw(tBaseDirLocation, tResultDate, tWithInjected, t12a17a, 
                                    tDrawV0Finder, tDrawK0s, tVertical, 
                                    tSave, tSaveDir);






//-----------------------------------------------------------------------------
//TODO TODO CLEANUP
/*

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tList object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tList);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  tList->Delete();
  delete tList;

*/



//-----------------------------------------------------------------------------
cout << "DONE" << endl;
  return 0;
}
