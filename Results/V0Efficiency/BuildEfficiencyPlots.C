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
#include "TDatabasePDG.h"
#include "TParticlePDG.h"

#include "PIDMapping.h"
#include "ThermCommon.h"

using namespace std; 

//________________________________________________________________________________________________________________
TString GetBinLabelsv2(int aBin)
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
  else if(aBin==11) return TString("Other(1)");
  else if(aBin==12) return TString("Other(2)");
  else if(aBin==13) return TString("Other(1+2)");
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
  for(int i=0; i<aListMinus->GetEntries(); i++) assert(TString(aListMinus->At(i)->GetName()).EqualTo(aListPlus->At(i)->GetName()));

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
    for(int j=1; j<=tHistMinus->GetNbinsX(); j++) assert(TString(tHistMinus->GetXaxis()->GetName()).EqualTo(tHistPlus->GetXaxis()->GetName()));

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
  for(int i=1; i<=13; i++)
  {
    tBinName = GetBinLabelsv2(i);
    tRecoEff = GetRecoEff(aHistList, aV0Name, tBinName);
//    cout << tBinName << " : " << tRecoEff << endl;
    tReturnVec.push_back(tRecoEff);
  }
  return tReturnVec;
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
bool IsKnownAdditionalParticle(int aPID)
{
  //  These are not observed in THERMINATOR, but are in HIJING apparently
  //  3112	Sigma-
  //  4122	Lambda_c+
  //  4132	Xi_c0
  //  4232	Xi_c+
  //  5122	Lambda_b0
  //  5132	Xi_b-
  //  5232	Xi_b0

  vector<PidInfo> cAdditionalPidInfo {
    PidInfo(3112, "#Sigma^{-}", 4.434e13),
    PidInfo(-3112, "#bar{#Sigma}^{+}", 4.434e13),
	
    PidInfo(4122, "#Lambda_{c}^{+}", 5.99e10),
    PidInfo(-4122, "#bar{#Lambda}_{c}^{-}", 5.99e10),

    PidInfo(4132, "#Xi_{c}^{0}", 3.36e10),
    PidInfo(-4132, "#bar{#Xi}_{c}^{0}", 3.36e10),

    PidInfo(4232, "#Xi_{c}^{+}", 1.32e11),
    PidInfo(-4232, "#bar{#Xi}_{c}^{-}", 1.32e11),

    PidInfo(5122, "#Lambda_{b}^{0}", 4.27e11),
    PidInfo(-5122, "#bar{#Lambda}_{b}^{0}", 4.27e11),

    PidInfo(5132, "#Xi_{b}^{-}", 4.68e11),
    PidInfo(-5132, "#bar{#Xi}_{b}^{+}", 4.68e11),

    PidInfo(5232, "#Xi_{b}^{0}", 4.47e11),
    PidInfo(-5232, "#bar{#Xi}_{b}^{0}", 4.47e11)
  };


  //---------------------------------------------
  for(int i=0; i<cAdditionalPidInfo.size(); i++)
  {
    if(aPID == cAdditionalPidInfo[i].pdgType) return true;
  }
  return false;
}





//________________________________________________________________________________________________________________
bool IsInThermPidInfo(int aPID)
{
  for(unsigned int i=0; i<cPidInfo.size(); i++)
  {
    if(aPID==cPidInfo[i].pdgType) return true;
  }
  assert(IsKnownAdditionalParticle(aPID));
  return false;
}


//________________________________________________________________________________________________________________
int GetMCTruthBinFromParticleOriginBin(int aParticleOriginBin, double aMaxDecayLength, IncludeResidualsType aIncResType)
{
  //TODO only works for (A)Lam!
  int tMotherPDG = aParticleOriginBin-1;
  TParticlePDG* tMother = TDatabasePDG::Instance()->GetParticle(tMotherPDG);

  if(tMotherPDG==0) return 1;
  else if(!TString(tMother->ParticleClass()).Contains("Baryon")) return 0;
  else if(tMother->Mass() < 1.115683) return 0;
  else if(!IsInThermPidInfo(tMotherPDG)) return 11;  //TODO what to do with these?  See IsKnownAdditionalParticle for list of these particles
  else if(tMotherPDG==3212) return 2;
  else if(tMotherPDG==3322) return 3;
  else if(tMotherPDG==3312) return 4;
  else if(aIncResType==kInclude10Residuals && tMotherPDG==3214) return 5;
  else if(aIncResType==kInclude10Residuals && tMotherPDG==3224) return 6;
  else if(aIncResType==kInclude10Residuals && tMotherPDG==3114) return 7;
  else if(aIncResType==kInclude10Residuals && tMotherPDG==313) return 8;
  else if(aIncResType==kInclude10Residuals && tMotherPDG==323) return 9;
  else if(IncludeAsPrimary(tMotherPDG, kPDGKchP, aMaxDecayLength)) return 1;
  else if(IncludeInOthers(tMotherPDG, kPDGKchP, aMaxDecayLength, aIncResType)) {cout << "tMotherPDG = " << tMotherPDG << endl; return 10;} // 10=Other(1); 11=Other(2); 12=Other(1+2)
  else
  {
    cout << "DO NOT RECOGNIZE: tMotherPDG = " << tMotherPDG << endl;
    return -1;
  }

  return -1;
}

//________________________________________________________________________________________________________________
TH1F* BuildMCTruthFromParticleOrigin(TH1F* aParticleOriginHist, double aMaxDecayLength, IncludeResidualsType aIncResType)
{
  TString tReturnBaseName;
  if     (TString(aParticleOriginHist->GetName()).Contains("fParticleOriginOfOriginalParticles")) tReturnBaseName = TString("fMCTruthOfOriginalParticles");
  else if(TString(aParticleOriginHist->GetName()).Contains("fParticleOriginOfV0FinderParticles")) tReturnBaseName = TString("fMCTruthOfV0FinderParticles");
  else if(TString(aParticleOriginHist->GetName()).Contains("fParticleOriginOfReconstructedParticles")) tReturnBaseName = TString("fMCTruthOfReconstructedParticles");
  else assert(0);

  TString tPartName;
  if     (TString(aParticleOriginHist->GetName()).Contains("ALam")) tPartName = TString("_ALam");
  else if(TString(aParticleOriginHist->GetName()).Contains("Lam")) tPartName = TString("_Lam");
  else if(TString(aParticleOriginHist->GetName()).Contains("K0s")) tPartName = TString("_K0s");
  else assert(0);

  TString tReturnName = tReturnBaseName + tPartName;

  //-----------------------------------------------------

  TH1F* tReturnHist = new TH1F(tReturnName, tReturnName, 13, 0, 13);


  for(int i=1; i<=tReturnHist->GetNbinsX(); i++) tReturnHist->GetXaxis()->SetBinLabel(i, GetBinLabelsv2(i));
  vector<double> tTempCount(tReturnHist->GetNbinsX());

  for(int i=1; i<=aParticleOriginHist->GetNbinsX(); i++)
  {
    if(aParticleOriginHist->GetBinContent(i)>0) 
    {
      int tBin = GetMCTruthBinFromParticleOriginBin(i, aMaxDecayLength, aIncResType);
//      cout << "GetMCTruthBinFromParticleOriginBin(" << i << ", " << aMaxDecayLength << ", " <<  aIncResType << ") = ";
//      cout << GetMCTruthBinFromParticleOriginBin(i, aMaxDecayLength, aIncResType) << endl;
      if(tBin > -1) 
      {
        tTempCount[tBin] += aParticleOriginHist->GetBinContent(i);
        if(tBin==10 || tBin==11) tTempCount[12] += aParticleOriginHist->GetBinContent(i);
      }
    }
  }

  for(int i=0; i<tTempCount.size(); i++) tReturnHist->SetBinContent(i+1, tTempCount[i]);

  return tReturnHist;
}


//________________________________________________________________________________________________________________
TList* BuildNewTListFromParticleOrigins(TList* aList, double aMaxDecayLength, IncludeResidualsType aIncResType)
{
  TList* tReturnList = new TList();

  vector<TString> tBaseNameVec{"fParticleOriginOfOriginalParticles", "fParticleOriginOfV0FinderParticles", "fParticleOriginOfReconstructedParticles"};
  vector<TString> tPartNameVec{"Lam", "ALam", "K0s"};

  TH1F* tTempPOHist = nullptr;
  TH1F* tTempMCTruthHist = nullptr;
  for(int iBaseName=0; iBaseName<tBaseNameVec.size(); iBaseName++)
  {
    for(int iPartName=0.; iPartName<tPartNameVec.size(); iPartName++)
    {
      tTempPOHist = GetHist(aList, tBaseNameVec[iBaseName], tPartNameVec[iPartName]);
      tTempMCTruthHist = BuildMCTruthFromParticleOrigin(tTempPOHist, aMaxDecayLength, aIncResType);
      tReturnList->Add((TH1F*)tTempMCTruthHist->Clone());
    }
  }

  return tReturnList;
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
  
  TString tBaseDirLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/V0Efficiency/";

  TString tResultDate = "20181024";
  bool tWithInjected = true;
  bool t12a17a = true;

  bool tDrawV0Finder=false;
  bool tDrawK0s=false;
  bool tVertical=false;


  double aMaxDecayLength = 10.;
  IncludeResidualsType aIncResType = kInclude3Residuals;



  bool tSave = false;
  TString tSaveDir = TString::Format("%s%s/", tBaseDirLocation.Data(), tResultDate.Data());

//-----------------------------------------------------------------------------

  vector<TString> tInjTags = {"woInjected", "wInjected"};
  vector<TString> tDatasetTags = {"_Lhc12a17d", ""};

  TString tTitle = TString::Format("%s%s_Built%s_MaxDecay%0.1f", tInjTags[tWithInjected].Data(), tDatasetTags[t12a17a].Data(), cIncludeResidualsTypeTags[aIncResType], aMaxDecayLength);

  TString tFileLocationMinus = TString::Format("%s%s/AnalysisResults_FemtoMinus_%s%s.root", tBaseDirLocation.Data(), tResultDate.Data(), tInjTags[tWithInjected].Data(), tDatasetTags[t12a17a].Data());
  TString tFileLocationPlus = TString::Format("%s%s/AnalysisResults_FemtoPlus_%s%s.root", tBaseDirLocation.Data(), tResultDate.Data(), tInjTags[tWithInjected].Data(), tDatasetTags[t12a17a].Data());

  TList* tHistsMinus = GetAllHistograms(tFileLocationMinus);
  TList* tHistsPlus = GetAllHistograms(tFileLocationPlus);

  TList* tHistsCombined = SimpleCombineBFields(tHistsMinus, tHistsPlus);

//-----------------------------------------------------------------------------
/*
TH1F* tTestHist = GetHist(tHistsCombined, "fParticleOriginOfReconstructedParticles", "Lam");
TH1F* tTestMCTruth = BuildMCTruthFromParticleOrigin(tTestHist, 10., kInclude3Residuals);


TCanvas* tCan = new TCanvas("tCan", "tCan");
tCan->cd();

tTestMCTruth->Draw();
*/



  TList* tNewList = BuildNewTListFromParticleOrigins(tHistsCombined, aMaxDecayLength, aIncResType);



  TCanvas* tCan = DrawMCTruthsAndRecoEffs(tNewList, false, false, false, tTitle);
  if(tSave)
  {
    tCan->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCan->GetName()));
  }



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
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
