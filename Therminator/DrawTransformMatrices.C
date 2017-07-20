#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"


//________________________________________________________________________________________________________________
TH2D* Get2dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH2D* ReturnHisto = (TH2D*)f1.Get(HistoName);
  TH2D *ReturnHistoClone = (TH2D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}


//________________________________________________________________________________________________________________
TCanvas* DrawTransform(TString tFileName, TString tHistoName, ParticlePDGType aMotherType1, ParticlePDGType aDaughterType1, ParticlePDGType aMotherType2, ParticlePDGType aDaughterType2, bool aDrawLogZ=false, bool bSaveFigures=false, TString aSaveLocationBase="")
{
  TH2D* tMatrix = Get2dHisto(tFileName,tHistoName);

  TString tMotherName1 = TString(GetPDGRootName(aMotherType1));
  TString tDaughterName1 = TString(GetPDGRootName(aDaughterType1));

  TString tMotherName2 = TString(GetPDGRootName(aMotherType2));
  TString tDaughterName2 = TString(GetPDGRootName(aDaughterType2));

  TString tMatrixTitle = tMotherName1 + tMotherName2 + TString(" To ") + 
                         tDaughterName1 + tDaughterName2 + TString(" TransformMatrix ");
  tMatrix->SetTitle(tMatrixTitle);

  TString tCanvasName = TString("can") + tMatrix->GetName();
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  if(aDrawLogZ) tReturnCan->SetLogz();


  TString tXName = TString("k*_{") + tDaughterName1 + tDaughterName2 + TString("}(GeV/c)");
  TString tYName = TString("k*_{") + tMotherName1 + tMotherName2 + TString("}(GeV/c)");


  tMatrix->GetXaxis()->SetTitle(tXName);
    tMatrix->GetXaxis()->SetTitleSize(0.04);
    tMatrix->GetXaxis()->SetTitleOffset(1.1);

  tMatrix->GetYaxis()->SetTitle(tYName);
    tMatrix->GetYaxis()->SetTitleSize(0.04);
    tMatrix->GetYaxis()->SetTitleOffset(1.2);

  tMatrix->GetZaxis()->SetLabelSize(0.02);
  tMatrix->GetZaxis()->SetLabelOffset(0.004);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  tReturnCan->cd();
  tMatrix->Draw("colz");


  //-------------------------
  TString tBoxText = tMotherName1 + tMotherName2 + TString(" To ") + 
                     tDaughterName1 + tDaughterName2;
  double tTextXmin = 0.15;
  double tTextXmax = 0.35;
  double tTextYmin = 0.75;
  double tTextYmax = 0.85;
  TPaveText* tText = new TPaveText(tTextXmin,tTextYmin,tTextXmax,tTextYmax,"NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->AddText(tBoxText);
  tText->Draw();

  if(bSaveFigures) tReturnCan->SaveAs(aSaveLocationBase+TString(tMatrix->GetName())+TString(".pdf"));

  return tReturnCan;
}


//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  bool bDrawLogZ = false;
  bool bSaveFigures = true;

  TString tFileName = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/TransformMatrices.root";
  TString tSaveLocationBase = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/Figures/";

  TString tSaveLocationBaseLamKchP = tSaveLocationBase + TString("LamKchP/");
  TString tSaveLocationBaseALamKchP = tSaveLocationBase + TString("ALamKchP/");
  TString tSaveLocationBaseLamKchM = tSaveLocationBase + TString("LamKchM/");
  TString tSaveLocationBaseALamKchM = tSaveLocationBase + TString("ALamKchM/");

  //LamKchP
  TString tSigToLamKchPName = "fSigToLamKchPTransform";
  TString tXiCToLamKchPName = "fXiCToLamKchPTransform";
  TString tXi0ToLamKchPName = "fXi0ToLamKchPTransform";
  TString tOmegaToLamKchPName = "fOmegaToLamKchPTransform";
  TString tSigStPToLamKchPName = "fSigStPToLamKchPTransform";
  TString tSigStMToLamKchPName = "fSigStMToLamKchPTransform";
  TString tSigSt0ToLamKchPName = "fSigSt0ToLamKchPTransform";
  TString tLamKSt0ToLamKchPName = "fLamKSt0ToLamKchPTransform";
  TString tSigKSt0ToLamKchPName = "fSigKSt0ToLamKchPTransform";
  TString tXiCKSt0ToLamKchPName = "fXiCKSt0ToLamKchPTransform";
  TString tXi0KSt0ToLamKchPName = "fXi0KSt0ToLamKchPTransform";

  //ALamKchP
  TString tASigToALamKchPName = "fASigToALamKchPTransform";
  TString tAXiCToALamKchPName = "fAXiCToALamKchPTransform";
  TString tAXi0ToALamKchPName = "fAXi0ToALamKchPTransform";
  TString tAOmegaToALamKchPName = "fAOmegaToALamKchPTransform";
  TString tASigStMToALamKchPName = "fASigStMToALamKchPTransform";
  TString tASigStPToALamKchPName = "fASigStPToALamKchPTransform";
  TString tASigSt0ToALamKchPName = "fASigSt0ToALamKchPTransform";
  TString tALamKSt0ToALamKchPName = "fALamKSt0ToALamKchPTransform";
  TString tASigKSt0ToALamKchPName = "fASigKSt0ToALamKchPTransform";
  TString tAXiCKSt0ToALamKchPName = "fAXiCKSt0ToALamKchPTransform";
  TString tAXi0KSt0ToALamKchPName = "fAXi0KSt0ToALamKchPTransform";

  //LamKchM
  TString tSigToLamKchMName = "fSigToLamKchMTransform";
  TString tXiCToLamKchMName = "fXiCToLamKchMTransform";
  TString tXi0ToLamKchMName = "fXi0ToLamKchMTransform";
  TString tOmegaToLamKchMName = "fOmegaToLamKchMTransform";
  TString tSigStPToLamKchMName = "fSigStPToLamKchMTransform";
  TString tSigStMToLamKchMName = "fSigStMToLamKchMTransform";
  TString tSigSt0ToLamKchMName = "fSigSt0ToLamKchMTransform";
  TString tLamAKSt0ToLamKchMName = "fLamAKSt0ToLamKchMTransform";
  TString tSigAKSt0ToLamKchMName = "fSigAKSt0ToLamKchMTransform";
  TString tXiCAKSt0ToLamKchMName = "fXiCAKSt0ToLamKchMTransform";
  TString tXi0AKSt0ToLamKchMName = "fXi0AKSt0ToLamKchMTransform";

  //ALamKchM
  TString tASigToALamKchMName = "fASigToALamKchMTransform";
  TString tAXiCToALamKchMName = "fAXiCToALamKchMTransform";
  TString tAXi0ToALamKchMName = "fAXi0ToALamKchMTransform";
  TString tAOmegaToALamKchMName = "fAOmegaToALamKchMTransform";
  TString tASigStMToALamKchMName = "fASigStMToALamKchMTransform";
  TString tASigStPToALamKchMName = "fASigStPToALamKchMTransform";
  TString tASigSt0ToALamKchMName = "fASigSt0ToALamKchMTransform";
  TString tALamAKSt0ToALamKchMName = "fALamAKSt0ToALamKchMTransform";
  TString tASigAKSt0ToALamKchMName = "fASigAKSt0ToALamKchMTransform";
  TString tAXiCAKSt0ToALamKchMName = "fAXiCAKSt0ToALamKchMTransform";
  TString tAXi0AKSt0ToALamKchMName = "fAXi0AKSt0ToALamKchMTransform";

  TString SigToLamName = "fSigToLamKchPTransform";

  TString XiCToLamName = "fXiCToLamKchPTransform";
  TString Xi0ToLamName = "fXi0ToLamKchPTransform";
  TString OmegaToLamName = "fOmegaToLamKchPTransform";
  TString ASigToALamName = "fASigToALamKchPTransform";
  TString AXiCToALamName = "fAXiCToALamKchPTransform";
  TString AXi0ToALamName = "fAXi0ToALamKchPTransform";
  TString AOmegaToALamName = "fAOmegaToALamKchPTransform";

  //-------------------------------------------------------------------------------------------------

  //LamKchP
  TCanvas* tCanSigToLamKchP = DrawTransform(tFileName, tSigToLamKchPName, kPDGSigma, kPDGLam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanXiCToLamKchP = DrawTransform(tFileName, tXiCToLamKchPName, kPDGXiC, kPDGLam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanXi0ToLamKchP = DrawTransform(tFileName, tXi0ToLamKchPName, kPDGXi0, kPDGLam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanOmegaToLamKchP = DrawTransform(tFileName, tOmegaToLamKchPName, kPDGOmega, kPDGLam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanSigStPToLamKchP = DrawTransform(tFileName, tSigStPToLamKchPName, kPDGSigStP, kPDGLam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanSigStMToLamKchP = DrawTransform(tFileName, tSigStMToLamKchPName, kPDGSigStM, kPDGLam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanSigSt0ToLamKchP = DrawTransform(tFileName, tSigSt0ToLamKchPName, kPDGSigSt0, kPDGLam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanLamKSt0ToLamKchP = DrawTransform(tFileName, tLamKSt0ToLamKchPName, kPDGLam, kPDGLam, kPDGKSt0, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanSigKSt0ToLamKchP = DrawTransform(tFileName, tSigKSt0ToLamKchPName, kPDGSigma, kPDGLam, kPDGKSt0, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanXiCKSt0ToLamKchP = DrawTransform(tFileName, tXiCKSt0ToLamKchPName, kPDGXiC, kPDGLam, kPDGKSt0, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);
  TCanvas* tCanXi0KSt0ToLamKchP = DrawTransform(tFileName, tXi0KSt0ToLamKchPName, kPDGXi0, kPDGLam, kPDGKSt0, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchP);

  //ALamKchP
  TCanvas* tCanASigToALamKchP = DrawTransform(tFileName, tASigToALamKchPName, kPDGASigma, kPDGALam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanAXiCToALamKchP = DrawTransform(tFileName, tAXiCToALamKchPName, kPDGAXiC, kPDGALam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanAXi0ToALamKchP = DrawTransform(tFileName, tAXi0ToALamKchPName, kPDGAXi0, kPDGALam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanAOmegaToALamKchP = DrawTransform(tFileName, tAOmegaToALamKchPName, kPDGAOmega, kPDGALam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanASigStMToALamKchP = DrawTransform(tFileName, tASigStMToALamKchPName, kPDGASigStM, kPDGALam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanASigStPToALamKchP = DrawTransform(tFileName, tASigStPToALamKchPName, kPDGASigStP, kPDGALam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanASigSt0ToALamKchP = DrawTransform(tFileName, tASigSt0ToALamKchPName, kPDGASigSt0, kPDGALam, kPDGKchP, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanALamKSt0ToALamKchP = DrawTransform(tFileName, tALamKSt0ToALamKchPName, kPDGALam, kPDGALam, kPDGKSt0, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanASigKSt0ToALamKchP = DrawTransform(tFileName, tASigKSt0ToALamKchPName, kPDGASigma, kPDGALam, kPDGKSt0, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanAXiCKSt0ToALamKchP = DrawTransform(tFileName, tAXiCKSt0ToALamKchPName, kPDGAXiC, kPDGALam, kPDGKSt0, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);
  TCanvas* tCanAXi0KSt0ToALamKchP = DrawTransform(tFileName, tAXi0KSt0ToALamKchPName, kPDGAXi0, kPDGALam, kPDGKSt0, kPDGKchP, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchP);


  //LamKchM
  TCanvas* tCanSigToLamKchM = DrawTransform(tFileName, tSigToLamKchMName, kPDGSigma, kPDGLam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanXiCToLamKchM = DrawTransform(tFileName, tXiCToLamKchMName, kPDGXiC, kPDGLam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanXi0ToLamKchM = DrawTransform(tFileName, tXi0ToLamKchMName, kPDGXi0, kPDGLam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanOmegaToLamKchM = DrawTransform(tFileName, tOmegaToLamKchMName, kPDGOmega, kPDGLam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanSigStPToLamKchM = DrawTransform(tFileName, tSigStPToLamKchMName, kPDGSigStP, kPDGLam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanSigStMToLamKchM = DrawTransform(tFileName, tSigStMToLamKchMName, kPDGSigStM, kPDGLam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanSigSt0ToLamKchM = DrawTransform(tFileName, tSigSt0ToLamKchMName, kPDGSigSt0, kPDGLam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanLamAKSt0ToLamKchM = DrawTransform(tFileName, tLamAKSt0ToLamKchMName, kPDGLam, kPDGLam, kPDGAKSt0, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanSigAKSt0ToLamKchM = DrawTransform(tFileName, tSigAKSt0ToLamKchMName, kPDGSigma, kPDGLam, kPDGAKSt0, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanXiCAKSt0ToLamKchM = DrawTransform(tFileName, tXiCAKSt0ToLamKchMName, kPDGXiC, kPDGLam, kPDGAKSt0, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);
  TCanvas* tCanXi0AKSt0ToLamKchM = DrawTransform(tFileName, tXi0AKSt0ToLamKchMName, kPDGXi0, kPDGLam, kPDGAKSt0, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseLamKchM);

  //ALamKchM
  TCanvas* tCanASigToALamKchM = DrawTransform(tFileName, tASigToALamKchMName, kPDGASigma, kPDGALam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanAXiCToALamKchM = DrawTransform(tFileName, tAXiCToALamKchMName, kPDGAXiC, kPDGALam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanAXi0ToALamKchM = DrawTransform(tFileName, tAXi0ToALamKchMName, kPDGAXi0, kPDGALam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanAOmegaToALamKchM = DrawTransform(tFileName, tAOmegaToALamKchMName, kPDGAOmega, kPDGALam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanASigStMToALamKchM = DrawTransform(tFileName, tASigStMToALamKchMName, kPDGASigStM, kPDGALam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanASigStPToALamKchM = DrawTransform(tFileName, tASigStPToALamKchMName, kPDGASigStP, kPDGALam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanASigSt0ToALamKchM = DrawTransform(tFileName, tASigSt0ToALamKchMName, kPDGASigSt0, kPDGALam, kPDGKchM, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanALamAKSt0ToALamKchM = DrawTransform(tFileName, tALamAKSt0ToALamKchMName, kPDGALam, kPDGALam, kPDGAKSt0, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanASigAKSt0ToALamKchM = DrawTransform(tFileName, tASigAKSt0ToALamKchMName, kPDGASigma, kPDGALam, kPDGAKSt0, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanAXiCAKSt0ToALamKchM = DrawTransform(tFileName, tAXiCAKSt0ToALamKchMName, kPDGAXiC, kPDGALam, kPDGAKSt0, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);
  TCanvas* tCanAXi0AKSt0ToALamKchM = DrawTransform(tFileName, tAXi0AKSt0ToALamKchMName, kPDGAXi0, kPDGALam, kPDGAKSt0, kPDGKchM, bDrawLogZ, bSaveFigures, tSaveLocationBaseALamKchM);


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.


  //LamKchP
  tCanSigToLamKchP->Close();
  tCanXiCToLamKchP->Close();
  tCanXi0ToLamKchP->Close();
  tCanOmegaToLamKchP->Close();
  tCanSigStPToLamKchP->Close();
  tCanSigStMToLamKchP->Close();
  tCanSigSt0ToLamKchP->Close();
  tCanLamKSt0ToLamKchP->Close();
  tCanSigKSt0ToLamKchP->Close();
  tCanXiCKSt0ToLamKchP->Close();
  tCanXi0KSt0ToLamKchP->Close();

  //ALamKchP
  tCanASigToALamKchP->Close();
  tCanAXiCToALamKchP->Close();
  tCanAXi0ToALamKchP->Close();
  tCanAOmegaToALamKchP->Close();
  tCanASigStMToALamKchP->Close();
  tCanASigStPToALamKchP->Close();
  tCanASigSt0ToALamKchP->Close();
  tCanALamKSt0ToALamKchP->Close();
  tCanASigKSt0ToALamKchP->Close();
  tCanAXiCKSt0ToALamKchP->Close();
  tCanAXi0KSt0ToALamKchP->Close();

  //LamKchM
  tCanSigToLamKchM->Close();
  tCanXiCToLamKchM->Close();
  tCanXi0ToLamKchM->Close();
  tCanOmegaToLamKchM->Close();
  tCanSigStPToLamKchM->Close();
  tCanSigStMToLamKchM->Close();
  tCanSigSt0ToLamKchM->Close();
  tCanLamAKSt0ToLamKchM->Close();
  tCanSigAKSt0ToLamKchM->Close();
  tCanXiCAKSt0ToLamKchM->Close();
  tCanXi0AKSt0ToLamKchM->Close();

  //ALamKchM
  tCanASigToALamKchM->Close();
  tCanAXiCToALamKchM->Close();
  tCanAXi0ToALamKchM->Close();
  tCanAOmegaToALamKchM->Close();
  tCanASigStMToALamKchM->Close();
  tCanASigStPToALamKchM->Close();
  tCanASigSt0ToALamKchM->Close();
  tCanALamAKSt0ToALamKchM->Close();
  tCanASigAKSt0ToALamKchM->Close();
  tCanAXiCAKSt0ToALamKchM->Close();
  tCanAXi0AKSt0ToALamKchM->Close();

  return 0;
}
