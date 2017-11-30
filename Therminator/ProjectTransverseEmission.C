#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TColor.h"
#include "TLatex.h"

#include "PIDMapping.h"
#include "ThermCommon.h"

TString gSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171130/Figures/TransverseEmission/";

//________________________________________________________________________________________________________________
TCanvas* DrawTransverseEmission(TString aFileLocation, ParticlePDGType aType, double aMinBetaX, double aMaxBetaX, bool aPrimaryOnly, bool bSaveImage=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanTransEmission_%s",GetPDGRootName(aType)),
                                    TString::Format("tCanTransEmission_%s",GetPDGRootName(aType)));
  tReturnCan->cd();

  tReturnCan->SetLogz();
//  gStyle->SetOptStat(0);


  TString tName3dHist;
  if(aPrimaryOnly) tName3dHist = TString::Format("%sTransverseEmissionPrimaryOnly", GetPDGRootName(aType));
  else tName3dHist = TString::Format("%sTransverseEmission", GetPDGRootName(aType));

  TH3D* t3dHist = Get3dHisto(aFileLocation, tName3dHist);

  t3dHist->GetZaxis()->SetRange(t3dHist->GetZaxis()->FindBin(aMinBetaX), t3dHist->GetZaxis()->FindBin(aMaxBetaX));
  TH2D* t2dHist = (TH2D*)t3dHist->Project3D("yx");
  t2dHist->Rebin2D(2,2);
  t2dHist->GetYaxis()->SetRangeUser(-15., 15.);

  t2dHist->GetXaxis()->SetTitle("R_{x} (fm)");
  t2dHist->GetYaxis()->SetTitle("R_{y} (fm)");

  t2dHist->Draw("colz");

  //------------------------------
  TLine* tLine = new TLine(t2dHist->GetMean(1), 
                           t2dHist->GetYaxis()->GetBinLowEdge(t2dHist->GetYaxis()->GetFirst()), 
                           t2dHist->GetMean(1), 
                           t2dHist->GetYaxis()->GetBinUpEdge(t2dHist->GetYaxis()->GetLast()));
//  tLine->SetLineColor(TColor::GetColorTransparent(kGray,0.75));
  tLine->SetLineColor(kBlack);
  tLine->Draw();

  //------------------------------
  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.04);
  tTex->DrawLatex(-15., 10., TString::Format("%0.2f < #beta_{X} < %0.2f", aMinBetaX, aMaxBetaX));
  tTex->DrawLatex(-15., 7.5, "-0.10 < #beta_{Y} < 0.10");

  //------------------------------
  if(bSaveImage)
  {
    TString tSaveLocationFull;
    TString tModifier = "";
    if(aPrimaryOnly) tModifier = TString("PrimaryOnly");

    tSaveLocationFull = gSaveLocationBase + TString::Format("%s%s%0.2fto%0.2f.eps", tReturnCan->GetName(), tModifier.Data(), aMinBetaX, aMaxBetaX);
    tReturnCan->SaveAs(tSaveLocationFull);
  }
  //------------------------------
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* DrawTransverseEmissionVsTau(TString aFileLocation, ParticlePDGType aType, double aMinMagBeta, double aMaxMagBeta, bool aPrimaryOnly, bool bSaveImage=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanTransEmissionVsTau_%s",GetPDGRootName(aType)),
                                    TString::Format("tCanTransEmissionVsTau_%s",GetPDGRootName(aType)));
  tReturnCan->cd();

  tReturnCan->SetLogz();
//  gStyle->SetOptStat(0);


  TString tName3dHist;
  if(aPrimaryOnly) tName3dHist = TString::Format("%sTransverseEmissionVsTauPrimaryOnly", GetPDGRootName(aType));
  else tName3dHist = TString::Format("%sTransverseEmissionVsTau", GetPDGRootName(aType));

  TH3D* t3dHist = Get3dHisto(aFileLocation, tName3dHist);

  t3dHist->GetZaxis()->SetRange(t3dHist->GetZaxis()->FindBin(aMinMagBeta), t3dHist->GetZaxis()->FindBin(aMaxMagBeta));
  TH2D* t2dHist = (TH2D*)t3dHist->Project3D("yx");
//  t2dHist->Rebin2D(2,2);
  t2dHist->GetXaxis()->SetRangeUser(0., 12.);
  t2dHist->GetYaxis()->SetRangeUser(0., 10.0);

  t2dHist->GetXaxis()->SetTitle("R_{T} (fm)");
  t2dHist->GetYaxis()->SetTitle("#tau (fm/c)");

  t2dHist->Draw("colz");

  //------------------------------
  if(bSaveImage)
  {
    TString tSaveLocationFull;
    TString tModifier = "";
    if(aPrimaryOnly) tModifier = TString("PrimaryOnly");

    tSaveLocationFull = gSaveLocationBase + TString::Format("%s%s%0.2fto%0.2f.eps", tReturnCan->GetName(), tModifier.Data(), aMinMagBeta, aMaxMagBeta);
    tReturnCan->SaveAs(tSaveLocationFull);
  }
  //------------------------------
  return tReturnCan;
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


  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocation = tDirectory + "SingleParticleAnalysesv2.root";

/*
  TString tDirectory = "~/Analysis/ReducedTherminator2Events/test/";
  TString tFileLocation = tDirectory + "testSingleParticleAnalysesv2.root";
*/  

  bool bDrawPrimaryOnly = true;
  bool bSaveFigures = false;
//-------------------------------------------------------------------------------

  double tBetaXMin = 0.60;
  double tBetaXMax = 0.69;

  TCanvas* tCanK0 = DrawTransverseEmission(tFileLocation, kPDGK0, tBetaXMin, tBetaXMax, bDrawPrimaryOnly, bSaveFigures);
  TCanvas* tCanKchP = DrawTransverseEmission(tFileLocation, kPDGKchP, tBetaXMin, tBetaXMax, bDrawPrimaryOnly, bSaveFigures);
  TCanvas* tCanProt = DrawTransverseEmission(tFileLocation, kPDGProt, tBetaXMin, tBetaXMax, bDrawPrimaryOnly, bSaveFigures);
  TCanvas* tCanLam = DrawTransverseEmission(tFileLocation, kPDGLam, tBetaXMin, tBetaXMax, bDrawPrimaryOnly, bSaveFigures);

//-------------------------------------------------------------------------------

  double tBetaMagMin = 0.00;
  double tBetaMagMax = 0.99;

  TCanvas* tCanVsTauK0 = DrawTransverseEmissionVsTau(tFileLocation, kPDGK0, tBetaMagMin, tBetaMagMax, bDrawPrimaryOnly, bSaveFigures);
  TCanvas* tCanVsTauKchP = DrawTransverseEmissionVsTau(tFileLocation, kPDGKchP, tBetaMagMin, tBetaMagMax, bDrawPrimaryOnly, bSaveFigures);
  TCanvas* tCanVsTauProt = DrawTransverseEmissionVsTau(tFileLocation, kPDGProt, tBetaMagMin, tBetaMagMax, bDrawPrimaryOnly, bSaveFigures);
  TCanvas* tCanVsTauLam = DrawTransverseEmissionVsTau(tFileLocation, kPDGLam, tBetaMagMin, tBetaMagMax, bDrawPrimaryOnly, bSaveFigures);


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}




