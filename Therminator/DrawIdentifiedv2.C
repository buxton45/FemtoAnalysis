#include "Identifiedv2.h"
#include "CanvasPartition.h"

#include "TLatex.h"


//---------------------------------------------------------------------------------------------------------------------------------
void BuildPanel(CanvasPartition* aCanPart, int aNx, int aNy, v2CentType aCentType)
{
  TGraphAsymmErrors* tGrKch = KchGraphColl[(int)aCentType];
  TGraphAsymmErrors* tGrK0s = K0sGraphColl[(int)aCentType];
  TGraphAsymmErrors* tGrLam = LamGraphColl[(int)aCentType];
  //-----
  TString tNameKch = TString::Format("K^{#pm} (%s)", cv2CentTags[aCentType]);
  TString tNameK0s = TString::Format("K^{0}_{S} (%s)", cv2CentTags[aCentType]);
  TString tNameLam = TString::Format("#Lambda + #bar{#Lambda} (%s)", cv2CentTags[aCentType]);
  //-----
  tGrKch->SetName(tNameKch);
  tGrKch->SetTitle(tNameKch);

  tGrK0s->SetName(tNameK0s);
  tGrK0s->SetTitle(tNameK0s);

  tGrLam->SetName(tNameLam);
  tGrLam->SetTitle(tNameLam);
  //-----
  //SO dumb...the following is to keep CanvasPartition from drawing the axes of tGrKch, and instead draw the axes 
  //          I set in the declaration of the CanvasPartiton object!!!
  vector<double> tAxesRanges = aCanPart->GetAxesRanges();
  tGrKch->GetXaxis()->SetLimits(tAxesRanges[0], tAxesRanges[1]);
  tGrKch->GetYaxis()->SetRangeUser(tAxesRanges[2], tAxesRanges[3]);

  aCanPart->AddGraph(aNx, aNy, tGrKch, "", MarkerStyleKch, ColorKch, MarkerSize, "APex0");
  aCanPart->AddGraph(aNx, aNy, tGrK0s, "", MarkerStyleK0s, ColorK0s, MarkerSize, "Pex0same");
  aCanPart->AddGraph(aNx, aNy, tGrLam, "", MarkerStyleLam, ColorLam, MarkerSize, "Pex0same");

  aCanPart->AddPadPaveText(aCanPart->SetupTPaveText(cv2CentTags[aCentType], aNx, aNy, 0.15, 0.85), aNx, aNy);
}

//---------------------------------------------------------------------------------------------------------------------------------
void PrintSystemInfo(CanvasPartition* aCanPart, int aNx=2, int aNy=0)
{
  TPad* tPad = aCanPart->GetPad(aNx, aNy);
  tPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.125);


  tTex->DrawLatex(2.0, 0.30, "ALICE");
  tTex->DrawLatex(0.5, 0.20, "Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  tTex->DrawLatex(2.0, 0.10, "|y| < 0.5");
}

//---------------------------------------------------------------------------------------------------------------------------------
void PrintParticleSpecies(CanvasPartition* aCanPart, int aNx=2, int aNy=1)
{
  TPad* tPad = aCanPart->GetPad(aNx, aNy);
  tPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.125);

  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(1.5);


  tTex->DrawLatex(1.0, 0.30, "K^{#pm}");
  tMarker->SetMarkerStyle(MarkerStyleKch);
  tMarker->SetMarkerColor(ColorKch);
  tMarker->DrawMarker(3.0, 0.30);

  tTex->DrawLatex(1.0, 0.20, "K^{0}_{S}");
  tMarker->SetMarkerStyle(MarkerStyleK0s);
  tMarker->SetMarkerColor(ColorK0s);
  tMarker->DrawMarker(3.0, 0.20);

  tTex->DrawLatex(0.5, 0.10, "#Lambda + #bar{#Lambda}");
  tMarker->SetMarkerStyle(MarkerStyleLam);
  tMarker->SetMarkerColor(ColorLam);
  tMarker->DrawMarker(3.0, 0.10);
}




//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  TString tCanName = "tCanIdv2";
  int tNx=3, tNy=3;
  double tXLow = -0.2;
  double tXHigh = 6.5;
  double tYLow = -0.02;
  double tYHigh = 0.425;

  TH1D* tDummyHist = new TH1D("tDummyHist", "tDummyHist", 1, tXLow, tXHigh);
  tDummyHist->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  CanvasPartition* tCanPartv2 = new CanvasPartition(tCanName, tNx, tNy, tXLow, tXHigh, tYLow, tYHigh, 0.12, 0.05, 0.13, 0.0025);
  tCanPartv2->SetDrawOptStat(false);
//  tCanPartv2->GetCanvas()->SetCanvasSize(1400,1500);

  BuildPanel(tCanPartv2, 0, 0, k0005);
  BuildPanel(tCanPartv2, 1, 0, k0510);
  tCanPartv2->AddGraph(2, 0, tDummyHist, "", 20, 1, 0., "AXIG");

  BuildPanel(tCanPartv2, 0, 1, k1020);
  BuildPanel(tCanPartv2, 1, 1, k2030);
  tCanPartv2->AddGraph(2, 1, tDummyHist, "", 20, 1, 0., "AXIG");

  BuildPanel(tCanPartv2, 0, 2, k3040);
  BuildPanel(tCanPartv2, 1, 2, k4050);
  BuildPanel(tCanPartv2, 2, 2, k5060);


  tCanPartv2->DrawAll();
  tCanPartv2->DrawXaxisTitle("#it{p}_{T} (GeV/#it{c})");
  tCanPartv2->DrawYaxisTitle("#it{v}_{2}{SP |#Delta #eta| > 0.9}",43,25,0.05,0.50);


  PrintSystemInfo(tCanPartv2);
  PrintParticleSpecies(tCanPartv2);







//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  cout << "DONE" << endl;
  return 0;
}




