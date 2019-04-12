#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"
#include "NumIntLednickyCf.h"
#include "SimulatedLednickyCf.h"

#include "TLegend.h"
#include "TLegendEntry.h"

//________________________________________________________________________________________________________________
TH1D* BuildNumIntCf(NumIntLednickyCf* aNumIntLedCf, TString aName, vector<double> &aKStarBinCenters, double* aParams, int aColor=kBlack, int aMarkerStyle=20)
{
  double tNBins = aKStarBinCenters.size();
  double tKStarBinSize = aKStarBinCenters[1]-aKStarBinCenters[0];

  TH1D* tReturnCf = new TH1D(aName, aName, tNBins, 0., tNBins*tKStarBinSize);

  for(int i=0; i<tNBins; i++)
  {
    tReturnCf->SetBinContent(i+1, aNumIntLedCf->GetFitCfContent(aKStarBinCenters[i], aParams));
  }
  tReturnCf->SetMarkerStyle(aMarkerStyle);
  tReturnCf->SetMarkerSize(0.75);
  tReturnCf->SetMarkerColor(aColor);
  tReturnCf->SetLineColor(aColor);

  return tReturnCf;
}
//________________________________________________________________________________________________________________
TH1D* BuildSimCf(SimulatedLednickyCf* aSimLedCf, TString aName, vector<double> &aKStarBinCenters, double* aParams, int aColor=kBlack, int aMarkerStyle=20)
{
  double tNBins = aKStarBinCenters.size();
  double tKStarBinSize = aKStarBinCenters[1]-aKStarBinCenters[0];

  TH1D* tReturnCf = new TH1D(aName, aName, tNBins, 0., tNBins*tKStarBinSize);

  for(int i=0; i<tNBins; i++)
  {
    tReturnCf->SetBinContent(i+1, aSimLedCf->GetFitCfContent(aKStarBinCenters[i], aParams));
  }
  tReturnCf->SetMarkerStyle(aMarkerStyle);
  tReturnCf->SetMarkerSize(0.75);
  tReturnCf->SetMarkerColor(aColor);
  tReturnCf->SetLineColor(aColor);

  return tReturnCf;
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

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  AnalysisType tAnType = kLamKchP;

  bool bDrawSimLedCfs=false;

  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";
//  TString tSaveDir = "/home/jesse/Analysis/Presentations/GroupMeetings/20190221/Figures/";
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/7_ResultsAndDiscussion/7.1_ResultsLamK/7.1.2_ResultsLamK_DiscussionOfmTScaling/OtherFigures/";


  double aMinX = 0.0;
  double aMaxX = 0.30;

  double aMinY = 0.89;
  double aMaxY = 1.01;

  if(tAnType==kLamKchP)
  {
    aMinY = 0.915;
    aMaxY = 1.005;
  }
  else if(tAnType==kLamKchM)
  {
     aMinY = 0.9825;
     aMaxY = 1.0275;
  }

  int tIntType = 2;
  int tNCalls = 50000;
  double tMaxIntRadius = 100.;

  NumIntLednickyCf* tNumIntLedCf = new NumIntLednickyCf(tIntType, tNCalls, tMaxIntRadius);

  double tKStarBinSize = 0.01;
  int tNBins = 50;
  vector<double> tKStarBinCenters;
  for(int i=0; i<tNBins; i++) tKStarBinCenters.push_back((i+0.5)*tKStarBinSize);

  //-----------------------------------------------------------------------------
  double tLambda = 1.12*0.527;
  double tRadius1 = 6.24;
  double tRef0   = -0.49;
  double tImf0   = 0.42;
  double td0     = -0.55;  
  double tNorm   = 1.0;
  double tMuOut1  = 0.0;

  if(tAnType==kLamKchP)
  {
    tRef0   = -0.49;
    tImf0   = 0.42;
    td0     = -0.55;  
  }
  else if(tAnType==kLamKchM)
  {
    tRef0   = 0.19;
    tImf0   = 0.29;
    td0     = -7.80;
  }

  double tParams1[7] = {tLambda, tRadius1, tRef0, tImf0, td0, tNorm, tMuOut1};

  //-----------------------------------------------------------------------------
//  double tRadius2 = 4.20;
  double tRadius2 = 5.0;

  double tMuOut2a = 8.0;
  double tMuOut2b = 6.0;
  double tMuOut2c = 4.0;
  double tMuOut2d = 2.0;
  double tMuOut2e = 0.0;
  vector<int> tColors2Vec{kRed, kOrange, kYellow, kGreen, kBlack};
  td2dVec tParams2 = {{tLambda, tRadius2, tRef0, tImf0, td0, tNorm, tMuOut2a},
                      {tLambda, tRadius2, tRef0, tImf0, td0, tNorm, tMuOut2b},
                      {tLambda, tRadius2, tRef0, tImf0, td0, tNorm, tMuOut2c},
                      {tLambda, tRadius2, tRef0, tImf0, td0, tNorm, tMuOut2d},
                      {tLambda, tRadius2, tRef0, tImf0, td0, tNorm, tMuOut2e}};
  //-----------------------------------------------------------------------------

  TH1D* tCf_LedEq = new TH1D("tCf_LedEq", "tCf_LedEq", tNBins, 0., tNBins*tKStarBinSize);
  double x[1];
  for(int i=0; i<tNBins; i++)
  {
    x[0] = tKStarBinCenters[i];
    tCf_LedEq->SetBinContent(i+1, FitPartialAnalysis::LednickyEq(x, tParams1));
  }
  tCf_LedEq->GetYaxis()->SetRangeUser(0.86, 1.07);
  tCf_LedEq->SetMarkerStyle(24);
  tCf_LedEq->SetMarkerSize(0.75);
  tCf_LedEq->SetMarkerColor(kMagenta);
  tCf_LedEq->SetLineColor(kMagenta);

  tCf_LedEq->GetXaxis()->SetRangeUser(aMinX, aMaxX);
  tCf_LedEq->GetYaxis()->SetRangeUser(aMinY, aMaxY);

  tCf_LedEq->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    tCf_LedEq->GetXaxis()->SetTitleSize(0.065);
    tCf_LedEq->GetXaxis()->SetTitleOffset(1.00);
    tCf_LedEq->GetXaxis()->SetLabelSize(0.05);

  tCf_LedEq->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
  if(tAnType==kLamKchP)
  {
    tCf_LedEq->GetYaxis()->SetTitleSize(0.075);
    tCf_LedEq->GetYaxis()->SetTitleOffset(0.80);
    tCf_LedEq->GetYaxis()->SetLabelSize(0.05);
  }
  else if(tAnType==kLamKchM)
  {
    tCf_LedEq->GetYaxis()->SetTitleSize(0.075);
    tCf_LedEq->GetYaxis()->SetTitleOffset(1.0);
    tCf_LedEq->GetYaxis()->SetLabelSize(0.05);
  }
  //-------------------------
  TH1D* tCf_NumInt1 = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt1"), tKStarBinCenters, tParams1, kBlack, 20);

  TH1D* tCf_NumInt2a = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt2a"), tKStarBinCenters, tParams2[0].data(), tColors2Vec[0], 22);
  TH1D* tCf_NumInt2b = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt2b"), tKStarBinCenters, tParams2[1].data(), tColors2Vec[1], 22);
  TH1D* tCf_NumInt2c = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt2c"), tKStarBinCenters, tParams2[2].data(), tColors2Vec[2], 22);
  TH1D* tCf_NumInt2d = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt2d"), tKStarBinCenters, tParams2[3].data(), tColors2Vec[3], 22);
  TH1D* tCf_NumInt2e = BuildNumIntCf(tNumIntLedCf, TString("tCf_NumInt2e"), tKStarBinCenters, tParams2[4].data(), tColors2Vec[4], 22);
  //-------------------------

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->SetTopMargin(0.025);
  tCan->SetRightMargin(0.025);
  tCan->SetBottomMargin(0.15);
  if(tAnType==kLamKchP)      tCan->SetLeftMargin(0.125);
  else if(tAnType==kLamKchM) tCan->SetLeftMargin(0.15);
  tCan->cd();

  tCf_LedEq->Draw("p");
  tCf_NumInt1->Draw("lpsame");
  tCf_NumInt2a->Draw("lpsame");
  tCf_NumInt2b->Draw("lpsame");
  tCf_NumInt2c->Draw("lpsame");
  tCf_NumInt2d->Draw("lpsame");
  tCf_NumInt2e->Draw("lpsame");

  if(bDrawSimLedCfs)
  {
    SimulatedLednickyCf* tSimLedCf = new SimulatedLednickyCf(tKStarBinSize, tNBins*tKStarBinSize, 50000);
    TH1D* tCf_SimLedCf = BuildSimCf(tSimLedCf, TString("tCf_SimLedCf"), tKStarBinCenters, tParams1, kBlack, 25);
    TH1D* tCf_SimLedCf2a = BuildSimCf(tSimLedCf, TString("tCf_SimLedCf2a"), tKStarBinCenters, tParams2[0].data(), tColors2Vec[0], 25);

    tCf_SimLedCf->Draw("psame");
    tCf_SimLedCf2a->Draw("psame");
  }

  //-------------------------------------------------------------------------------
  double tTextSize = 0.040;
  TLegend *tLeg1, *tLeg2, *tLeg3;
  if(tAnType==kLamKchP)
  {
    tLeg1 = new TLegend(0.40, 0.75, 0.85, 0.825, "", "NDC");  
    tLeg2 = new TLegend(0.40, 0.50, 0.60, 0.70, "", "NDC");  
    tLeg3 = new TLegend(0.70, 0.35, 0.90, 0.70, "", "NDC");  
  }
  else if(tAnType==kLamKchM)
  {
    tLeg1 = new TLegend(0.40, 0.90, 0.85, 0.95, "", "NDC");  
    tLeg2 = new TLegend(0.40, 0.65, 0.60, 0.85, "", "NDC");  
    tLeg3 = new TLegend(0.70, 0.50, 0.90, 0.85, "", "NDC");  
  }

    tLeg1->SetFillColor(0);
    tLeg1->SetBorderSize(0);
    tLeg1->SetTextSize(tTextSize);
    tLeg1->SetTextAlign(21);

    tLeg2->SetFillColor(0);
    tLeg2->SetBorderSize(0);
    tLeg2->SetTextSize(tTextSize);

    tLeg3->SetFillColor(0);
    tLeg3->SetBorderSize(0);
    tLeg3->SetTextSize(tTextSize);

    //------------------------------
    tLeg1->AddEntry((TObject*)0, TString::Format("#Rgothicf_{0} = %0.2f fm, #Jgothicf_{0} = %0.2f fm, d_{0} = %0.2f fm", tRef0, tImf0, td0), "");

    tLeg2->AddEntry((TObject*)0, TString::Format("R = %0.2f fm", tRadius1), "");
    tLeg2->AddEntry((TObject*)0, "#mu_{O} = 0.0 fm", "");
    tLeg2->AddEntry(tCf_LedEq, "Lednicky Eq.", "p");
    tLeg2->AddEntry(tCf_NumInt1, "Num. Int.", "p");

    tLeg3->AddEntry((TObject*)0, TString::Format("R = %0.2f fm", tRadius2), "");
    tLeg3->AddEntry((TObject*)0, "Num. Int.", "");
    tLeg3->AddEntry(tCf_NumInt2e, TString::Format("#mu_{O} = %0.1f fm", tMuOut2e), "p");
    tLeg3->AddEntry(tCf_NumInt2d, TString::Format("#mu_{O} = %0.1f fm", tMuOut2d), "p");
    tLeg3->AddEntry(tCf_NumInt2c, TString::Format("#mu_{O} = %0.1f fm", tMuOut2c), "p");
    tLeg3->AddEntry(tCf_NumInt2b, TString::Format("#mu_{O} = %0.1f fm", tMuOut2b), "p");
    tLeg3->AddEntry(tCf_NumInt2a, TString::Format("#mu_{O} = %0.1f fm", tMuOut2a), "p");


  TLegendEntry *tLegEntry;
  tLegEntry = (TLegendEntry*)tLeg1->GetListOfPrimitives()->First();
    tLegEntry->SetTextFont(62);

  tLegEntry = (TLegendEntry*)tLeg2->GetListOfPrimitives()->First();
    tLegEntry->SetTextFont(62);
  tLegEntry = (TLegendEntry*)tLeg2->GetListOfPrimitives()->At(1);
    tLegEntry->SetTextFont(62);

  tLegEntry = (TLegendEntry*)tLeg3->GetListOfPrimitives()->First();
    tLegEntry->SetTextFont(62);
  tLegEntry = (TLegendEntry*)tLeg3->GetListOfPrimitives()->At(1);
    tLegEntry->SetTextFont(62);

  tLeg1->Draw();
  tLeg2->Draw();
  tLeg3->Draw();

  //-----------------------------------
  if(bSaveFigures)
  {
    tCan->SaveAs(TString::Format("%sNumIntLednickyCf_%s.%s", tSaveDir.Data(), cAnalysisBaseTags[tAnType], tSaveFileType.Data()));
  }

//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
