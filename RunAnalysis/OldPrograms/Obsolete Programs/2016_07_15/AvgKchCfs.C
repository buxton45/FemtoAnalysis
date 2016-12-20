#include "buildAllcLamcKch3.cxx"
#include "buildAllcLamK03.cxx"

void AvgKchCfs()
{
  vector<TString> cLamcKch_VectorOfFileNames;
    cLamcKch_VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp1NEWAS.root");
    cLamcKch_VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp2NEWAS.root");
    cLamcKch_VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm1NEWAS.root");
    cLamcKch_VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm2NEWAS.root");
    cLamcKch_VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm3NEWAS.root");

  buildAllcLamcKch3 *cLamcKchAnalysis3_0010 = new buildAllcLamcKch3(cLamcKch_VectorOfFileNames,"LamKchP_0010","LamKchM_0010","ALamKchP_0010","ALamKchM_0010");

  cLamcKchAnalysis3_0010->BuildCFCollections();

  TH1F *Cf_LamKchP_Tot = cLamcKchAnalysis3_0010->GetCf_LamKchP_Tot();
  TH1F *Cf_ALamKchP_Tot = cLamcKchAnalysis3_0010->GetCf_ALamKchP_Tot();
  TH1F *Cf_LamKchM_Tot = cLamcKchAnalysis3_0010->GetCf_LamKchM_Tot();
  TH1F *Cf_ALamKchM_Tot = cLamcKchAnalysis3_0010->GetCf_ALamKchM_Tot();

  TH1F *Cf_AverageLamKchPM_Tot = cLamcKchAnalysis3_0010->GetCf_AverageLamKchPM_Tot();
  TH1F *Cf_AverageALamKchPM_Tot = cLamcKchAnalysis3_0010->GetCf_AverageALamKchPM_Tot();

//-------------------------------------------------------------------------------------------------

  vector<TString> cLamK0_VectorOfFileNames;
    cLamK0_VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bp1NEWAS.root");
    cLamK0_VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bp2NEWAS.root");
    cLamK0_VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm1NEWAS.root");
    cLamK0_VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm2NEWAS.root");
    cLamK0_VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm3NEWAS.root");

  buildAllcLamK03 *cLamK0Analysis3_0010 = new buildAllcLamK03(cLamK0_VectorOfFileNames,"LamK0_0010","ALamK0_0010");

  cLamK0Analysis3_0010->BuildCFCollections();

  TH1F *Cf_LamK0_Tot = cLamK0Analysis3_0010->GetCf_LamK0_Tot();
  TH1F *Cf_ALamK0_Tot = cLamK0Analysis3_0010->GetCf_ALamK0_Tot();

//-------------------------------------------------------------------------------------------------

  TCanvas* aCanvas = new TCanvas("aCanvas","Plotting Canvas");
  aCanvas->Divide(2,2);
  gStyle->SetOptStat(0);
  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);
  //---------------
  aCanvas->cd(1);
  TAxis *xax1 = Cf_LamKchP_Tot->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
  TAxis *yax1 = Cf_LamKchP_Tot->GetYaxis();
    yax1->SetRangeUser(0.9,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();
  Cf_LamKchP_Tot->SetLineColor(2);
  Cf_LamKchP_Tot->SetMarkerColor(2);
  Cf_LamKchP_Tot->SetMarkerStyle(20);
  Cf_LamKchP_Tot->SetMarkerSize(0.50);
  Cf_LamKchP_Tot->SetTitle("#LambdaK+ & #LambdaK-");
  Cf_LamKchP_Tot->Draw();

  Cf_LamKchM_Tot->SetLineColor(4);
  Cf_LamKchM_Tot->SetMarkerColor(4);
  Cf_LamKchM_Tot->SetMarkerStyle(20);
  Cf_LamKchM_Tot->SetMarkerSize(0.50);
  Cf_LamKchM_Tot->Draw("same");
  line->Draw();

  leg1 = new TLegend(0.60,0.12,0.89,0.32);
  leg1->SetFillColor(0);
  leg1->AddEntry(Cf_LamKchP_Tot, "#LambdaK+","lp");
  leg1->AddEntry(Cf_LamKchM_Tot, "#LambdaK-","lp");
  leg1->Draw();

  //---------------
  aCanvas->cd(2);
  TAxis *xax2 = Cf_ALamKchP_Tot->GetXaxis();
    xax2->SetTitle("k* (GeV/c)");
    xax2->SetTitleSize(0.05);
    xax2->SetTitleOffset(1.0);
  TAxis *yax2 = Cf_ALamKchP_Tot->GetYaxis();
    yax2->SetRangeUser(0.9,1.1);
    yax2->SetTitle("C(k*)");
    yax2->SetTitleSize(0.05);
    yax2->SetTitleOffset(1.0);
    yax2->CenterTitle();
  Cf_ALamKchP_Tot->SetLineColor(4);
  Cf_ALamKchP_Tot->SetMarkerColor(4);
  Cf_ALamKchP_Tot->SetMarkerStyle(20);
  Cf_ALamKchP_Tot->SetMarkerSize(0.50);
  Cf_ALamKchP_Tot->SetTitle("#bar{#Lambda}K+ & #bar{#Lambda}K-");
  Cf_ALamKchP_Tot->Draw();

  Cf_ALamKchM_Tot->SetLineColor(2);
  Cf_ALamKchM_Tot->SetMarkerColor(2);
  Cf_ALamKchM_Tot->SetMarkerStyle(20);
  Cf_ALamKchM_Tot->SetMarkerSize(0.50);
  Cf_ALamKchM_Tot->Draw("same");
  line->Draw();

  leg2 = new TLegend(0.60,0.12,0.89,0.32);
  leg2->SetFillColor(0);
  leg2->AddEntry(Cf_ALamKchM_Tot, "#bar{#Lambda}K-","lp");
  leg2->AddEntry(Cf_ALamKchP_Tot, "#bar{#Lambda}K+","lp");
  leg2->Draw();

  //---------------
  aCanvas->cd(3);
  TAxis *xax3 = Cf_LamK0_Tot->GetXaxis();
    xax3->SetTitle("k* (GeV/c)");
    xax3->SetTitleSize(0.05);
    xax3->SetTitleOffset(1.0);
    //xax3->SetRangeUser(0.0,0.2);
  TAxis *yax3 = Cf_LamK0_Tot->GetYaxis();
    yax3->SetRangeUser(0.9,1.1);
    yax3->SetTitle("C(k*)");
    yax3->SetTitleSize(0.05);
    yax3->SetTitleOffset(1.0);
    yax3->CenterTitle();
  Cf_LamK0_Tot->SetLineColor(1);
  Cf_LamK0_Tot->SetMarkerColor(1);
  Cf_LamK0_Tot->SetMarkerStyle(20);
  Cf_LamK0_Tot->SetMarkerSize(0.50);
  Cf_LamK0_Tot->SetTitle("#LambdaK^{0} & #LambdaK+-");
  Cf_LamK0_Tot->Draw();

  Cf_AverageLamKchPM_Tot->SetLineColor(6);
  Cf_AverageLamKchPM_Tot->SetMarkerColor(6);
  Cf_AverageLamKchPM_Tot->SetMarkerStyle(20);
  Cf_AverageLamKchPM_Tot->SetMarkerSize(0.50);
  Cf_AverageLamKchPM_Tot->Draw("same");
  line->Draw();

  //---Clones so I can control the interval over which the Chi2Test occurs
  //---while still plotting the full range
  TH1F* Cf_LamK0_TotCLONE = Cf_LamK0_Tot->Clone("Cf_LamK0_TotCLONE");
  Cf_LamK0_TotCLONE->GetXaxis()->SetRangeUser(0.01,0.2);
  TH1F* Cf_AverageLamKchPM_TotCLONE = Cf_AverageLamKchPM_Tot->Clone("Cf_AverageLamKchPM_TotCLONE");
  double pLamK0 = Cf_LamK0_TotCLONE->Chi2Test(Cf_AverageLamKchPM_TotCLONE,"NORM");
  cout << "pLamK0 = " << pLamK0 << endl;

  leg3 = new TLegend(0.50,0.12,0.89,0.32);
  leg3->SetFillColor(0);
  leg3->AddEntry(Cf_LamK0_Tot, "#LambdaK^{0}", "lp");
  leg3->AddEntry(Cf_AverageLamKchPM_Tot, "Combined #LambdaK+ & #LambdaK-","lp");
  leg3->Draw();

  text3 = new TPaveText(0.70,0.60,0.89,0.75,"NDC");
  char buffer[50];
  sprintf(buffer, "p = %.9f",pLamK0);
  text3->AddText(buffer);
  text3->Draw();

  //---------------
  aCanvas->cd(4);
  TAxis *xax4 = Cf_ALamK0_Tot->GetXaxis();
    xax4->SetTitle("k* (GeV/c)");
    xax4->SetTitleSize(0.05);
    xax4->SetTitleOffset(1.0);
    //xax4->SetRangeUser(0.0,0.2);
  TAxis *yax4 = Cf_ALamK0_Tot->GetYaxis();
    yax4->SetRangeUser(0.9,1.1);
    yax4->SetTitle("C(k*)");
    yax4->SetTitleSize(0.05);
    yax4->SetTitleOffset(1.0);
    yax4->CenterTitle();
  Cf_ALamK0_Tot->SetLineColor(1);
  Cf_ALamK0_Tot->SetMarkerColor(1);
  Cf_ALamK0_Tot->SetMarkerStyle(20);
  Cf_ALamK0_Tot->SetMarkerSize(0.50);
  Cf_ALamK0_Tot->SetTitle("#bar{#Lambda}K^{0} & #bar{#Lambda}K+-");
  Cf_ALamK0_Tot->Draw();

  Cf_AverageALamKchPM_Tot->SetLineColor(6);
  Cf_AverageALamKchPM_Tot->SetMarkerColor(6);
  Cf_AverageALamKchPM_Tot->SetMarkerStyle(20);
  Cf_AverageALamKchPM_Tot->SetMarkerSize(0.50);
  Cf_AverageALamKchPM_Tot->Draw("same");
  line->Draw();

  //---Clones so I can control the interval over which the Chi2Test occurs
  //---while still plotting the full range
  TH1F* Cf_ALamK0_TotCLONE = Cf_ALamK0_Tot->Clone("Cf_ALamK0_TotCLONE");
  Cf_ALamK0_TotCLONE->GetXaxis()->SetRangeUser(0.01,0.2);
  TH1F* Cf_AverageALamKchPM_TotCLONE = Cf_AverageALamKchPM_Tot->Clone("Cf_AverageALamKchPM_TotCLONE");
  double pALamK0 = Cf_ALamK0_TotCLONE->Chi2Test(Cf_AverageALamKchPM_TotCLONE,"NORM");
  cout << "pALamK0 = " << pALamK0 << endl;

  leg4 = new TLegend(0.50,0.12,0.89,0.32);
  leg4->SetFillColor(0);
  leg4->AddEntry(Cf_ALamK0_Tot, "#bar{#Lambda}K^{0}", "lp");
  leg4->AddEntry(Cf_AverageALamKchPM_Tot, "Combined #bar{#Lambda}K- & #bar{#Lambda}K+","lp");
  leg4->Draw();

  text4 = new TPaveText(0.70,0.60,0.89,0.75,"NDC");
  char buffer[50];
  sprintf(buffer, "p = %.9f",pALamK0);
  text4->AddText(buffer);
  text4->Draw();

  //---------------
  aCanvas->SaveAs("~/Analysis/Presentations/AliFemto/2015.04.01/AvgKchCfs.pdf");

}
