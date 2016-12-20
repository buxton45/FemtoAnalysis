#include "UsefulMacros.C"
//_________________________________________________________________________________________
void buildCorrCombined()
{
  const Int_t MinNormBin = 60;
  const Int_t MaxNormBin = 75;

  TString File_Bp1 = "Resultsgrid_cLamK0_CentBins_Bp1NEW.root";
  TString File_Bp2 = "Resultsgrid_cLamK0_CentBins_Bp2NEW.root";
  TString File_Bm1 = "Resultsgrid_cLamK0_CentBins_Bm1NEW.root";
  TString File_Bm2 = "Resultsgrid_cLamK0_CentBins_Bm2NEW.root";
  TString File_Bm3 = "Resultsgrid_cLamK0_CentBins_Bm3NEW.root";


  //_____________________________________Lam-K0________________________________
  TString NumName_LamK0 = "NumKStarCF_LamK0";
  TString DenName_LamK0 = "DenKStarCF_LamK0";

  TH1F *NumLamK0Bp1 = GetHisto(File_Bp1, "LamK0", NumName_LamK0);
  TH1F *DenLamK0Bp1 = GetHisto(File_Bp1, "LamK0", DenName_LamK0);
  TH1F *CfLamK0Bp1 = buildCF("CfLamK0Bp1","Lam-K0 (B+ 1)",NumLamK0Bp1,DenLamK0Bp1,MinNormBin,MaxNormBin);
    CfLamK0Bp1->SetLineColor(1);

  TH1F *NumLamK0Bp2 = GetHisto(File_Bp2, "LamK0", NumName_LamK0);
  TH1F *DenLamK0Bp2 = GetHisto(File_Bp2, "LamK0", DenName_LamK0);
  TH1F *CfLamK0Bp2 = buildCF("CfLamK0Bp2","Lam-K0 (B+ 2)",NumLamK0Bp2,DenLamK0Bp2,MinNormBin,MaxNormBin);
    CfLamK0Bp2->SetLineColor(1);

  TH1F *NumLamK0Bm1 = GetHisto(File_Bm1, "LamK0", NumName_LamK0);
  TH1F *DenLamK0Bm1 = GetHisto(File_Bm1, "LamK0", DenName_LamK0);
  TH1F *CfLamK0Bm1 = buildCF("CfLamK0Bm1","Lam-K0 (B- 1)",NumLamK0Bm1,DenLamK0Bm1,MinNormBin,MaxNormBin);
    CfLamK0Bm1->SetLineColor(1);

  TH1F *NumLamK0Bm2 = GetHisto(File_Bm2, "LamK0", NumName_LamK0);
  TH1F *DenLamK0Bm2 = GetHisto(File_Bm2, "LamK0", DenName_LamK0);
  TH1F *CfLamK0Bm2 = buildCF("CfLamK0Bm2","Lam-K0 (B- 2)",NumLamK0Bm2,DenLamK0Bm2,MinNormBin,MaxNormBin);
    CfLamK0Bm2->SetLineColor(1);

  TH1F *NumLamK0Bm3 = GetHisto(File_Bm3, "LamK0", NumName_LamK0);
  TH1F *DenLamK0Bm3 = GetHisto(File_Bm3, "LamK0", DenName_LamK0);
  TH1F *CfLamK0Bm3 = buildCF("CfLamK0Bm3","Lam-K0 (B- 3)",NumLamK0Bm3,DenLamK0Bm3,MinNormBin,MaxNormBin);
    CfLamK0Bm3->SetLineColor(1);

  //-------------------------------------
  TList* NumList_LamK0_BpTot = new TList();
    NumList_LamK0_BpTot->Add(NumLamK0Bp1);
    NumList_LamK0_BpTot->Add(NumLamK0Bp2);
  TList* DenList_LamK0_BpTot = new TList();
    DenList_LamK0_BpTot->Add(DenLamK0Bp1);
    DenList_LamK0_BpTot->Add(DenLamK0Bp2);
  TList* CfList_LamK0_BpTot = new TList();
    CfList_LamK0_BpTot->Add(CfLamK0Bp1);
    CfList_LamK0_BpTot->Add(CfLamK0Bp2);
  TH1F* CfLamK0BpTot = CombineCFs("CfLamK0BpTot","Lam-K0 (B+ Tot)",CfList_LamK0_BpTot,NumList_LamK0_BpTot,MinNormBin,MaxNormBin);

  TList* NumList_LamK0_BmTot = new TList();
    NumList_LamK0_BmTot->Add(NumLamK0Bm1);
    NumList_LamK0_BmTot->Add(NumLamK0Bm2);
    NumList_LamK0_BmTot->Add(NumLamK0Bm3);
  TList* DenList_LamK0_BmTot = new TList();
    DenList_LamK0_BmTot->Add(DenLamK0Bm1);
    DenList_LamK0_BmTot->Add(DenLamK0Bm2);
    DenList_LamK0_BmTot->Add(DenLamK0Bm3);
  TList* CfList_LamK0_BmTot = new TList();
    CfList_LamK0_BmTot->Add(CfLamK0Bm1);
    CfList_LamK0_BmTot->Add(CfLamK0Bm2);
    CfList_LamK0_BmTot->Add(CfLamK0Bm3);
  TH1F* CfLamK0BmTot = CombineCFs("CfLamK0BmTot","Lam-K0 (B- Tot)",CfList_LamK0_BmTot,NumList_LamK0_BmTot,MinNormBin,MaxNormBin);

  TList* CfList_LamK0_Tot = Merge2Lists(CfList_LamK0_BpTot,CfList_LamK0_BmTot);
  TList* NumList_LamK0_Tot = Merge2Lists(NumList_LamK0_BpTot,NumList_LamK0_BmTot);
  TList* DenList_LamK0_Tot = Merge2Lists(DenList_LamK0_BpTot,DenList_LamK0_BmTot);
  TH1F* CfLamK0Tot = CombineCFs("CfLamK0Tot","Lam-K0 (Tot)",CfList_LamK0_Tot,NumList_LamK0_Tot,MinNormBin,MaxNormBin);

  //_____________________________________ALam-K0________________________________
  TString NumName_ALamK0 = "NumKStarCF_ALamK0";
  TString DenName_ALamK0 = "DenKStarCF_ALamK0";

  TH1F *NumALamK0Bp1 = GetHisto(File_Bp1, "ALamK0", NumName_ALamK0);
  TH1F *DenALamK0Bp1 = GetHisto(File_Bp1, "ALamK0", DenName_ALamK0);
  TH1F *CfALamK0Bp1 = buildCF("CfALamK0Bp1","ALam-K0 (B+ 1)",NumALamK0Bp1,DenALamK0Bp1,MinNormBin,MaxNormBin);
    CfALamK0Bp1->SetLineColor(1);

  TH1F *NumALamK0Bp2 = GetHisto(File_Bp2, "ALamK0", NumName_ALamK0);
  TH1F *DenALamK0Bp2 = GetHisto(File_Bp2, "ALamK0", DenName_ALamK0);
  TH1F *CfALamK0Bp2 = buildCF("CfALamK0Bp2","ALam-K0 (B+ 2)",NumALamK0Bp2,DenALamK0Bp2,MinNormBin,MaxNormBin);
    CfALamK0Bp2->SetLineColor(1);

  TH1F *NumALamK0Bm1 = GetHisto(File_Bm1, "ALamK0", NumName_ALamK0);
  TH1F *DenALamK0Bm1 = GetHisto(File_Bm1, "ALamK0", DenName_ALamK0);
  TH1F *CfALamK0Bm1 = buildCF("CfALamK0Bm1","ALam-K0 (B- 1)",NumALamK0Bm1,DenALamK0Bm1,MinNormBin,MaxNormBin);
    CfALamK0Bm1->SetLineColor(1);

  TH1F *NumALamK0Bm2 = GetHisto(File_Bm2, "ALamK0", NumName_ALamK0);
  TH1F *DenALamK0Bm2 = GetHisto(File_Bm2, "ALamK0", DenName_ALamK0);
  TH1F *CfALamK0Bm2 = buildCF("CfALamK0Bm2","ALam-K0 (B- 2)",NumALamK0Bm2,DenALamK0Bm2,MinNormBin,MaxNormBin);
    CfALamK0Bm2->SetLineColor(1);

  TH1F *NumALamK0Bm3 = GetHisto(File_Bm3, "ALamK0", NumName_ALamK0);
  TH1F *DenALamK0Bm3 = GetHisto(File_Bm3, "ALamK0", DenName_ALamK0);
  TH1F *CfALamK0Bm3 = buildCF("CfALamK0Bm3","ALam-K0 (B- 3)",NumALamK0Bm3,DenALamK0Bm3,MinNormBin,MaxNormBin);
    CfALamK0Bm3->SetLineColor(1);

  //-------------------------------------
  TList* NumList_ALamK0_BpTot = new TList();
    NumList_ALamK0_BpTot->Add(NumALamK0Bp1);
    NumList_ALamK0_BpTot->Add(NumALamK0Bp2);
  TList* DenList_ALamK0_BpTot = new TList();
    DenList_ALamK0_BpTot->Add(DenALamK0Bp1);
    DenList_ALamK0_BpTot->Add(DenALamK0Bp2);
  TList* CfList_ALamK0_BpTot = new TList();
    CfList_ALamK0_BpTot->Add(CfALamK0Bp1);
    CfList_ALamK0_BpTot->Add(CfALamK0Bp2);
  TH1F* CfALamK0BpTot = CombineCFs("CfALamK0BpTot","ALam-K0 (B+ Tot)",CfList_ALamK0_BpTot,NumList_ALamK0_BpTot,MinNormBin,MaxNormBin);

  TList* NumList_ALamK0_BmTot = new TList();
    NumList_ALamK0_BmTot->Add(NumALamK0Bm1);
    NumList_ALamK0_BmTot->Add(NumALamK0Bm2);
    NumList_ALamK0_BmTot->Add(NumALamK0Bm3);
  TList* DenList_ALamK0_BmTot = new TList();
    DenList_ALamK0_BmTot->Add(DenALamK0Bm1);
    DenList_ALamK0_BmTot->Add(DenALamK0Bm2);
    DenList_ALamK0_BmTot->Add(DenALamK0Bm3);
  TList* CfList_ALamK0_BmTot = new TList();
    CfList_ALamK0_BmTot->Add(CfALamK0Bm1);
    CfList_ALamK0_BmTot->Add(CfALamK0Bm2);
    CfList_ALamK0_BmTot->Add(CfALamK0Bm3);
  TH1F* CfALamK0BmTot = CombineCFs("CfALamK0BmTot","ALam-K0 (B- Tot)",CfList_ALamK0_BmTot,NumList_ALamK0_BmTot,MinNormBin,MaxNormBin);

  TList* CfList_ALamK0_Tot = Merge2Lists(CfList_ALamK0_BpTot,CfList_ALamK0_BmTot);
  TList* NumList_ALamK0_Tot = Merge2Lists(NumList_ALamK0_BpTot,NumList_ALamK0_BmTot);
  TList* DenList_ALamK0_Tot = Merge2Lists(DenList_ALamK0_BpTot,DenList_ALamK0_BmTot);
  TH1F* CfALamK0Tot = CombineCFs("CfALamK0Tot","ALam-K0 (Tot)",CfList_ALamK0_Tot,NumList_ALamK0_Tot,MinNormBin,MaxNormBin);

//__________________________________________________________________________________________________________________

  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
//  c1->Divide(2,3);
  gStyle->SetOptStat(0);
  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);
/*
  c1->cd(1);
  CfLamK0BpTot->GetYaxis()->SetRangeUser(0.6,1.1);
  CfLamK0BpTot->Draw();
  line->Draw();

  c1->cd(3);
  CfLamK0BmTot->GetYaxis()->SetRangeUser(0.6,1.1);
  CfLamK0BmTot->Draw();
  line->Draw();

  c1->cd(5);
*/
  TAxis *xax5 = CfLamK0Tot->GetXaxis();
    xax5->SetTitle("k* (GeV/c)");
    xax5->SetTitleSize(0.05);
    xax5->SetTitleOffset(1.0);
    //xax5->CenterTitle();
  TAxis *yax5 = CfLamK0Tot->GetYaxis();
    yax5->SetRangeUser(0.8,1.1);
    yax5->SetTitle("C(k*)");
    yax5->SetTitleSize(0.05);
    yax5->SetTitleOffset(1.0);
    yax5->CenterTitle();
  CfLamK0Tot->SetMarkerStyle(20);
  CfLamK0Tot->SetMarkerSize(1);
  CfLamK0Tot->SetMarkerColor(1);
  CfLamK0Tot->SetLineColor(1);
  CfLamK0Tot->SetTitle("#Lambda(#bar{#Lambda})-K^{0}");
  CfLamK0Tot->Draw();
/*
  line->Draw();
  TLegend *leg5 = new TLegend(0.70,0.15,0.85,0.30);
    leg5->SetFillColor(0);
    leg5->AddEntry(CfLamK0Tot, "#Lambda-K^{0}", "p");
    leg5->Draw();
*/
/*
  c1->cd(2);
  CfALamK0BpTot->GetYaxis()->SetRangeUser(0.6,1.1);
  CfALamK0BpTot->Draw();
  line->Draw();

  c1->cd(4);
  CfALamK0BmTot->GetYaxis()->SetRangeUser(0.6,1.1);
  CfALamK0BmTot->Draw();
  line->Draw();

  c1->cd(6);
*/

/*
  TAxis *xax6 = CfALamK0Tot->GetXaxis();
    xax6->SetTitle("k* (GeV/c)");
    xax6->SetTitleSize(0.05);
    xax6->SetTitleOffset(1.0);
    //xax6->CenterTitle();
  TAxis *yax6 = CfALamK0Tot->GetYaxis();
    yax6->SetRangeUser(0.8,1.1);
    yax6->SetTitle("C(k*)");
    yax6->SetTitleSize(0.05);
    yax6->SetTitleOffset(1.0);
    yax6->CenterTitle();
*/
  CfALamK0Tot->SetMarkerStyle(20);
  CfALamK0Tot->SetMarkerSize(1);
  CfALamK0Tot->SetMarkerColor(2);
  CfALamK0Tot->SetLineColor(2);
  CfALamK0Tot->SetTitle("#bar{#Lambda}-K^{0}");
  CfALamK0Tot->Draw("same");
  line->Draw();
  TLegend *leg6 = new TLegend(0.65,0.15,0.85,0.35);
    leg6->SetFillColor(0);
    leg6->AddEntry(CfLamK0Tot, "#Lambda-K^{0}", "p");
    leg6->AddEntry(CfALamK0Tot, "#bar{#Lambda}-K^{0}", "p");
    leg6->Draw();

  c1->SaveAs("Resultsgrid_cLamK0_CentBins_TotNEW.pdf");


}
