#include "UsefulMacros.C"


//_________________________________________________________________________________________
void buildCorr()
{

  const double MinNorm = 0.30;
  const double MaxNorm = 0.50;

  const int fRebinFactor = 1;
//-------------------------------------------------------------------------------------

  TString FileName1 = "~/Analysis/K0Lam/Results_cXicKch_20160125/Results_cXicKch_20160125_Bp1.root";

  TH1F *Num1a1 = GetHisto(FileName1,"XiKchP_0010","NumKStarCf_XiKchP");
  TH1F *Den1a1 = GetHisto(FileName1,"XiKchP_0010","DenKStarCf_XiKchP");
  TH1F *Cf1a1 = buildCF("Cf_XiKchP1","Xi-KchP1",Num1a1,Den1a1,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num1b1 = GetHisto(FileName1,"AXiKchM_0010","NumKStarCf_AXiKchM");
  TH1F *Den1b1 = GetHisto(FileName1,"AXiKchM_0010","DenKStarCf_AXiKchM");
  TH1F *Cf1b1 = buildCF("Cf_AXiKchM1","AXi-KchM1",Num1b1,Den1b1,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2a1 = GetHisto(FileName1,"XiKchM_0010","NumKStarCf_XiKchM");
  TH1F *Den2a1 = GetHisto(FileName1,"XiKchM_0010","DenKStarCf_XiKchM");
  TH1F *Cf2a1 = buildCF("Cf_XiKchM1","Xi-KchM1",Num2a1,Den2a1,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2b1 = GetHisto(FileName1,"AXiKchP_0010","NumKStarCf_AXiKchP");
  TH1F *Den2b1 = GetHisto(FileName1,"AXiKchP_0010","DenKStarCf_AXiKchP");
  TH1F *Cf2b1 = buildCF("Cf_AXiKchP1","AXi-KchP1",Num2b1,Den2b1,MinNorm,MaxNorm,fRebinFactor);

//-------------------------------------------------------------------------------------

  TString FileName2 = "~/Analysis/K0Lam/Results_cXicKch_20160125/Results_cXicKch_20160125_Bp1.root";

  TH1F *Num1a2 = GetHisto(FileName2,"XiKchP_0010","NumKStarCf_XiKchP");
  TH1F *Den1a2 = GetHisto(FileName2,"XiKchP_0010","DenKStarCf_XiKchP");
  TH1F *Cf1a2 = buildCF("Cf_XiKchP2","Xi-KchP2",Num1a2,Den1a2,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num1b2 = GetHisto(FileName2,"AXiKchM_0010","NumKStarCf_AXiKchM");
  TH1F *Den1b2 = GetHisto(FileName2,"AXiKchM_0010","DenKStarCf_AXiKchM");
  TH1F *Cf1b2 = buildCF("Cf_AXiKchM2","AXi-KchM2",Num1b2,Den1b2,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2a2 = GetHisto(FileName2,"XiKchM_0010","NumKStarCf_XiKchM");
  TH1F *Den2a2 = GetHisto(FileName2,"XiKchM_0010","DenKStarCf_XiKchM");
  TH1F *Cf2a2 = buildCF("Cf_XiKchM2","Xi-KchM2",Num2a2,Den2a2,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2b2 = GetHisto(FileName2,"AXiKchP_0010","NumKStarCf_AXiKchP");
  TH1F *Den2b2 = GetHisto(FileName2,"AXiKchP_0010","DenKStarCf_AXiKchP");
  TH1F *Cf2b2 = buildCF("Cf_AXiKchP2","AXi-KchP2",Num2b2,Den2b2,MinNorm,MaxNorm,fRebinFactor);

//-------------------------------------------------------------------------------------
  TString FileName3 = "~/Analysis/K0Lam/Results_cXicKch_20160125/Results_cXicKch_20160125_Bm1.root";

  TH1F *Num1a3 = GetHisto(FileName3,"XiKchP_0010","NumKStarCf_XiKchP");
  TH1F *Den1a3 = GetHisto(FileName3,"XiKchP_0010","DenKStarCf_XiKchP");
  TH1F *Cf1a3 = buildCF("Cf_XiKchP3","Xi-KchP3",Num1a3,Den1a3,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num1b3 = GetHisto(FileName3,"AXiKchM_0010","NumKStarCf_AXiKchM");
  TH1F *Den1b3 = GetHisto(FileName3,"AXiKchM_0010","DenKStarCf_AXiKchM");
  TH1F *Cf1b3 = buildCF("Cf_AXiKchM3","AXi-KchM3",Num1b3,Den1b3,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2a3 = GetHisto(FileName3,"XiKchM_0010","NumKStarCf_XiKchM");
  TH1F *Den2a3 = GetHisto(FileName3,"XiKchM_0010","DenKStarCf_XiKchM");
  TH1F *Cf2a3 = buildCF("Cf_XiKchM3","Xi-KchM3",Num2a3,Den2a3,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2b3 = GetHisto(FileName3,"AXiKchP_0010","NumKStarCf_AXiKchP");
  TH1F *Den2b3 = GetHisto(FileName3,"AXiKchP_0010","DenKStarCf_AXiKchP");
  TH1F *Cf2b3 = buildCF("Cf_AXiKchP3","AXi-KchP3",Num2b3,Den2b3,MinNorm,MaxNorm,fRebinFactor);

//-------------------------------------------------------------------------------------

  TString FileName4 = "~/Analysis/K0Lam/Results_cXicKch_20160125/Results_cXicKch_20160125_Bm2.root";

  TH1F *Num1a4 = GetHisto(FileName4,"XiKchP_0010","NumKStarCf_XiKchP");
  TH1F *Den1a4 = GetHisto(FileName4,"XiKchP_0010","DenKStarCf_XiKchP");
  TH1F *Cf1a4 = buildCF("Cf_XiKchP4","Xi-KchP4",Num1a4,Den1a4,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num1b4 = GetHisto(FileName4,"AXiKchM_0010","NumKStarCf_AXiKchM");
  TH1F *Den1b4 = GetHisto(FileName4,"AXiKchM_0010","DenKStarCf_AXiKchM");
  TH1F *Cf1b4 = buildCF("Cf_AXiKchM4","AXi-KchM4",Num1b4,Den1b4,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2a4 = GetHisto(FileName4,"XiKchM_0010","NumKStarCf_XiKchM");
  TH1F *Den2a4 = GetHisto(FileName4,"XiKchM_0010","DenKStarCf_XiKchM");
  TH1F *Cf2a4 = buildCF("Cf_XiKchM4","Xi-KchM4",Num2a4,Den2a4,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2b4 = GetHisto(FileName4,"AXiKchP_0010","NumKStarCf_AXiKchP");
  TH1F *Den2b4 = GetHisto(FileName4,"AXiKchP_0010","DenKStarCf_AXiKchP");
  TH1F *Cf2b4 = buildCF("Cf_AXiKchP4","AXi-KchP4",Num2b4,Den2b4,MinNorm,MaxNorm,fRebinFactor);

//-------------------------------------------------------------------------------------

  TString FileName5 = "~/Analysis/K0Lam/Results_cXicKch_20160125/Results_cXicKch_20160125_Bm3.root";

  TH1F *Num1a5 = GetHisto(FileName5,"XiKchP_0010","NumKStarCf_XiKchP");
  TH1F *Den1a5 = GetHisto(FileName5,"XiKchP_0010","DenKStarCf_XiKchP");
  TH1F *Cf1a5 = buildCF("Cf_XiKchP5","Xi-KchP5",Num1a5,Den1a5,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num1b5 = GetHisto(FileName5,"AXiKchM_0010","NumKStarCf_AXiKchM");
  TH1F *Den1b5 = GetHisto(FileName5,"AXiKchM_0010","DenKStarCf_AXiKchM");
  TH1F *Cf1b5 = buildCF("Cf_AXiKchM5","AXi-KchM5",Num1b5,Den1b5,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2a5 = GetHisto(FileName5,"XiKchM_0010","NumKStarCf_XiKchM");
  TH1F *Den2a5 = GetHisto(FileName5,"XiKchM_0010","DenKStarCf_XiKchM");
  TH1F *Cf2a5 = buildCF("Cf_XiKchM5","Xi-KchM5",Num2a5,Den2a5,MinNorm,MaxNorm,fRebinFactor);

  TH1F *Num2b5 = GetHisto(FileName5,"AXiKchP_0010","NumKStarCf_AXiKchP");
  TH1F *Den2b5 = GetHisto(FileName5,"AXiKchP_0010","DenKStarCf_AXiKchP");
  TH1F *Cf2b5 = buildCF("Cf_AXiKchP5","AXi-KchP5",Num2b5,Den2b5,MinNorm,MaxNorm,fRebinFactor);

//-------------------------------------------------------------------------------------

//---------
  TList *NumList1a = new TList();
    NumList1a->Add(Num1a1);
    NumList1a->Add(Num1a2);
    NumList1a->Add(Num1a3);
    NumList1a->Add(Num1a4);
    NumList1a->Add(Num1a5);
  TList *CfList1a = new TList();
    CfList1a->Add(Cf1a1);
    CfList1a->Add(Cf1a2);
    CfList1a->Add(Cf1a3);
    CfList1a->Add(Cf1a4);
    CfList1a->Add(Cf1a5);

  TH1F *Cf1a = CombineCFs("XiKchP","XiKchP",CfList1a,NumList1a,MinNorm,MaxNorm);

//---------
  TList *NumList1b = new TList();
    NumList1b->Add(Num1b1);
    NumList1b->Add(Num1b2);
    NumList1b->Add(Num1b3);
    NumList1b->Add(Num1b4);
    NumList1b->Add(Num1b5);
  TList *CfList1b = new TList();
    CfList1b->Add(Cf1b1);
    CfList1b->Add(Cf1b2);
    CfList1b->Add(Cf1b3);
    CfList1b->Add(Cf1b4);
    CfList1b->Add(Cf1b5);

  TH1F *Cf1b = CombineCFs("XiKchP","XiKchP",CfList1b,NumList1b,MinNorm,MaxNorm);

//---------
  TList *NumList2a = new TList();
    NumList2a->Add(Num2a1);
    NumList2a->Add(Num2a2);
    NumList2a->Add(Num2a3);
    NumList2a->Add(Num2a4);
    NumList2a->Add(Num2a5);
  TList *CfList2a = new TList();
    CfList2a->Add(Cf2a1);
    CfList2a->Add(Cf2a2);
    CfList2a->Add(Cf2a3);
    CfList2a->Add(Cf2a4);
    CfList2a->Add(Cf2a5);

  TH1F *Cf2a = CombineCFs("XiKchP","XiKchP",CfList2a,NumList2a,MinNorm,MaxNorm);

//---------
  TList *NumList2b = new TList();
    NumList2b->Add(Num2b1);
    NumList2b->Add(Num2b2);
    NumList2b->Add(Num2b3);
    NumList2b->Add(Num2b4);
    NumList2b->Add(Num2b5);
  TList *CfList2b = new TList();
    CfList2b->Add(Cf2b1);
    CfList2b->Add(Cf2b2);
    CfList2b->Add(Cf2b3);
    CfList2b->Add(Cf2b4);
    CfList2b->Add(Cf2b5);

  TH1F *Cf2b = CombineCFs("XiKchP","XiKchP",CfList2b,NumList2b,MinNorm,MaxNorm);

//-------------------------------------------------------------------------------------
  TCanvas *aDrawingCanvas = new TCanvas("aDrawingCanvas","aDrawingCanvas");
  aDrawingCanvas->Divide(2,2);

  double tYmin = 0.6;
  double tYmax = 1.5;

  aDrawingCanvas->cd(1);
  Cf1a->GetYaxis()->SetRangeUser(tYmin,tYmax);
  Cf1a->SetMarkerStyle(20);
  Cf1a->Draw();

  aDrawingCanvas->cd(2);
  Cf1b->GetYaxis()->SetRangeUser(tYmin,tYmax);
  Cf1b->SetMarkerStyle(20);
  Cf1b->Draw();

  aDrawingCanvas->cd(3);
  Cf2a->GetYaxis()->SetRangeUser(tYmin,tYmax);
  Cf2a->SetMarkerStyle(20);
  Cf2a->Draw();

  aDrawingCanvas->cd(4);
  Cf2b->GetYaxis()->SetRangeUser(tYmin,tYmax);
  Cf2b->SetMarkerStyle(20);
  Cf2b->Draw();
//-------------------------------------------------------------------------------------



}
