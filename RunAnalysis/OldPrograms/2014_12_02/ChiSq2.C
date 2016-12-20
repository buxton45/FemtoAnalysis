TH1F* GetHisto(char* FileName, char* ListName, char* ArrayName, char* HistoName)
{
  TFile f1(FileName);
  TList *femtolist = (TList*)f1.Get(ListName);
  if(ArrayName)
    {
      TObjArray *array = (TObjArray*)femtolist->FindObject(ArrayName);
      TH1F *ReturnHisto = (TH1F*)array->FindObject(HistoName);
    }
  else
    {
      TH1F *ReturnHisto = (TH1F*)femtolist->FindObject(HistoName);
    }
  return ReturnHisto;
}
//_________________________________________________________________________________________
void Normalize(TH1F* fHisto, Int_t fMinNormBin, Int_t fMaxNormBin)
{
  float scale = fHisto->Integral(fMinNormBin,fMaxNormBin);
  fHisto->Scale(1./scale);
}
//_________________________________________________________________________________________
TH1F *BuildCorrFctn(char* name, TH1F* fNum, TH1F* fDen)
{
  TH1F* fCorrFctn = fNum->Clone(name);
  fCorrFctn->Divide(fDen);
  return fCorrFctn;
}
//_________________________________________________________________________________________
void ChiSq2()
{
  const Int_t MinNormBin = 60;
  const Int_t MaxNormBin = 75;
  //-----------------------------
  TH1F *NumK0Lam1 = GetHisto("Analysis1Resultsgrid_K0Lam_Bp1_Pur.root", "femtolist", "K0Lam", "NumK0LamKStarcf");
    Normalize(NumK0Lam1,MinNormBin,MaxNormBin);
  TH1F *DenK0Lam1 = GetHisto("Analysis1Resultsgrid_K0Lam_Bp1_Pur.root", "femtolist", "K0Lam", "DenK0LamKStarcf");
    Normalize(DenK0Lam1,MinNormBin,MaxNormBin);
  TH1F *CfK0Lam1 = BuildCorrFctn("K0Lam1", NumK0Lam1, DenK0Lam1);
  NumK0Lam1->SetLineColor(1);
  DenK0Lam1->SetLineColor(1);
  CfK0Lam1->SetLineColor(1);

  //-----------------------------
  TH1F *NumK0Lam2 = GetHisto("Analysis1Resultsgrid_K0Lam_v3_Bp1.root", "femtolist", 0, "NummyKStarcf");
    Normalize(NumK0Lam2,MinNormBin,MaxNormBin);
  TH1F *DenK0Lam2 = GetHisto("Analysis1Resultsgrid_K0Lam_v3_Bp1.root", "femtolist", 0, "DenmyKStarcf");
    Normalize(DenK0Lam2,MinNormBin,MaxNormBin);
  TH1F *CfK0Lam2 = BuildCorrFctn("K0Lam2", NumK0Lam2, DenK0Lam2);
  NumK0Lam2->SetLineColor(2);
  DenK0Lam2->SetLineColor(2);
  CfK0Lam2->SetLineColor(2);
  //-----------------------------
  TH1F *NumK0Lam3 = GetHisto("Analysis1Resultsgrid_K0Lam_v3_Bp1_Pur.root", "femtolist", 0, "NumK0LamKStarcf");
    Normalize(NumK0Lam3,MinNormBin,MaxNormBin);
  TH1F *DenK0Lam3 = GetHisto("Analysis1Resultsgrid_K0Lam_v3_Bp1_Pur.root", "femtolist", 0, "DenK0LamKStarcf");
    Normalize(DenK0Lam3,MinNormBin,MaxNormBin);
  TH1F *CfK0Lam3 = BuildCorrFctn("K0Lam3", NumK0Lam3, DenK0Lam3);
  NumK0Lam3->SetLineColor(3);
  DenK0Lam3->SetLineColor(3);
  CfK0Lam3->SetLineColor(3);

  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(1,3);
  //--------Numerators--------------------------
  c1->cd(1);
  Double_t p12Num = NumK0Lam1->Chi2Test(NumK0Lam2,"WW");
  Double_t p13Num = NumK0Lam1->Chi2Test(NumK0Lam3,"WW");
  Double_t p23Num = NumK0Lam2->Chi2Test(NumK0Lam3,"WW");
  NumK0Lam1->Draw();
  NumK0Lam2->Draw("same");
  NumK0Lam3->Draw("same");
  //--------Denominators--------------------------
  c1->cd(2);
  Double_t p12Den = DenK0Lam1->Chi2Test(DenK0Lam2,"WW");
  Double_t p13Den = DenK0Lam1->Chi2Test(DenK0Lam3,"WW");
  Double_t p23Den = DenK0Lam2->Chi2Test(DenK0Lam3,"WW");
  DenK0Lam1->Draw();
  DenK0Lam2->Draw("same");
  DenK0Lam3->Draw("same");
  //--------Correlation Functions--------------------------
  c1->cd(3);
  Double_t p12Cf = CfK0Lam1->Chi2Test(CfK0Lam2,"WW");
  Double_t p13Cf = CfK0Lam1->Chi2Test(CfK0Lam3,"WW");
  Double_t p23Cf = CfK0Lam2->Chi2Test(CfK0Lam3,"WW");
  CfK0Lam1->Draw();
  CfK0Lam2->Draw("same");
  CfK0Lam3->Draw("same");
  //-------------------------------------------------------
  cout << "p12Num = " << p12Num << endl;
  cout << "p13Num = " << p13Num << endl;
  cout << "p23Num = " << p23Num << endl << endl;

  cout << "p12Den = " << p12Den << endl;
  cout << "p13Den = " << p13Den << endl;
  cout << "p23Den = " << p23Den << endl << endl;

  cout << "p12Cf = " << p12Cf << endl;
  cout << "p13Cf = " << p13Cf << endl;
  cout << "p23Cf = " << p23Cf << endl << endl;

}
