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
TH1F* buildCF(char* name, char* title, TH1F* Num, TH1F* Denom, int fMinNormBin, int fMaxNormBin)
{
  double NumScale = Num->Integral(fMinNormBin,fMaxNormBin);
  Num->Scale(1./NumScale);
  double DenScale = Denom->Integral(fMinNormBin,fMaxNormBin);
  Denom->Scale(1./DenScale);

  TH1F* CF = Num->Clone(name);
  CF->Divide(Denom);
  CF->SetTitle(title);

  return CF;
}

//_________________________________________________________________________________________
void buildCorr_cLamKch()
{
  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(2,2);

  const int MinNormBin = 60;
  const int MaxNormBin = 75;

  TH1F* NumLamKchPKStarCF = GetHisto("Resultsgrid_cLamcKch_CentBins_Bp1.root","femtolist","LamKchP_1030","NumKStarCF_LamKchP_1030");
  TH1F* DenLamKchPKStarCF = GetHisto("Resultsgrid_cLamcKch_CentBins_Bp1.root","femtolist","LamKchP_1030","DenKStarCF_LamKchP_1030");
  TH1F* LamKchPKStarCF = buildCF("LamKchPKStarCF","Lam-K+",NumLamKchPKStarCF,DenLamKchPKStarCF,MinNormBin,MaxNormBin);

  TH1F* NumLamKchMKStarCF = GetHisto("Resultsgrid_cLamcKch_CentBins_Bp1.root","femtolist","LamKchM_1030","NumKStarCF_LamKchM_1030");
  TH1F* DenLamKchMKStarCF = GetHisto("Resultsgrid_cLamcKch_CentBins_Bp1.root","femtolist","LamKchM_1030","DenKStarCF_LamKchM_1030");
  TH1F* LamKchMKStarCF = buildCF("LamKchMKStarCF","Lam-K-",NumLamKchMKStarCF,DenLamKchMKStarCF,MinNormBin,MaxNormBin);

  TH1F* NumALamKchPKStarCF = GetHisto("Resultsgrid_cLamcKch_CentBins_Bp1.root","femtolist","ALamKchP_1030","NumKStarCF_ALamKchP_1030");
  TH1F* DenALamKchPKStarCF = GetHisto("Resultsgrid_cLamcKch_CentBins_Bp1.root","femtolist","ALamKchP_1030","DenKStarCF_ALamKchP_1030");
  TH1F* ALamKchPKStarCF = buildCF("ALamKchPKStarCF","ALam-K+",NumALamKchPKStarCF,DenALamKchPKStarCF,MinNormBin,MaxNormBin);

  TH1F* NumALamKchMKStarCF = GetHisto("Resultsgrid_cLamcKch_CentBins_Bp1.root","femtolist","ALamKchM_1030","NumKStarCF_ALamKchM_1030");
  TH1F* DenALamKchMKStarCF = GetHisto("Resultsgrid_cLamcKch_CentBins_Bp1.root","femtolist","ALamKchM_1030","DenKStarCF_ALamKchM_1030");
  TH1F* ALamKchMKStarCF = buildCF("ALamKchMKStarCF","ALam-K-",NumALamKchMKStarCF,DenALamKchMKStarCF,MinNormBin,MaxNormBin);

  c1->cd(1);
  LamKchPKStarCF->GetYaxis()->SetRangeUser(0.8,1.2);
  LamKchPKStarCF->Draw();
  c1->cd(2);
  ALamKchMKStarCF->GetYaxis()->SetRangeUser(0.8,1.2);
  ALamKchMKStarCF->Draw();
  c1->cd(3);
  LamKchMKStarCF->GetYaxis()->SetRangeUser(0.8,1.2);
  LamKchMKStarCF->Draw();
  c1->cd(4);
  ALamKchPKStarCF->GetYaxis()->SetRangeUser(0.8,1.2);
  ALamKchPKStarCF->Draw();

}
