Bool_t reject;
Double_t ExcludeMassMin, ExcludeMassMax;

Double_t FitFunction(Double_t *x, Double_t *par)
{
  if(reject && x[0]>ExcludeMassMin && x[0]<ExcludeMassMax)
    {
      TF1::RejectPoint();
      return 0;
    }
  
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
}

TList* BuildFit(char* name, TH1F* PurityHisto, Double_t ExcludeMin, Double_t ExcludeMax, Double_t ROImin, Double_t ROImax, Double_t *info)
{
  char buffer[50];
  sprintf(buffer, "fitBgd_%s",name);

  reject = kTRUE;
  ExcludeMassMin = ExcludeMin;
  ExcludeMassMax = ExcludeMax;
  TF1 *fitBgd = new TF1("fitBgd",FitFunction,PurityHisto->GetBinLowEdge(1),PurityHisto->GetBinLowEdge(PurityHisto->GetNbinsX()+1),5);
  PurityHisto->Fit("fitBgd","0");

  reject = kFALSE;
  TF1 *fitBgd2 = new TF1(buffer,FitFunction,PurityHisto->GetBinLowEdge(1),PurityHisto->GetBinLowEdge(PurityHisto->GetNbinsX()+1),5);
  fitBgd2->SetParameters(fitBgd->GetParameters());

  //--------------------------------------------------------------------------------------------
  Double_t bgd = fitBgd2->Integral(ROImin,ROImax);
  bgd /= PurityHisto->GetBinWidth(0);  //divide by bin size
  cout << name << ": " << "bgd = " << bgd << endl;
  //-----
  Double_t sigpbgd = PurityHisto->Integral(PurityHisto->FindBin(ROImin),PurityHisto->FindBin(ROImax));
  cout << name << ": " << "sig+bgd = " << sigpbgd << endl;
  //-----
  Double_t sig = sigpbgd - bgd;
  cout << name << ": " << "sig = " << sig << endl;
  //-----
  Double_t pur = sig/sigpbgd;
  cout << name << ": " << "Pur = " << pur << endl << endl;

  info[0] = bgd;
  info[1] = sigpbgd;
  info[2] = sig;
  info[3] = pur;
  //--------------------------------------------------------------------------------------------
  Double_t HistoMaxValue = PurityHisto->GetMaximum();
  TLine *lROImin = new TLine(ROImin,0,ROImin,HistoMaxValue);
  lROImin->SetLineColor(3);
  TLine *lROImax = new TLine(ROImax,0,ROImax,HistoMaxValue);
  lROImax->SetLineColor(3);
  TLine *lExcludeMin = new TLine(ExcludeMin,0,ExcludeMin,HistoMaxValue);
  lExcludeMin->SetLineColor(2);
  TLine *lExcludeMax = new TLine(ExcludeMax,0,ExcludeMax,HistoMaxValue);
  lExcludeMax->SetLineColor(2);


  TList* temp = new TList();
  temp->Add(fitBgd2);
  temp->Add(lROImin);
  temp->Add(lROImax);
  temp->Add(lExcludeMin);
  temp->Add(lExcludeMax);

  return temp;

}

void calcPur()
{
  const Double_t LambdaMass = 1.115683, KaonMass = 0.493677;

  TFile f1("Analysis1Resultsgrid_K0ALam_v3_Bp1_Pur.root");
  TList *femtolist = (TList*)f1.Get("femtolist");


  //------------------------------AntiLambdas---------------------------------------------------
  TH1F *AntiLambdaPurity = (TH1F*)femtolist->FindObject("AntiLambdaPurity");
  AntiLambdaPurity->SetLineColor(1);
  AntiLambdaPurity->SetLineWidth(3);

  Double_t ALamInfo[4];
  Double_t ALamExcludeMin = 1.106;
  Double_t ALamExcludeMax = 1.126; 
  Double_t ALamROImin = LambdaMass-0.0019;
  Double_t ALamROImax = LambdaMass+0.0019;
  TList* ALamList = BuildFit("AntiLambda",AntiLambdaPurity,ALamExcludeMin,ALamExcludeMax,ALamROImin,ALamROImax,ALamInfo);
  TIter iter(ALamList);
  TF1* fitBgd_AntiLambda = *(iter.Begin());
  TLine* lALamROImin = (TLine*)iter.Next();
  TLine* lALamROImax = (TLine*)iter.Next();
  TLine* lALamExcludeMin = (TLine*)iter.Next();
  TLine* lALamExcludeMax = (TLine*)iter.Next();
  fitBgd_AntiLambda->SetLineColor(4);
  Double_t ALamBgd, ALamSigpBgd, ALamSig, ALamPur;
  ALamBgd = ALamInfo[0];
  ALamSigpBgd = ALamInfo[1];
  ALamSig = ALamInfo[2];
  ALamPur = ALamInfo[3];

  //------------------------------K0Shorts---------------------------------------------------------
  TH1F *K0ShortPurity = (TH1F*)femtolist->FindObject("K0ShortPurity");
  K0ShortPurity->SetLineColor(1);
  K0ShortPurity->SetLineWidth(3);

  Double_t K0Info[4];
  Double_t K0ExcludeMin = 0.478;
  Double_t K0ExcludeMax = 0.516; 
  Double_t K0ROImin = KaonMass-0.013677;
  Double_t K0ROImax = KaonMass+0.020323;
  TList* K0List = BuildFit("K0Short",K0ShortPurity,K0ExcludeMin,K0ExcludeMax,K0ROImin,K0ROImax,K0Info);
  TIter iter(K0List);
  TF1* fitBgd_K0Short = *(iter.Begin());
  TLine* lK0ROImin = (TLine*)iter.Next();
  TLine* lK0ROImax = (TLine*)iter.Next();
  TLine* lK0ExcludeMin = (TLine*)iter.Next();
  TLine* lK0ExcludeMax = (TLine*)iter.Next();
  fitBgd_K0Short->SetLineColor(4);
  Double_t K0Bgd, K0SigpBgd, K0Sig, K0Pur;
  K0Bgd = K0Info[0];
  K0SigpBgd = K0Info[1];
  K0Sig = K0Info[2];
  K0Pur = K0Info[3];


  

//-----------------------------------------------------------------------------
  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(1,2);
  //--------------------------------
  c1->cd(1);
  AntiLambdaPurity->Draw();
  fitBgd_AntiLambda->Draw("same");
  lALamROImin->Draw();
  lALamROImax->Draw();
  lALamExcludeMin->Draw();
  lALamExcludeMax->Draw();
  TPaveText *myALamText = new TPaveText(0.15,0.65,0.30,0.85,"NDC");
  char buffer[50];
  sprintf(buffer, "ALam Purity = %.2f\%", 100*ALamPur);
  myALamText->AddText(buffer);
  myALamText->Draw();
  //--------------------------------
  c1->cd(2);
  K0ShortPurity->Draw();
  fitBgd_K0Short->Draw("same");
  lK0ROImin->Draw();
  lK0ROImax->Draw();
  lK0ExcludeMin->Draw();
  lK0ExcludeMax->Draw();
  TPaveText *myK0Text = new TPaveText(0.15,0.65,0.30,0.85,"NDC");
  char buffer[50];
  sprintf(buffer, "K^{0}_{s} Purity = %.2f\%", 100.*K0Pur);
  myK0Text->AddText(buffer);
  myK0Text->Draw();

}
