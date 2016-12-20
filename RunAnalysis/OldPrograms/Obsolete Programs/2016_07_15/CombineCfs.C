#include "TH1F.h"
#include "TObjArray.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TLine.h"
#include "TStyle.h"

#include <iostream>

//________________________________________________________________________________________________________________
TH1F* BuildCombinedCFs(TString aReturnName, TString aReturnTitle, TObjArray* aCfCollection, TObjArray* aNumCollection, int aMinNormBin, int aMaxNormBin)
{
  double scale = 0.;
  int counter = 0;
  double temp = 0.;

  int SizeOfCfCollection = aCfCollection->GetEntries();
  int SizeOfNumCollection = aNumCollection->GetEntries();

  cout << "Cf Collection size(!): " << SizeOfCfCollection << endl;
  cout << "Num Collection size(!): " << SizeOfNumCollection << endl;

  if(SizeOfCfCollection != SizeOfNumCollection) {cout << "ERROR: In BuildCombinedCFs, the CfCollection and NumCollection ARE NOT EQUAL IN SIZE!!!!" << endl;}

  TH1F* ReturnCf = (TH1F*)(aCfCollection->At(0))->Clone(aReturnName);
  ReturnCf->Sumw2();
  ReturnCf->SetTitle(aReturnTitle);
    //cout << "Name: " << ReturnCf->GetName() << endl;

  TH1F* Num1 = (TH1F*)(aNumCollection->At(0));
  temp = Num1->Integral(aMinNormBin,aMaxNormBin);
    //cout << "Name: " << Num1->GetName() << "  NumScale: " << Num1->Integral(aMinNormBin,aMaxNormBin) << endl;

  scale+=temp;
  counter++;

  ReturnCf->Scale(temp);

  for(int i=1; i<SizeOfCfCollection; i++)
  {
    //cout << "Name: " << ((TH1F*)aCfCollection->At(i))->GetName() << endl;
    temp = ((TH1F*)aNumCollection->At(i))->Integral(aMinNormBin,aMaxNormBin);
    //cout << "Name: " << ((TH1F*)aNumCollection->At(i))->GetName() << " NumScale: " << ((TH1F*)aNumCollection->At(i))->Integral(aMinNormBin,aMaxNormBin) << endl;
    ReturnCf->Add((TH1F*)aCfCollection->At(i),temp);
    scale += temp;
    counter ++;
  }
  //cout << "SCALE = " << scale << endl;
  cout << "counter = " << counter << endl;
  ReturnCf->Scale(1./scale);

  return ReturnCf;

}

//________________________________________________________________________________________________________________
TH1F* buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int aMinNormBin, int aMaxNormBin)
{
//  cout << "For hist "<< Num->GetName()<<", num pair = " << Num->Integral(1,Num->FindBin(.05)) << endl;
  double NumScale = Num->Integral(aMinNormBin,aMaxNormBin);
  double DenScale = Denom->Integral(aMinNormBin,aMaxNormBin);

  TH1F* CF = (TH1F*)Num->Clone(name);
  CF->Sumw2();
  CF->Divide(Denom);
  CF->Scale(DenScale/NumScale);
  CF->SetTitle(title);

  return CF;
}

//________________________________________________________________________________________________________________
void RebinAll(int aRebinFactor, TObjArray* aCfCollection, TObjArray* aNumCollection, TObjArray* aDenCollection, int aMinNormBin, int aMaxNormBin)
{
  if(aCfCollection->GetEntries() != aNumCollection->GetEntries()) {cout << "CfCollection and NumCollection have unequal number of entries!!!!!" << endl;}

  vector<TString> fVectorOfHistoNames;
  for(int i=0; i<aCfCollection->GetEntries(); i++) {fVectorOfHistoNames.push_back(((TH1F*)aCfCollection->At(i))->GetName());}

  aCfCollection->Clear();
  cout << "After clear, aCfCollection now has " << aCfCollection->GetEntries() << " entries." << endl;

  for(int i=0; i<aNumCollection->GetEntries(); i++)
  {
    ((TH1F*)aNumCollection->At(i))->Rebin(aRebinFactor);
      ((TH1F*)aNumCollection->At(i))->Scale(1./aRebinFactor);

    ((TH1F*)aDenCollection->At(i))->Rebin(aRebinFactor);
      ((TH1F*)aDenCollection->At(i))->Scale(1./aRebinFactor);

    aCfCollection->Add(buildCF(fVectorOfHistoNames[i],fVectorOfHistoNames[i],((TH1F*)aNumCollection->At(i)),((TH1F*)aDenCollection->At(i)),aMinNormBin,aMaxNormBin));

  }
  cout << "Now, aCfCollection now has " << aCfCollection->GetEntries() << " entries." << endl;
  cout << "With names: " << endl;
  for(int i=0; i<aCfCollection->GetEntries(); i++) {cout << ((TH1F*)aCfCollection->At(i))->GetName() << endl;}

}


//________________________________________________________________________________________________________________
void CombineCfs()
{
  bool RebinCfs = true;
  bool DrawData = true;

  TObjArray* NumCollection_LamKchP = new TObjArray();
  TObjArray* DenCollection_LamKchP = new TObjArray();
  TObjArray* CfCollection_LamKchP = new TObjArray();

  TObjArray* NumCollection_LamKchM = new TObjArray();
  TObjArray* DenCollection_LamKchM = new TObjArray();
  TObjArray* CfCollection_LamKchM = new TObjArray();

  TObjArray* NumCollection_ALamKchP = new TObjArray();
  TObjArray* DenCollection_ALamKchP = new TObjArray();
  TObjArray* CfCollection_ALamKchP = new TObjArray();

  TObjArray* NumCollection_ALamKchM = new TObjArray();
  TObjArray* DenCollection_ALamKchM = new TObjArray();
  TObjArray* CfCollection_ALamKchM = new TObjArray();



  //---------- First Analysis ----------//
  TFile File1("Resultsgrid_cLamcKch_CentBins_NEWASmc_0010.root");

  TObjArray* NumCollection_LamKchP_1 = (TObjArray*)File1.Get("fNumCollection_LamKchP");
  TObjArray* DenCollection_LamKchP_1 = (TObjArray*)File1.Get("fDenCollection_LamKchP");
  TObjArray* CfCollection_LamKchP_1 = (TObjArray*)File1.Get("fCfCollection_LamKchP");

  TObjArray* NumCollection_LamKchM_1 = (TObjArray*)File1.Get("fNumCollection_LamKchM");
  TObjArray* DenCollection_LamKchM_1 = (TObjArray*)File1.Get("fDenCollection_LamKchM");
  TObjArray* CfCollection_LamKchM_1 = (TObjArray*)File1.Get("fCfCollection_LamKchM");

  TObjArray* NumCollection_ALamKchP_1 = (TObjArray*)File1.Get("fNumCollection_ALamKchP");
  TObjArray* DenCollection_ALamKchP_1 = (TObjArray*)File1.Get("fDenCollection_ALamKchP");
  TObjArray* CfCollection_ALamKchP_1 = (TObjArray*)File1.Get("fCfCollection_ALamKchP");

  TObjArray* NumCollection_ALamKchM_1 = (TObjArray*)File1.Get("fNumCollection_ALamKchM");
  TObjArray* DenCollection_ALamKchM_1 = (TObjArray*)File1.Get("fDenCollection_ALamKchM");
  TObjArray* CfCollection_ALamKchM_1 = (TObjArray*)File1.Get("fCfCollection_ALamKchM");


    //-----Add the historgrams to final collection
    for(int i=0; i<(NumCollection_LamKchP_1->GetEntries()); i++) {NumCollection_LamKchP->Add(NumCollection_LamKchP_1->At(i));}
    for(int i=0; i<(DenCollection_LamKchP_1->GetEntries()); i++) {DenCollection_LamKchP->Add(DenCollection_LamKchP_1->At(i));}
    for(int i=0; i<CfCollection_LamKchP_1->GetEntries(); i++) {CfCollection_LamKchP->Add(CfCollection_LamKchP_1->At(i));}

    for(int i=0; i<NumCollection_LamKchM_1->GetEntries(); i++) {NumCollection_LamKchM->Add(NumCollection_LamKchM_1->At(i));}
    for(int i=0; i<DenCollection_LamKchM_1->GetEntries(); i++) {DenCollection_LamKchM->Add(DenCollection_LamKchM_1->At(i));}
    for(int i=0; i<CfCollection_LamKchM_1->GetEntries(); i++) {CfCollection_LamKchM->Add(CfCollection_LamKchM_1->At(i));}

    for(int i=0; i<NumCollection_ALamKchP_1->GetEntries(); i++) {NumCollection_ALamKchP->Add(NumCollection_ALamKchP_1->At(i));}
    for(int i=0; i<DenCollection_ALamKchP_1->GetEntries(); i++) {DenCollection_ALamKchP->Add(DenCollection_ALamKchP_1->At(i));}
    for(int i=0; i<CfCollection_ALamKchP_1->GetEntries(); i++) {CfCollection_ALamKchP->Add(CfCollection_ALamKchP_1->At(i));}

    for(int i=0; i<NumCollection_ALamKchM_1->GetEntries(); i++) {NumCollection_ALamKchM->Add(NumCollection_ALamKchM_1->At(i));}
    for(int i=0; i<DenCollection_ALamKchM_1->GetEntries(); i++) {DenCollection_ALamKchM->Add(DenCollection_ALamKchM_1->At(i));}
    for(int i=0; i<CfCollection_ALamKchM_1->GetEntries(); i++) {CfCollection_ALamKchM->Add(CfCollection_ALamKchM_1->At(i));}

  File1.Close();


  //---------- Second Analysis ----------//
  TFile File2("Resultsgrid_cLamcKch_CentBins_NEWASmcD_0010.root");

  TObjArray* NumCollection_LamKchP_2 = (TObjArray*)File2.Get("fNumCollection_LamKchP");
  TObjArray* DenCollection_LamKchP_2 = (TObjArray*)File2.Get("fDenCollection_LamKchP");
  TObjArray* CfCollection_LamKchP_2 = (TObjArray*)File2.Get("fCfCollection_LamKchP");

  TObjArray* NumCollection_LamKchM_2 = (TObjArray*)File2.Get("fNumCollection_LamKchM");
  TObjArray* DenCollection_LamKchM_2 = (TObjArray*)File2.Get("fDenCollection_LamKchM");
  TObjArray* CfCollection_LamKchM_2 = (TObjArray*)File2.Get("fCfCollection_LamKchM");

  TObjArray* NumCollection_ALamKchP_2 = (TObjArray*)File2.Get("fNumCollection_ALamKchP");
  TObjArray* DenCollection_ALamKchP_2 = (TObjArray*)File2.Get("fDenCollection_ALamKchP");
  TObjArray* CfCollection_ALamKchP_2 = (TObjArray*)File2.Get("fCfCollection_ALamKchP");

  TObjArray* NumCollection_ALamKchM_2 = (TObjArray*)File2.Get("fNumCollection_ALamKchM");
  TObjArray* DenCollection_ALamKchM_2 = (TObjArray*)File2.Get("fDenCollection_ALamKchM");
  TObjArray* CfCollection_ALamKchM_2 = (TObjArray*)File2.Get("fCfCollection_ALamKchM");


    //-----Add the historgrams to final collection
    for(int i=0; i<(NumCollection_LamKchP_2->GetEntries()); i++) {NumCollection_LamKchP->Add(NumCollection_LamKchP_2->At(i));}
    for(int i=0; i<(DenCollection_LamKchP_2->GetEntries()); i++) {DenCollection_LamKchP->Add(DenCollection_LamKchP_2->At(i));}
    for(int i=0; i<CfCollection_LamKchP_2->GetEntries(); i++) {CfCollection_LamKchP->Add(CfCollection_LamKchP_2->At(i));}

    for(int i=0; i<NumCollection_LamKchM_2->GetEntries(); i++) {NumCollection_LamKchM->Add(NumCollection_LamKchM_2->At(i));}
    for(int i=0; i<DenCollection_LamKchM_2->GetEntries(); i++) {DenCollection_LamKchM->Add(DenCollection_LamKchM_2->At(i));}
    for(int i=0; i<CfCollection_LamKchM_2->GetEntries(); i++) {CfCollection_LamKchM->Add(CfCollection_LamKchM_2->At(i));}

    for(int i=0; i<NumCollection_ALamKchP_2->GetEntries(); i++) {NumCollection_ALamKchP->Add(NumCollection_ALamKchP_2->At(i));}
    for(int i=0; i<DenCollection_ALamKchP_2->GetEntries(); i++) {DenCollection_ALamKchP->Add(DenCollection_ALamKchP_2->At(i));}
    for(int i=0; i<CfCollection_ALamKchP_2->GetEntries(); i++) {CfCollection_ALamKchP->Add(CfCollection_ALamKchP_2->At(i));}

    for(int i=0; i<NumCollection_ALamKchM_2->GetEntries(); i++) {NumCollection_ALamKchM->Add(NumCollection_ALamKchM_2->At(i));}
    for(int i=0; i<DenCollection_ALamKchM_2->GetEntries(); i++) {DenCollection_ALamKchM->Add(DenCollection_ALamKchM_2->At(i));}
    for(int i=0; i<CfCollection_ALamKchM_2->GetEntries(); i++) {CfCollection_ALamKchM->Add(CfCollection_ALamKchM_2->At(i));}

  File2.Close();


  //----------Build the Cfs
  int MinNormBinCF = 60;
  int MaxNormBinCF = 75;
    cout << "MinNormBinCF = " << MinNormBinCF << "  MaxNormBinCF = " << MaxNormBinCF << endl;

    //-----Rebin?-----
  if(RebinCfs)
  {
    int RebinFactor = 3;
    MinNormBinCF /= RebinFactor;
    MaxNormBinCF /= RebinFactor;
    RebinAll(RebinFactor,CfCollection_LamKchP,NumCollection_LamKchP,DenCollection_LamKchP,MinNormBinCF,MaxNormBinCF);
    RebinAll(RebinFactor,CfCollection_LamKchM,NumCollection_LamKchM,DenCollection_LamKchM,MinNormBinCF,MaxNormBinCF);
    RebinAll(RebinFactor,CfCollection_ALamKchP,NumCollection_ALamKchP,DenCollection_ALamKchP,MinNormBinCF,MaxNormBinCF);
    RebinAll(RebinFactor,CfCollection_ALamKchM,NumCollection_ALamKchM,DenCollection_ALamKchM,MinNormBinCF,MaxNormBinCF);

      cout << "MinNormBinCF = " << MinNormBinCF << "  MaxNormBinCF = " << MaxNormBinCF << endl;
  }
    //----------------

  TH1F* Cf_LamKchP = BuildCombinedCFs("Cf_LamKchP","Lam-K+ (Tot)",CfCollection_LamKchP,NumCollection_LamKchP,MinNormBinCF,MaxNormBinCF);
  TH1F* Cf_LamKchM = BuildCombinedCFs("Cf_LamKchM","Lam-K- (Tot)",CfCollection_LamKchM,NumCollection_LamKchM,MinNormBinCF,MaxNormBinCF);
  TH1F* Cf_ALamKchP = BuildCombinedCFs("Cf_ALamKchP","ALam-K+ (Tot)",CfCollection_ALamKchP,NumCollection_ALamKchP,MinNormBinCF,MaxNormBinCF);
  TH1F* Cf_ALamKchM = BuildCombinedCFs("Cf_ALamKchM","ALam-K- (Tot)",CfCollection_ALamKchM,NumCollection_ALamKchM,MinNormBinCF,MaxNormBinCF);


  //----------Draw everything
  TCanvas* c1_0010 = new TCanvas("c1_0010","Plotting Canvas1_0010",1400,500);
  gStyle->SetOptStat(0);

  TAxis *xax1 = Cf_LamKchP->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
  TAxis *yax1 = Cf_LamKchP->GetYaxis();
    yax1->SetRangeUser(0.9,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();

  TAxis *xax2 = Cf_LamKchM->GetXaxis();
    xax2->SetTitle("k* (GeV/c)");
    xax2->SetTitleSize(0.05);
    xax2->SetTitleOffset(1.0);
    //xax2->CenterTitle();
  TAxis *yax2 = Cf_LamKchM->GetYaxis();
    yax2->SetRangeUser(0.9,1.1);
    yax2->SetTitle("C(k*)");
    yax2->SetTitleSize(0.05);
    yax2->SetTitleOffset(1.0);
    yax2->CenterTitle();

  TAxis *xax3 = Cf_ALamKchP->GetXaxis();
    xax3->SetTitle("k* (GeV/c)");
    xax3->SetTitleSize(0.05);
    xax3->SetTitleOffset(1.0);
    //xax3->CenterTitle();
  TAxis *yax3 = Cf_ALamKchP->GetYaxis();
    yax3->SetRangeUser(0.9,1.1);
    yax3->SetTitle("C(k*)");
    yax3->SetTitleSize(0.05);
    yax3->SetTitleOffset(1.0);
    yax3->CenterTitle();

  TAxis *xax4 = Cf_ALamKchM->GetXaxis();
    xax4->SetTitle("k* (GeV/c)");
    xax4->SetTitleSize(0.05);
    xax4->SetTitleOffset(1.0);
    //xax4->CenterTitle();
  TAxis *yax4 = Cf_ALamKchM->GetYaxis();
    yax4->SetRangeUser(0.9,1.1);
    yax4->SetTitle("C(k*)");
    yax4->SetTitleSize(0.05);
    yax4->SetTitleOffset(1.0);
    yax4->CenterTitle();

  //------------------------------------------------------
  Cf_LamKchP->SetMarkerStyle(4);
  Cf_LamKchP->SetMarkerSize(0.75);
  Cf_LamKchP->SetMarkerColor(2);
  Cf_LamKchP->SetLineColor(2);
  Cf_LamKchP->SetTitle("#LambdaK+ & #LambdaK-");

  Cf_LamKchM->SetMarkerStyle(4);
  Cf_LamKchM->SetMarkerSize(0.75);
  Cf_LamKchM->SetMarkerColor(4);
  Cf_LamKchM->SetLineColor(4);
  //Cf_LamKchM->SetTitle("#Lambda - K-");

  Cf_ALamKchP->SetMarkerStyle(4);
  Cf_ALamKchP->SetMarkerSize(0.75);
  Cf_ALamKchP->SetMarkerColor(4);
  Cf_ALamKchP->SetLineColor(4);
  Cf_ALamKchP->SetTitle("#bar{#Lambda}K+ & #bar{#Lambda}K-");

  Cf_ALamKchM->SetMarkerStyle(4);
  Cf_ALamKchM->SetMarkerSize(0.75);
  Cf_ALamKchM->SetMarkerColor(2);
  Cf_ALamKchM->SetLineColor(2);
  //Cf_ALamKchM->SetTitle("#bar{#Lambda} - K-");

  //------------------------------------------------------
  if(DrawData)
  {
    TFile DataFile("Resultsgrid_cLamcKch_CentBins_NEWAS_0010.root");

    TH1F* DataCf_LamKchP = (TH1F*)DataFile.Get("fCf_LamKchP_Tot");
    TH1F* DataCf_LamKchM = (TH1F*)DataFile.Get("fCf_LamKchM_Tot");
    TH1F* DataCf_ALamKchP = (TH1F*)DataFile.Get("fCf_ALamKchP_Tot");
    TH1F* DataCf_ALamKchM = (TH1F*)DataFile.Get("fCf_ALamKchM_Tot");

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //IMPORTANT:  without SetDirectory(0), the histograms are not drawn!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    DataCf_LamKchP->SetDirectory(0);
    DataCf_LamKchM->SetDirectory(0);
    DataCf_ALamKchP->SetDirectory(0);
    DataCf_ALamKchM->SetDirectory(0);

    DataCf_LamKchP->SetMarkerStyle(20);
    DataCf_LamKchP->SetMarkerSize(0.75);
    DataCf_LamKchP->SetMarkerColor(2);
    DataCf_LamKchP->SetLineColor(2);
    DataCf_LamKchP->SetTitle("#LambdaK+ & #LambdaK-");

    DataCf_LamKchM->SetMarkerStyle(20);
    DataCf_LamKchM->SetMarkerSize(0.75);
    DataCf_LamKchM->SetMarkerColor(4);
    DataCf_LamKchM->SetLineColor(4);
    //DataCf_LamKchM->SetTitle("#Lambda - K-");

    DataCf_ALamKchP->SetMarkerStyle(20);
    DataCf_ALamKchP->SetMarkerSize(0.75);
    DataCf_ALamKchP->SetMarkerColor(4);
    DataCf_ALamKchP->SetLineColor(4);
    DataCf_ALamKchP->SetTitle("#bar{#Lambda}K+ & #bar{#Lambda}K-");

    DataCf_ALamKchM->SetMarkerStyle(20);
    DataCf_ALamKchM->SetMarkerSize(0.75);
    DataCf_ALamKchM->SetMarkerColor(2);
    DataCf_ALamKchM->SetLineColor(2);
    //DataCf_ALamKchM->SetTitle("#bar{#Lambda} - K-");
  }

  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);

  c1_0010->Divide(2,1);

  c1_0010->cd(1);
  Cf_LamKchP->DrawCopy();
  Cf_LamKchM->Draw("same");
  if(DrawData)
  {
    DataCf_LamKchP->Draw("same");
    DataCf_LamKchM->Draw("same");
  }
  line->Draw();
  TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
  leg1->SetFillColor(0);
  leg1->AddEntry(Cf_LamKchP, "#LambdaK+","lp");
  leg1->AddEntry(Cf_LamKchM, "#LambdaK-","lp");
  leg1->Draw();


  c1_0010->cd(2);
  Cf_ALamKchP->DrawCopy();
  Cf_ALamKchM->Draw("same");
  if(DrawData)
  {
    DataCf_ALamKchP->Draw("same");
    DataCf_ALamKchM->Draw("same");
  }
  line->Draw();
  TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
  leg2->SetFillColor(0);
  leg2->AddEntry(Cf_ALamKchM, "#bar{#Lambda}K-","lp");
  leg2->AddEntry(Cf_ALamKchP, "#bar{#Lambda}K+","lp");
  leg2->Draw();

  c1_0010->SaveAs("LamKch_DatavMC1.pdf");

  if(DrawData)
  {
    TCanvas* c2_0010 = new TCanvas("c2_0010","Plotting Canvas2_0010");
    c2_0010->Divide(2,2);

    c2_0010->cd(1);
    Cf_LamKchP->SetTitle("#LambdaK+");
    Cf_LamKchP->Draw();
    DataCf_LamKchP->Draw("same");

    c2_0010->cd(2);
    Cf_ALamKchM->SetTitle("#bar{#Lambda}K-");
    Cf_ALamKchM->Draw();
    DataCf_ALamKchM->Draw("same");

    c2_0010->cd(3);
    Cf_LamKchM->SetTitle("#LambdaK-");
    Cf_LamKchM->Draw();
    DataCf_LamKchM->Draw("same");

    c2_0010->cd(4);
    Cf_ALamKchP->SetTitle("#bar{#Lambda}K+");
    Cf_ALamKchP->Draw();
    DataCf_ALamKchP->Draw("same");

    c2_0010->SaveAs("LamKch_DatavMC2.pdf");


  }

}
