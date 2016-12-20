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
  if(aCfCollection->GetEntries() != aNumCollection->GetEntries()) {cout "CfCollection and NumCollection have unequal number of entries!!!!!" << endl;}

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
void CombineAvgSepCfs()
{
  bool RebinCfs = true;

  TObjArray* AvgSepNumCollection_TrackPos_LamKchP = new TObjArray();
  TObjArray* AvgSepDenCollection_TrackPos_LamKchP = new TObjArray();
  TObjArray* AvgSepCfCollection_TrackPos_LamKchP = new TObjArray();
  //
  TObjArray* AvgSepNumCollection_TrackNeg_LamKchP = new TObjArray();
  TObjArray* AvgSepDenCollection_TrackNeg_LamKchP = new TObjArray();
  TObjArray* AvgSepCfCollection_TrackNeg_LamKchP = new TObjArray();


  TObjArray* AvgSepNumCollection_TrackPos_LamKchM = new TObjArray();
  TObjArray* AvgSepDenCollection_TrackPos_LamKchM = new TObjArray();
  TObjArray* AvgSepCfCollection_TrackPos_LamKchM = new TObjArray();
  //
  TObjArray* AvgSepNumCollection_TrackNeg_LamKchM = new TObjArray();
  TObjArray* AvgSepDenCollection_TrackNeg_LamKchM = new TObjArray();
  TObjArray* AvgSepCfCollection_TrackNeg_LamKchM = new TObjArray();


  TObjArray* AvgSepNumCollection_TrackPos_ALamKchP = new TObjArray();
  TObjArray* AvgSepDenCollection_TrackPos_ALamKchP = new TObjArray();
  TObjArray* AvgSepCfCollection_TrackPos_ALamKchP = new TObjArray();
  //
  TObjArray* AvgSepNumCollection_TrackNeg_ALamKchP = new TObjArray();
  TObjArray* AvgSepDenCollection_TrackNeg_ALamKchP = new TObjArray();
  TObjArray* AvgSepCfCollection_TrackNeg_ALamKchP = new TObjArray();


  TObjArray* AvgSepNumCollection_TrackPos_ALamKchM = new TObjArray();
  TObjArray* AvgSepDenCollection_TrackPos_ALamKchM = new TObjArray();
  TObjArray* AvgSepCfCollection_TrackPos_ALamKchM = new TObjArray();
  //
  TObjArray* AvgSepNumCollection_TrackNeg_ALamKchM = new TObjArray();
  TObjArray* AvgSepDenCollection_TrackNeg_ALamKchM = new TObjArray();
  TObjArray* AvgSepCfCollection_TrackNeg_ALamKchM = new TObjArray();


  //---------- First Analysis ----------//
  TFile File1("Resultsgrid_cLamcKch_CentBins_NEWmc_0010.root");

  TObjArray* AvgSepNumCollection_TrackPos_LamKchP_1 = (TObjArray*)File1.Get("fAvgSepNumCollection_TrackPos_LamKchP");
  TObjArray* AvgSepDenCollection_TrackPos_LamKchP_1 = (TObjArray*)File1.Get("fAvgSepDenCollection_TrackPos_LamKchP");
  TObjArray* AvgSepCfCollection_TrackPos_LamKchP_1 = (TObjArray*)File1.Get("fAvgSepCfCollection_TrackPos_LamKchP");
  //
  TObjArray* AvgSepNumCollection_TrackNeg_LamKchP_1 = (TObjArray*)File1.Get("fAvgSepNumCollection_TrackNeg_LamKchP");
  TObjArray* AvgSepDenCollection_TrackNeg_LamKchP_1 = (TObjArray*)File1.Get("fAvgSepDenCollection_TrackNeg_LamKchP");
  TObjArray* AvgSepCfCollection_TrackNeg_LamKchP_1 = (TObjArray*)File1.Get("fAvgSepCfCollection_TrackNeg_LamKchP");


  TObjArray* AvgSepNumCollection_TrackPos_LamKchM_1 = (TObjArray*)File1.Get("fAvgSepNumCollection_TrackPos_LamKchM");
  TObjArray* AvgSepDenCollection_TrackPos_LamKchM_1 = (TObjArray*)File1.Get("fAvgSepDenCollection_TrackPos_LamKchM");
  TObjArray* AvgSepCfCollection_TrackPos_LamKchM_1 = (TObjArray*)File1.Get("fAvgSepCfCollection_TrackPos_LamKchM");
  //
  TObjArray* AvgSepNumCollection_TrackNeg_LamKchM_1 = (TObjArray*)File1.Get("fAvgSepNumCollection_TrackNeg_LamKchM");
  TObjArray* AvgSepDenCollection_TrackNeg_LamKchM_1 = (TObjArray*)File1.Get("fAvgSepDenCollection_TrackNeg_LamKchM");
  TObjArray* AvgSepCfCollection_TrackNeg_LamKchM_1 = (TObjArray*)File1.Get("fAvgSepCfCollection_TrackNeg_LamKchM");


  TObjArray* AvgSepNumCollection_TrackPos_ALamKchP_1 = (TObjArray*)File1.Get("fAvgSepNumCollection_TrackPos_ALamKchP");
  TObjArray* AvgSepDenCollection_TrackPos_ALamKchP_1 = (TObjArray*)File1.Get("fAvgSepDenCollection_TrackPos_ALamKchP");
  TObjArray* AvgSepCfCollection_TrackPos_ALamKchP_1 = (TObjArray*)File1.Get("fAvgSepCfCollection_TrackPos_ALamKchP");
  //
  TObjArray* AvgSepNumCollection_TrackNeg_ALamKchP_1 = (TObjArray*)File1.Get("fAvgSepNumCollection_TrackNeg_ALamKchP");
  TObjArray* AvgSepDenCollection_TrackNeg_ALamKchP_1 = (TObjArray*)File1.Get("fAvgSepDenCollection_TrackNeg_ALamKchP");
  TObjArray* AvgSepCfCollection_TrackNeg_ALamKchP_1 = (TObjArray*)File1.Get("fAvgSepCfCollection_TrackNeg_ALamKchP");


  TObjArray* AvgSepNumCollection_TrackPos_ALamKchM_1 = (TObjArray*)File1.Get("fAvgSepNumCollection_TrackPos_ALamKchM");
  TObjArray* AvgSepDenCollection_TrackPos_ALamKchM_1 = (TObjArray*)File1.Get("fAvgSepDenCollection_TrackPos_ALamKchM");
  TObjArray* AvgSepCfCollection_TrackPos_ALamKchM_1 = (TObjArray*)File1.Get("fAvgSepCfCollection_TrackPos_ALamKchM");
  //
  TObjArray* AvgSepNumCollection_TrackNeg_ALamKchM_1 = (TObjArray*)File1.Get("fAvgSepNumCollection_TrackNeg_ALamKchM");
  TObjArray* AvgSepDenCollection_TrackNeg_ALamKchM_1 = (TObjArray*)File1.Get("fAvgSepDenCollection_TrackNeg_ALamKchM");
  TObjArray* AvgSepCfCollection_TrackNeg_ALamKchM_1 = (TObjArray*)File1.Get("fAvgSepCfCollection_TrackNeg_ALamKchM");


    //-----Add the historgrams to final collection
    for(int i=0; i<(AvgSepNumCollection_TrackPos_LamKchP_1->GetEntries()); i++) {AvgSepNumCollection_TrackPos_LamKchP->Add(AvgSepNumCollection_TrackPos_LamKchP_1->At(i));}
    for(int i=0; i<(AvgSepDenCollection_TrackPos_LamKchP_1->GetEntries()); i++) {AvgSepDenCollection_TrackPos_LamKchP->Add(AvgSepDenCollection_TrackPos_LamKchP_1->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackPos_LamKchP_1->GetEntries(); i++) {AvgSepCfCollection_TrackPos_LamKchP->Add(AvgSepCfCollection_TrackPos_LamKchP_1->At(i));}
    //
    for(int i=0; i<(AvgSepNumCollection_TrackNeg_LamKchP_1->GetEntries()); i++) {AvgSepNumCollection_TrackNeg_LamKchP->Add(AvgSepNumCollection_TrackNeg_LamKchP_1->At(i));}
    for(int i=0; i<(AvgSepDenCollection_TrackNeg_LamKchP_1->GetEntries()); i++) {AvgSepDenCollection_TrackNeg_LamKchP->Add(AvgSepDenCollection_TrackNeg_LamKchP_1->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackNeg_LamKchP_1->GetEntries(); i++) {AvgSepCfCollection_TrackNeg_LamKchP->Add(AvgSepCfCollection_TrackNeg_LamKchP_1->At(i));}


    for(int i=0; i<AvgSepNumCollection_TrackPos_LamKchM_1->GetEntries(); i++) {AvgSepNumCollection_TrackPos_LamKchM->Add(AvgSepNumCollection_TrackPos_LamKchM_1->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackPos_LamKchM_1->GetEntries(); i++) {AvgSepDenCollection_TrackPos_LamKchM->Add(AvgSepDenCollection_TrackPos_LamKchM_1->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackPos_LamKchM_1->GetEntries(); i++) {AvgSepCfCollection_TrackPos_LamKchM->Add(AvgSepCfCollection_TrackPos_LamKchM_1->At(i));}
    //
    for(int i=0; i<AvgSepNumCollection_TrackNeg_LamKchM_1->GetEntries(); i++) {AvgSepNumCollection_TrackNeg_LamKchM->Add(AvgSepNumCollection_TrackNeg_LamKchM_1->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackNeg_LamKchM_1->GetEntries(); i++) {AvgSepDenCollection_TrackNeg_LamKchM->Add(AvgSepDenCollection_TrackNeg_LamKchM_1->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackNeg_LamKchM_1->GetEntries(); i++) {AvgSepCfCollection_TrackNeg_LamKchM->Add(AvgSepCfCollection_TrackNeg_LamKchM_1->At(i));}


    for(int i=0; i<AvgSepNumCollection_TrackPos_ALamKchP_1->GetEntries(); i++) {AvgSepNumCollection_TrackPos_ALamKchP->Add(AvgSepNumCollection_TrackPos_ALamKchP_1->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackPos_ALamKchP_1->GetEntries(); i++) {AvgSepDenCollection_TrackPos_ALamKchP->Add(AvgSepDenCollection_TrackPos_ALamKchP_1->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackPos_ALamKchP_1->GetEntries(); i++) {AvgSepCfCollection_TrackPos_ALamKchP->Add(AvgSepCfCollection_TrackPos_ALamKchP_1->At(i));}
    //
    for(int i=0; i<AvgSepNumCollection_TrackNeg_ALamKchP_1->GetEntries(); i++) {AvgSepNumCollection_TrackNeg_ALamKchP->Add(AvgSepNumCollection_TrackNeg_ALamKchP_1->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackNeg_ALamKchP_1->GetEntries(); i++) {AvgSepDenCollection_TrackNeg_ALamKchP->Add(AvgSepDenCollection_TrackNeg_ALamKchP_1->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackNeg_ALamKchP_1->GetEntries(); i++) {AvgSepCfCollection_TrackNeg_ALamKchP->Add(AvgSepCfCollection_TrackNeg_ALamKchP_1->At(i));}


    for(int i=0; i<AvgSepNumCollection_TrackPos_ALamKchM_1->GetEntries(); i++) {AvgSepNumCollection_TrackPos_ALamKchM->Add(AvgSepNumCollection_TrackPos_ALamKchM_1->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackPos_ALamKchM_1->GetEntries(); i++) {AvgSepDenCollection_TrackPos_ALamKchM->Add(AvgSepDenCollection_TrackPos_ALamKchM_1->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackPos_ALamKchM_1->GetEntries(); i++) {AvgSepCfCollection_TrackPos_ALamKchM->Add(AvgSepCfCollection_TrackPos_ALamKchM_1->At(i));}
    //
    for(int i=0; i<AvgSepNumCollection_TrackNeg_ALamKchM_1->GetEntries(); i++) {AvgSepNumCollection_TrackNeg_ALamKchM->Add(AvgSepNumCollection_TrackNeg_ALamKchM_1->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackNeg_ALamKchM_1->GetEntries(); i++) {AvgSepDenCollection_TrackNeg_ALamKchM->Add(AvgSepDenCollection_TrackNeg_ALamKchM_1->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackNeg_ALamKchM_1->GetEntries(); i++) {AvgSepCfCollection_TrackNeg_ALamKchM->Add(AvgSepCfCollection_TrackNeg_ALamKchM_1->At(i));}

  File1.Close();


  //---------- Second Analysis ----------//
  TFile File2("Resultsgrid_cLamcKch_CentBins_NEWmcD_0010.root");

  TObjArray* AvgSepNumCollection_TrackPos_LamKchP_2 = (TObjArray*)File2.Get("fAvgSepNumCollection_TrackPos_LamKchP");
  TObjArray* AvgSepDenCollection_TrackPos_LamKchP_2 = (TObjArray*)File2.Get("fAvgSepDenCollection_TrackPos_LamKchP");
  TObjArray* AvgSepCfCollection_TrackPos_LamKchP_2 = (TObjArray*)File2.Get("fAvgSepCfCollection_TrackPos_LamKchP");
  //
  TObjArray* AvgSepNumCollection_TrackNeg_LamKchP_2 = (TObjArray*)File2.Get("fAvgSepNumCollection_TrackNeg_LamKchP");
  TObjArray* AvgSepDenCollection_TrackNeg_LamKchP_2 = (TObjArray*)File2.Get("fAvgSepDenCollection_TrackNeg_LamKchP");
  TObjArray* AvgSepCfCollection_TrackNeg_LamKchP_2 = (TObjArray*)File2.Get("fAvgSepCfCollection_TrackNeg_LamKchP");


  TObjArray* AvgSepNumCollection_TrackPos_LamKchM_2 = (TObjArray*)File2.Get("fAvgSepNumCollection_TrackPos_LamKchM");
  TObjArray* AvgSepDenCollection_TrackPos_LamKchM_2 = (TObjArray*)File2.Get("fAvgSepDenCollection_TrackPos_LamKchM");
  TObjArray* AvgSepCfCollection_TrackPos_LamKchM_2 = (TObjArray*)File2.Get("fAvgSepCfCollection_TrackPos_LamKchM");
  //
  TObjArray* AvgSepNumCollection_TrackNeg_LamKchM_2 = (TObjArray*)File2.Get("fAvgSepNumCollection_TrackNeg_LamKchM");
  TObjArray* AvgSepDenCollection_TrackNeg_LamKchM_2 = (TObjArray*)File2.Get("fAvgSepDenCollection_TrackNeg_LamKchM");
  TObjArray* AvgSepCfCollection_TrackNeg_LamKchM_2 = (TObjArray*)File2.Get("fAvgSepCfCollection_TrackNeg_LamKchM");


  TObjArray* AvgSepNumCollection_TrackPos_ALamKchP_2 = (TObjArray*)File2.Get("fAvgSepNumCollection_TrackPos_ALamKchP");
  TObjArray* AvgSepDenCollection_TrackPos_ALamKchP_2 = (TObjArray*)File2.Get("fAvgSepDenCollection_TrackPos_ALamKchP");
  TObjArray* AvgSepCfCollection_TrackPos_ALamKchP_2 = (TObjArray*)File2.Get("fAvgSepCfCollection_TrackPos_ALamKchP");
  //
  TObjArray* AvgSepNumCollection_TrackNeg_ALamKchP_2 = (TObjArray*)File2.Get("fAvgSepNumCollection_TrackNeg_ALamKchP");
  TObjArray* AvgSepDenCollection_TrackNeg_ALamKchP_2 = (TObjArray*)File2.Get("fAvgSepDenCollection_TrackNeg_ALamKchP");
  TObjArray* AvgSepCfCollection_TrackNeg_ALamKchP_2 = (TObjArray*)File2.Get("fAvgSepCfCollection_TrackNeg_ALamKchP");


  TObjArray* AvgSepNumCollection_TrackPos_ALamKchM_2 = (TObjArray*)File2.Get("fAvgSepNumCollection_TrackPos_ALamKchM");
  TObjArray* AvgSepDenCollection_TrackPos_ALamKchM_2 = (TObjArray*)File2.Get("fAvgSepDenCollection_TrackPos_ALamKchM");
  TObjArray* AvgSepCfCollection_TrackPos_ALamKchM_2 = (TObjArray*)File2.Get("fAvgSepCfCollection_TrackPos_ALamKchM");
  //
  TObjArray* AvgSepNumCollection_TrackNeg_ALamKchM_2 = (TObjArray*)File2.Get("fAvgSepNumCollection_TrackNeg_ALamKchM");
  TObjArray* AvgSepDenCollection_TrackNeg_ALamKchM_2 = (TObjArray*)File2.Get("fAvgSepDenCollection_TrackNeg_ALamKchM");
  TObjArray* AvgSepCfCollection_TrackNeg_ALamKchM_2 = (TObjArray*)File2.Get("fAvgSepCfCollection_TrackNeg_ALamKchM");


    //-----Add the historgrams to final collection
    for(int i=0; i<(AvgSepNumCollection_TrackPos_LamKchP_2->GetEntries()); i++) {AvgSepNumCollection_TrackPos_LamKchP->Add(AvgSepNumCollection_TrackPos_LamKchP_2->At(i));}
    for(int i=0; i<(AvgSepDenCollection_TrackPos_LamKchP_2->GetEntries()); i++) {AvgSepDenCollection_TrackPos_LamKchP->Add(AvgSepDenCollection_TrackPos_LamKchP_2->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackPos_LamKchP_2->GetEntries(); i++) {AvgSepCfCollection_TrackPos_LamKchP->Add(AvgSepCfCollection_TrackPos_LamKchP_2->At(i));}
    //
    for(int i=0; i<(AvgSepNumCollection_TrackNeg_LamKchP_2->GetEntries()); i++) {AvgSepNumCollection_TrackNeg_LamKchP->Add(AvgSepNumCollection_TrackNeg_LamKchP_2->At(i));}
    for(int i=0; i<(AvgSepDenCollection_TrackNeg_LamKchP_2->GetEntries()); i++) {AvgSepDenCollection_TrackNeg_LamKchP->Add(AvgSepDenCollection_TrackNeg_LamKchP_2->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackNeg_LamKchP_2->GetEntries(); i++) {AvgSepCfCollection_TrackNeg_LamKchP->Add(AvgSepCfCollection_TrackNeg_LamKchP_2->At(i));}


    for(int i=0; i<AvgSepNumCollection_TrackPos_LamKchM_2->GetEntries(); i++) {AvgSepNumCollection_TrackPos_LamKchM->Add(AvgSepNumCollection_TrackPos_LamKchM_2->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackPos_LamKchM_2->GetEntries(); i++) {AvgSepDenCollection_TrackPos_LamKchM->Add(AvgSepDenCollection_TrackPos_LamKchM_2->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackPos_LamKchM_2->GetEntries(); i++) {AvgSepCfCollection_TrackPos_LamKchM->Add(AvgSepCfCollection_TrackPos_LamKchM_2->At(i));}
    //
    for(int i=0; i<AvgSepNumCollection_TrackNeg_LamKchM_2->GetEntries(); i++) {AvgSepNumCollection_TrackNeg_LamKchM->Add(AvgSepNumCollection_TrackNeg_LamKchM_2->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackNeg_LamKchM_2->GetEntries(); i++) {AvgSepDenCollection_TrackNeg_LamKchM->Add(AvgSepDenCollection_TrackNeg_LamKchM_2->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackNeg_LamKchM_2->GetEntries(); i++) {AvgSepCfCollection_TrackNeg_LamKchM->Add(AvgSepCfCollection_TrackNeg_LamKchM_2->At(i));}


    for(int i=0; i<AvgSepNumCollection_TrackPos_ALamKchP_2->GetEntries(); i++) {AvgSepNumCollection_TrackPos_ALamKchP->Add(AvgSepNumCollection_TrackPos_ALamKchP_2->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackPos_ALamKchP_2->GetEntries(); i++) {AvgSepDenCollection_TrackPos_ALamKchP->Add(AvgSepDenCollection_TrackPos_ALamKchP_2->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackPos_ALamKchP_2->GetEntries(); i++) {AvgSepCfCollection_TrackPos_ALamKchP->Add(AvgSepCfCollection_TrackPos_ALamKchP_2->At(i));}
    //
    for(int i=0; i<AvgSepNumCollection_TrackNeg_ALamKchP_2->GetEntries(); i++) {AvgSepNumCollection_TrackNeg_ALamKchP->Add(AvgSepNumCollection_TrackNeg_ALamKchP_2->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackNeg_ALamKchP_2->GetEntries(); i++) {AvgSepDenCollection_TrackNeg_ALamKchP->Add(AvgSepDenCollection_TrackNeg_ALamKchP_2->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackNeg_ALamKchP_2->GetEntries(); i++) {AvgSepCfCollection_TrackNeg_ALamKchP->Add(AvgSepCfCollection_TrackNeg_ALamKchP_2->At(i));}


    for(int i=0; i<AvgSepNumCollection_TrackPos_ALamKchM_2->GetEntries(); i++) {AvgSepNumCollection_TrackPos_ALamKchM->Add(AvgSepNumCollection_TrackPos_ALamKchM_2->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackPos_ALamKchM_2->GetEntries(); i++) {AvgSepDenCollection_TrackPos_ALamKchM->Add(AvgSepDenCollection_TrackPos_ALamKchM_2->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackPos_ALamKchM_2->GetEntries(); i++) {AvgSepCfCollection_TrackPos_ALamKchM->Add(AvgSepCfCollection_TrackPos_ALamKchM_2->At(i));}
    //
    for(int i=0; i<AvgSepNumCollection_TrackNeg_ALamKchM_2->GetEntries(); i++) {AvgSepNumCollection_TrackNeg_ALamKchM->Add(AvgSepNumCollection_TrackNeg_ALamKchM_2->At(i));}
    for(int i=0; i<AvgSepDenCollection_TrackNeg_ALamKchM_2->GetEntries(); i++) {AvgSepDenCollection_TrackNeg_ALamKchM->Add(AvgSepDenCollection_TrackNeg_ALamKchM_2->At(i));}
    for(int i=0; i<AvgSepCfCollection_TrackNeg_ALamKchM_2->GetEntries(); i++) {AvgSepCfCollection_TrackNeg_ALamKchM->Add(AvgSepCfCollection_TrackNeg_ALamKchM_2->At(i));}

  File2.Close();


  //----------Build the Cfs
  int MinNormBinCF = 150;
  int MaxNormBinCF = 200;
    cout << "MinNormBinCF = " << MinNormBinCF << "  MaxNormBinCF = " << MaxNormBinCF << endl;

    //-----Rebin?-----
  if(RebinCfs)
  {
    int RebinFactor = 5;
    MinNormBinCF /= RebinFactor;
    MaxNormBinCF /= RebinFactor;

    RebinAll(RebinFactor,AvgSepCfCollection_TrackPos_LamKchP,AvgSepNumCollection_TrackPos_LamKchP,AvgSepDenCollection_TrackPos_LamKchP,MinNormBinCF,MaxNormBinCF);
    RebinAll(RebinFactor,AvgSepCfCollection_TrackNeg_LamKchP,AvgSepNumCollection_TrackNeg_LamKchP,AvgSepDenCollection_TrackNeg_LamKchP,MinNormBinCF,MaxNormBinCF);

    RebinAll(RebinFactor,AvgSepCfCollection_TrackPos_LamKchM,AvgSepNumCollection_TrackPos_LamKchM,AvgSepDenCollection_TrackPos_LamKchM,MinNormBinCF,MaxNormBinCF);
    RebinAll(RebinFactor,AvgSepCfCollection_TrackNeg_LamKchM,AvgSepNumCollection_TrackNeg_LamKchM,AvgSepDenCollection_TrackNeg_LamKchM,MinNormBinCF,MaxNormBinCF);

    RebinAll(RebinFactor,AvgSepCfCollection_TrackPos_ALamKchP,AvgSepNumCollection_TrackPos_ALamKchP,AvgSepDenCollection_TrackPos_ALamKchP,MinNormBinCF,MaxNormBinCF);
    RebinAll(RebinFactor,AvgSepCfCollection_TrackNeg_ALamKchP,AvgSepNumCollection_TrackNeg_ALamKchP,AvgSepDenCollection_TrackNeg_ALamKchP,MinNormBinCF,MaxNormBinCF);

    RebinAll(RebinFactor,AvgSepCfCollection_TrackPos_ALamKchM,AvgSepNumCollection_TrackPos_ALamKchM,AvgSepDenCollection_TrackPos_ALamKchM,MinNormBinCF,MaxNormBinCF);
    RebinAll(RebinFactor,AvgSepCfCollection_TrackNeg_ALamKchM,AvgSepNumCollection_TrackNeg_ALamKchM,AvgSepDenCollection_TrackNeg_ALamKchM,MinNormBinCF,MaxNormBinCF);

      cout << "MinNormBinCF = " << MinNormBinCF << "  MaxNormBinCF = " << MaxNormBinCF << endl;
  }
    //----------------

  TH1F* Cf_TrackPos_LamKchP = BuildCombinedCFs("Cf_TrackPos_LamKchP","Lam-K+ (Tot)",AvgSepCfCollection_TrackPos_LamKchP,AvgSepNumCollection_TrackPos_LamKchP,MinNormBinCF,MaxNormBinCF);
  TH1F* Cf_TrackNeg_LamKchP = BuildCombinedCFs("Cf_TrackNeg_LamKchP","Lam-K+ (Tot)",AvgSepCfCollection_TrackNeg_LamKchP,AvgSepNumCollection_TrackNeg_LamKchP,MinNormBinCF,MaxNormBinCF);

  TH1F* Cf_TrackPos_LamKchM = BuildCombinedCFs("Cf_TrackPos_LamKchM","Lam-K- (Tot)",AvgSepCfCollection_TrackPos_LamKchM,AvgSepNumCollection_TrackPos_LamKchM,MinNormBinCF,MaxNormBinCF);
  TH1F* Cf_TrackNeg_LamKchM = BuildCombinedCFs("Cf_TrackNeg_LamKchM","Lam-K- (Tot)",AvgSepCfCollection_TrackNeg_LamKchM,AvgSepNumCollection_TrackNeg_LamKchM,MinNormBinCF,MaxNormBinCF);

  TH1F* Cf_TrackPos_ALamKchP = BuildCombinedCFs("Cf_TrackPos_ALamKchP","ALam-K+ (Tot)",AvgSepCfCollection_TrackPos_ALamKchP,AvgSepNumCollection_TrackPos_ALamKchP,MinNormBinCF,MaxNormBinCF);
  TH1F* Cf_TrackNeg_ALamKchP = BuildCombinedCFs("Cf_TrackNeg_ALamKchP","ALam-K+ (Tot)",AvgSepCfCollection_TrackNeg_ALamKchP,AvgSepNumCollection_TrackNeg_ALamKchP,MinNormBinCF,MaxNormBinCF);

  TH1F* Cf_TrackPos_ALamKchM = BuildCombinedCFs("Cf_TrackPos_ALamKchM","ALam-K- (Tot)",AvgSepCfCollection_TrackPos_ALamKchM,AvgSepNumCollection_TrackPos_ALamKchM,MinNormBinCF,MaxNormBinCF);
  TH1F* Cf_TrackNeg_ALamKchM = BuildCombinedCFs("Cf_TrackNeg_ALamKchM","ALam-K- (Tot)",AvgSepCfCollection_TrackNeg_ALamKchM,AvgSepNumCollection_TrackNeg_ALamKchM,MinNormBinCF,MaxNormBinCF);

  //----------Draw everything

  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  //------------------------------------------------------------------
  //-------LamKchP------------
  TCanvas* aCanvasLamKchP = new TCanvas("aCanvasLamKchP","aCanvasLamKchP");
  aCanvasLamKchP->cd();
  aCanvasLamKchP->Divide(1,2);

  aCanvasLamKchP->cd(1);
  Cf_TrackPos_LamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  Cf_TrackPos_LamKchP->SetTitle("p(#Lambda) - K+");
  Cf_TrackPos_LamKchP->Draw();
  line->Draw();

  aCanvasLamKchP->cd(2);
  Cf_TrackNeg_LamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  Cf_TrackNeg_LamKchP->SetTitle("#pi^{-}(#Lambda) - K+");
  Cf_TrackNeg_LamKchP->Draw();
  line->Draw();

  //-------LamKchM------------
  TCanvas* aCanvasLamKchM = new TCanvas("aCanvasLamKchM","aCanvasLamKchM");
  aCanvasLamKchM->cd();
  aCanvasLamKchM->Divide(1,2);

  aCanvasLamKchM->cd(1);
  Cf_TrackPos_LamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  Cf_TrackPos_LamKchM->SetTitle("p(#Lambda) - K-");
  Cf_TrackPos_LamKchM->Draw();
  line->Draw();

  aCanvasLamKchM->cd(2);
  Cf_TrackNeg_LamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  Cf_TrackNeg_LamKchM->SetTitle("#pi^{-}(#Lambda) - K-");
  Cf_TrackNeg_LamKchM->Draw();
  line->Draw();

  //-------ALamKchP------------
  TCanvas* aCanvasALamKchP = new TCanvas("aCanvasALamKchP","aCanvasALamKchP");
  aCanvasALamKchP->cd();
  aCanvasALamKchP->Divide(1,2);

  aCanvasALamKchP->cd(1);
  Cf_TrackPos_ALamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  Cf_TrackPos_ALamKchP->SetTitle("#pi^{+}(#bar{#Lambda}) - K+");
  Cf_TrackPos_ALamKchP->Draw();
  line->Draw();

  aCanvasALamKchP->cd(2);
  Cf_TrackNeg_ALamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  Cf_TrackNeg_ALamKchP->SetTitle("#bar{p}(#bar{#Lambda}) - K+");
  Cf_TrackNeg_ALamKchP->Draw();
  line->Draw();

  //-------ALamKchM------------
  TCanvas* aCanvasALamKchM = new TCanvas("aCanvasALamKchM","aCanvasALamKchM");
  aCanvasALamKchM->cd();
  aCanvasALamKchM->Divide(1,2);

  aCanvasALamKchM->cd(1);
  Cf_TrackPos_ALamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  Cf_TrackPos_ALamKchM->SetTitle("#pi^{+}(#bar{#Lambda}) - K-");
  Cf_TrackPos_ALamKchM->Draw();
  line->Draw();

  aCanvasALamKchM->cd(2);
  Cf_TrackNeg_ALamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  Cf_TrackNeg_ALamKchM->SetTitle("#bar{p}(#bar{#Lambda}) - K-");
  Cf_TrackNeg_ALamKchM->Draw();
  line->Draw();
}
