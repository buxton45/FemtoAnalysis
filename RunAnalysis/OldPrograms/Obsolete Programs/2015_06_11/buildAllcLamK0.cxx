///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamK0                                                        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "buildAllcLamK0.h"

#ifdef __ROOT__
ClassImp(buildAllcLamK0)
#endif




//____________________________
buildAllcLamK0::buildAllcLamK0():
  //KStar CF------------------
  fMinNormBinCF(60),
  fMaxNormBinCF(75),

  fNumLamK0Bp1(0),
  fNumLamK0Bp2(0),
  fNumLamK0Bm1(0),
  fNumLamK0Bm2(0),
  fNumLamK0Bm3(0),
  fNumList_LamK0_BpTot(0),
  fNumList_LamK0_BmTot(0),
  fNumList_LamK0_Tot(0),
  fDenLamK0Bp1(0),
  fDenLamK0Bp2(0),
  fDenLamK0Bm1(0),
  fDenLamK0Bm2(0),
  fDenLamK0Bm3(0),
  fDenList_LamK0_BpTot(0),
  fDenList_LamK0_BmTot(0),
  fDenList_LamK0_Tot(0),
  fCfLamK0Bp1(0),
  fCfLamK0Bp2(0),
  fCfLamK0Bm1(0),
  fCfLamK0Bm2(0),
  fCfLamK0Bm3(0),
  fCfList_LamK0_BpTot(0),
  fCfList_LamK0_BmTot(0),
  fCfList_LamK0_Tot(0),
  fCfLamK0BpTot(0),
  fCfLamK0BmTot(0),
  fCfLamK0Tot(0),

  fNumALamK0Bp1(0),
  fNumALamK0Bp2(0),
  fNumALamK0Bm1(0),
  fNumALamK0Bm2(0),
  fNumALamK0Bm3(0),
  fNumList_ALamK0_BpTot(0),
  fNumList_ALamK0_BmTot(0),
  fNumList_ALamK0_Tot(0),
  fDenALamK0Bp1(0),
  fDenALamK0Bp2(0),
  fDenALamK0Bm1(0),
  fDenALamK0Bm2(0),
  fDenALamK0Bm3(0),
  fDenList_ALamK0_BpTot(0),
  fDenList_ALamK0_BmTot(0),
  fDenList_ALamK0_Tot(0),
  fCfALamK0Bp1(0),
  fCfALamK0Bp2(0),
  fCfALamK0Bm1(0),
  fCfALamK0Bm2(0),
  fCfALamK0Bm3(0),
  fCfList_ALamK0_BpTot(0),
  fCfList_ALamK0_BmTot(0),
  fCfList_ALamK0_Tot(0),
  fCfALamK0BpTot(0),
  fCfALamK0BmTot(0),
  fCfALamK0Tot(0),

  //Average Separation CF------------------
  fMinNormBinAvgSepCF(150),
  fMaxNormBinAvgSepCF(200),
  //_____ ++ ___________________
  fNumPosPosAvgSepCfLamK0Bp1(0),
  fNumPosPosAvgSepCfLamK0Bp2(0),
  fNumPosPosAvgSepCfLamK0Bm1(0),
  fNumPosPosAvgSepCfLamK0Bm2(0),
  fNumPosPosAvgSepCfLamK0Bm3(0),
  fNumPosPosAvgSepCfList_LamK0_BpTot(0),
  fNumPosPosAvgSepCfList_LamK0_BmTot(0),
  fNumPosPosAvgSepCfList_LamK0_Tot(0),
  fDenPosPosAvgSepCfLamK0Bp1(0),
  fDenPosPosAvgSepCfLamK0Bp2(0),
  fDenPosPosAvgSepCfLamK0Bm1(0),
  fDenPosPosAvgSepCfLamK0Bm2(0),
  fDenPosPosAvgSepCfLamK0Bm3(0),
  fDenPosPosAvgSepCfList_LamK0_BpTot(0),
  fDenPosPosAvgSepCfList_LamK0_BmTot(0),
  fDenPosPosAvgSepCfList_LamK0_Tot(0),
  fCfPosPosAvgSepCfLamK0Bp1(0),
  fCfPosPosAvgSepCfLamK0Bp2(0),
  fCfPosPosAvgSepCfLamK0Bm1(0),
  fCfPosPosAvgSepCfLamK0Bm2(0),
  fCfPosPosAvgSepCfLamK0Bm3(0),
  fCfPosPosAvgSepCfList_LamK0_BpTot(0),
  fCfPosPosAvgSepCfList_LamK0_BmTot(0),
  fCfPosPosAvgSepCfList_LamK0_Tot(0),
  fCfPosPosAvgSepCfLamK0BpTot(0),
  fCfPosPosAvgSepCfLamK0BmTot(0),
  fCfPosPosAvgSepCfLamK0Tot(0),

  fNumPosPosAvgSepCfALamK0Bp1(0),
  fNumPosPosAvgSepCfALamK0Bp2(0),
  fNumPosPosAvgSepCfALamK0Bm1(0),
  fNumPosPosAvgSepCfALamK0Bm2(0),
  fNumPosPosAvgSepCfALamK0Bm3(0),
  fNumPosPosAvgSepCfList_ALamK0_BpTot(0),
  fNumPosPosAvgSepCfList_ALamK0_BmTot(0),
  fNumPosPosAvgSepCfList_ALamK0_Tot(0),
  fDenPosPosAvgSepCfALamK0Bp1(0),
  fDenPosPosAvgSepCfALamK0Bp2(0),
  fDenPosPosAvgSepCfALamK0Bm1(0),
  fDenPosPosAvgSepCfALamK0Bm2(0),
  fDenPosPosAvgSepCfALamK0Bm3(0),
  fDenPosPosAvgSepCfList_ALamK0_BpTot(0),
  fDenPosPosAvgSepCfList_ALamK0_BmTot(0),
  fDenPosPosAvgSepCfList_ALamK0_Tot(0),
  fCfPosPosAvgSepCfALamK0Bp1(0),
  fCfPosPosAvgSepCfALamK0Bp2(0),
  fCfPosPosAvgSepCfALamK0Bm1(0),
  fCfPosPosAvgSepCfALamK0Bm2(0),
  fCfPosPosAvgSepCfALamK0Bm3(0),
  fCfPosPosAvgSepCfList_ALamK0_BpTot(0),
  fCfPosPosAvgSepCfList_ALamK0_BmTot(0),
  fCfPosPosAvgSepCfList_ALamK0_Tot(0),
  fCfPosPosAvgSepCfALamK0BpTot(0),
  fCfPosPosAvgSepCfALamK0BmTot(0),
  fCfPosPosAvgSepCfALamK0Tot(0),
  
  //_____ +- ___________________
  fNumPosNegAvgSepCfLamK0Bp1(0),
  fNumPosNegAvgSepCfLamK0Bp2(0),
  fNumPosNegAvgSepCfLamK0Bm1(0),
  fNumPosNegAvgSepCfLamK0Bm2(0),
  fNumPosNegAvgSepCfLamK0Bm3(0),
  fNumPosNegAvgSepCfList_LamK0_BpTot(0),
  fNumPosNegAvgSepCfList_LamK0_BmTot(0),
  fNumPosNegAvgSepCfList_LamK0_Tot(0),
  fDenPosNegAvgSepCfLamK0Bp1(0),
  fDenPosNegAvgSepCfLamK0Bp2(0),
  fDenPosNegAvgSepCfLamK0Bm1(0),
  fDenPosNegAvgSepCfLamK0Bm2(0),
  fDenPosNegAvgSepCfLamK0Bm3(0),
  fDenPosNegAvgSepCfList_LamK0_BpTot(0),
  fDenPosNegAvgSepCfList_LamK0_BmTot(0),
  fDenPosNegAvgSepCfList_LamK0_Tot(0),
  fCfPosNegAvgSepCfLamK0Bp1(0),
  fCfPosNegAvgSepCfLamK0Bp2(0),
  fCfPosNegAvgSepCfLamK0Bm1(0),
  fCfPosNegAvgSepCfLamK0Bm2(0),
  fCfPosNegAvgSepCfLamK0Bm3(0),
  fCfPosNegAvgSepCfList_LamK0_BpTot(0),
  fCfPosNegAvgSepCfList_LamK0_BmTot(0),
  fCfPosNegAvgSepCfList_LamK0_Tot(0),
  fCfPosNegAvgSepCfLamK0BpTot(0),
  fCfPosNegAvgSepCfLamK0BmTot(0),
  fCfPosNegAvgSepCfLamK0Tot(0),

  fNumPosNegAvgSepCfALamK0Bp1(0),
  fNumPosNegAvgSepCfALamK0Bp2(0),
  fNumPosNegAvgSepCfALamK0Bm1(0),
  fNumPosNegAvgSepCfALamK0Bm2(0),
  fNumPosNegAvgSepCfALamK0Bm3(0),
  fNumPosNegAvgSepCfList_ALamK0_BpTot(0),
  fNumPosNegAvgSepCfList_ALamK0_BmTot(0),
  fNumPosNegAvgSepCfList_ALamK0_Tot(0),
  fDenPosNegAvgSepCfALamK0Bp1(0),
  fDenPosNegAvgSepCfALamK0Bp2(0),
  fDenPosNegAvgSepCfALamK0Bm1(0),
  fDenPosNegAvgSepCfALamK0Bm2(0),
  fDenPosNegAvgSepCfALamK0Bm3(0),
  fDenPosNegAvgSepCfList_ALamK0_BpTot(0),
  fDenPosNegAvgSepCfList_ALamK0_BmTot(0),
  fDenPosNegAvgSepCfList_ALamK0_Tot(0),
  fCfPosNegAvgSepCfALamK0Bp1(0),
  fCfPosNegAvgSepCfALamK0Bp2(0),
  fCfPosNegAvgSepCfALamK0Bm1(0),
  fCfPosNegAvgSepCfALamK0Bm2(0),
  fCfPosNegAvgSepCfALamK0Bm3(0),
  fCfPosNegAvgSepCfList_ALamK0_BpTot(0),
  fCfPosNegAvgSepCfList_ALamK0_BmTot(0),
  fCfPosNegAvgSepCfList_ALamK0_Tot(0),
  fCfPosNegAvgSepCfALamK0BpTot(0),
  fCfPosNegAvgSepCfALamK0BmTot(0),
  fCfPosNegAvgSepCfALamK0Tot(0),

  //_____ -+ ___________________
  fNumNegPosAvgSepCfLamK0Bp1(0),
  fNumNegPosAvgSepCfLamK0Bp2(0),
  fNumNegPosAvgSepCfLamK0Bm1(0),
  fNumNegPosAvgSepCfLamK0Bm2(0),
  fNumNegPosAvgSepCfLamK0Bm3(0),
  fNumNegPosAvgSepCfList_LamK0_BpTot(0),
  fNumNegPosAvgSepCfList_LamK0_BmTot(0),
  fNumNegPosAvgSepCfList_LamK0_Tot(0),
  fDenNegPosAvgSepCfLamK0Bp1(0),
  fDenNegPosAvgSepCfLamK0Bp2(0),
  fDenNegPosAvgSepCfLamK0Bm1(0),
  fDenNegPosAvgSepCfLamK0Bm2(0),
  fDenNegPosAvgSepCfLamK0Bm3(0),
  fDenNegPosAvgSepCfList_LamK0_BpTot(0),
  fDenNegPosAvgSepCfList_LamK0_BmTot(0),
  fDenNegPosAvgSepCfList_LamK0_Tot(0),
  fCfNegPosAvgSepCfLamK0Bp1(0),
  fCfNegPosAvgSepCfLamK0Bp2(0),
  fCfNegPosAvgSepCfLamK0Bm1(0),
  fCfNegPosAvgSepCfLamK0Bm2(0),
  fCfNegPosAvgSepCfLamK0Bm3(0),
  fCfNegPosAvgSepCfList_LamK0_BpTot(0),
  fCfNegPosAvgSepCfList_LamK0_BmTot(0),
  fCfNegPosAvgSepCfList_LamK0_Tot(0),
  fCfNegPosAvgSepCfLamK0BpTot(0),
  fCfNegPosAvgSepCfLamK0BmTot(0),
  fCfNegPosAvgSepCfLamK0Tot(0),

  fNumNegPosAvgSepCfALamK0Bp1(0),
  fNumNegPosAvgSepCfALamK0Bp2(0),
  fNumNegPosAvgSepCfALamK0Bm1(0),
  fNumNegPosAvgSepCfALamK0Bm2(0),
  fNumNegPosAvgSepCfALamK0Bm3(0),
  fNumNegPosAvgSepCfList_ALamK0_BpTot(0),
  fNumNegPosAvgSepCfList_ALamK0_BmTot(0),
  fNumNegPosAvgSepCfList_ALamK0_Tot(0),
  fDenNegPosAvgSepCfALamK0Bp1(0),
  fDenNegPosAvgSepCfALamK0Bp2(0),
  fDenNegPosAvgSepCfALamK0Bm1(0),
  fDenNegPosAvgSepCfALamK0Bm2(0),
  fDenNegPosAvgSepCfALamK0Bm3(0),
  fDenNegPosAvgSepCfList_ALamK0_BpTot(0),
  fDenNegPosAvgSepCfList_ALamK0_BmTot(0),
  fDenNegPosAvgSepCfList_ALamK0_Tot(0),
  fCfNegPosAvgSepCfALamK0Bp1(0),
  fCfNegPosAvgSepCfALamK0Bp2(0),
  fCfNegPosAvgSepCfALamK0Bm1(0),
  fCfNegPosAvgSepCfALamK0Bm2(0),
  fCfNegPosAvgSepCfALamK0Bm3(0),
  fCfNegPosAvgSepCfList_ALamK0_BpTot(0),
  fCfNegPosAvgSepCfList_ALamK0_BmTot(0),
  fCfNegPosAvgSepCfList_ALamK0_Tot(0),
  fCfNegPosAvgSepCfALamK0BpTot(0),
  fCfNegPosAvgSepCfALamK0BmTot(0),
  fCfNegPosAvgSepCfALamK0Tot(0),

  //_____ -- ___________________
  fNumNegNegAvgSepCfLamK0Bp1(0),
  fNumNegNegAvgSepCfLamK0Bp2(0),
  fNumNegNegAvgSepCfLamK0Bm1(0),
  fNumNegNegAvgSepCfLamK0Bm2(0),
  fNumNegNegAvgSepCfLamK0Bm3(0),
  fNumNegNegAvgSepCfList_LamK0_BpTot(0),
  fNumNegNegAvgSepCfList_LamK0_BmTot(0),
  fNumNegNegAvgSepCfList_LamK0_Tot(0),
  fDenNegNegAvgSepCfLamK0Bp1(0),
  fDenNegNegAvgSepCfLamK0Bp2(0),
  fDenNegNegAvgSepCfLamK0Bm1(0),
  fDenNegNegAvgSepCfLamK0Bm2(0),
  fDenNegNegAvgSepCfLamK0Bm3(0),
  fDenNegNegAvgSepCfList_LamK0_BpTot(0),
  fDenNegNegAvgSepCfList_LamK0_BmTot(0),
  fDenNegNegAvgSepCfList_LamK0_Tot(0),
  fCfNegNegAvgSepCfLamK0Bp1(0),
  fCfNegNegAvgSepCfLamK0Bp2(0),
  fCfNegNegAvgSepCfLamK0Bm1(0),
  fCfNegNegAvgSepCfLamK0Bm2(0),
  fCfNegNegAvgSepCfLamK0Bm3(0),
  fCfNegNegAvgSepCfList_LamK0_BpTot(0),
  fCfNegNegAvgSepCfList_LamK0_BmTot(0),
  fCfNegNegAvgSepCfList_LamK0_Tot(0),
  fCfNegNegAvgSepCfLamK0BpTot(0),
  fCfNegNegAvgSepCfLamK0BmTot(0),
  fCfNegNegAvgSepCfLamK0Tot(0),

  fNumNegNegAvgSepCfALamK0Bp1(0),
  fNumNegNegAvgSepCfALamK0Bp2(0),
  fNumNegNegAvgSepCfALamK0Bm1(0),
  fNumNegNegAvgSepCfALamK0Bm2(0),
  fNumNegNegAvgSepCfALamK0Bm3(0),
  fNumNegNegAvgSepCfList_ALamK0_BpTot(0),
  fNumNegNegAvgSepCfList_ALamK0_BmTot(0),
  fNumNegNegAvgSepCfList_ALamK0_Tot(0),
  fDenNegNegAvgSepCfALamK0Bp1(0),
  fDenNegNegAvgSepCfALamK0Bp2(0),
  fDenNegNegAvgSepCfALamK0Bm1(0),
  fDenNegNegAvgSepCfALamK0Bm2(0),
  fDenNegNegAvgSepCfALamK0Bm3(0),
  fDenNegNegAvgSepCfList_ALamK0_BpTot(0),
  fDenNegNegAvgSepCfList_ALamK0_BmTot(0),
  fDenNegNegAvgSepCfList_ALamK0_Tot(0),
  fCfNegNegAvgSepCfALamK0Bp1(0),
  fCfNegNegAvgSepCfALamK0Bp2(0),
  fCfNegNegAvgSepCfALamK0Bm1(0),
  fCfNegNegAvgSepCfALamK0Bm2(0),
  fCfNegNegAvgSepCfALamK0Bm3(0),
  fCfNegNegAvgSepCfList_ALamK0_BpTot(0),
  fCfNegNegAvgSepCfList_ALamK0_BmTot(0),
  fCfNegNegAvgSepCfList_ALamK0_Tot(0),
  fCfNegNegAvgSepCfALamK0BpTot(0),
  fCfNegNegAvgSepCfALamK0BmTot(0),
  fCfNegNegAvgSepCfALamK0Tot(0),

  //Purity calculations------------------
  fLambdaPurityBp1(0),
  fLambdaPurityBp2(0),
  fLambdaPurityBm1(0),
  fLambdaPurityBm2(0),
  fLambdaPurityBm3(0),

  fK0Short1PurityBp1(0),
  fK0Short1PurityBp2(0),
  fK0Short1PurityBm1(0),
  fK0Short1PurityBm2(0),
  fK0Short1PurityBm3(0),

  fAntiLambdaPurityBp1(0),
  fAntiLambdaPurityBp2(0),
  fAntiLambdaPurityBm1(0),
  fAntiLambdaPurityBm2(0),
  fAntiLambdaPurityBm3(0),

  fK0Short2PurityBp1(0),
  fK0Short2PurityBp2(0),
  fK0Short2PurityBm1(0),
  fK0Short2PurityBm2(0),
  fK0Short2PurityBm3(0)


{


}




//____________________________
TH1F* buildAllcLamK0::GetHistoClone(TString FileName, TString ArrayName, TString HistoName)
{
  TFile f1(FileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  if(ArrayName)
    {
      TObjArray *array = (TObjArray*)femtolist->FindObject(ArrayName);
      TH1F *ReturnHisto = (TH1F*)array->FindObject(HistoName);
    }
  else
    {
      TH1F *ReturnHisto = (TH1F*)femtolist->FindObject(HistoName);
    }
  TH1F *ReturnHistoCopy = (TH1F*)ReturnHisto->Clone();
  ReturnHistoCopy->SetDirectory(0);
  return ReturnHistoCopy;
}

//____________________________
TH1F* buildAllcLamK0::buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int aMinNormBin, int aMaxNormBin)
{
//  cout << "For hist "<< Num->GetName()<<", num pair = " << Num->Integral(1,Num->FindBin(.05)) << endl;
  double NumScale = Num->Integral(aMinNormBin,aMaxNormBin);
  double DenScale = Denom->Integral(aMinNormBin,aMaxNormBin);

  TH1F* CF = Num->Clone(name);
  CF->Divide(Denom);
  CF->Scale(DenScale/NumScale);
  CF->SetTitle(title);

  return CF;
}

//____________________________
TH1F* buildAllcLamK0::CombineCFs(TString name, TString title, TList* CfList, TList* NumList, int fMinNormBin, int fMaxNormBin)
{
  TIter CfIter(CfList);
  TIter NumIter(NumList);
  double scale = 0.;
  int counter = 0;
  double temp = 0.;

  TH1F* CF = (*CfIter.Begin())->Clone(name);
  CF->SetTitle(title);
    cout << "Name: " << CF->GetName() << endl;

//  TH1F* CF = (*CfIter.Begin())->Clone(name);
//  CF->SetTitle(title);

  TH1F* Num1 = (*NumIter.Begin());
  temp = Num1->Integral(fMinNormBin,fMaxNormBin);
    cout << "Name: " << Num1->GetName() << "  NumScale: " << Num1->Integral(fMinNormBin,fMaxNormBin) << endl;
  scale+=temp;
  counter++;

  CF->Scale(temp);

  while( (tempCF = (TH1F*)CfIter.Next()) && (tempNum = (TH1F*)NumIter.Next()) )
  {
    cout << "Name: " << tempCF->GetName() << endl;
    temp = tempNum->Integral(fMinNormBin,fMaxNormBin);
    cout << "Name: " << tempNum->GetName() << "  NumScale: " << tempNum->Integral(fMinNormBin,fMaxNormBin) << endl;
    CF->Add(tempCF,temp);
    scale += temp;
    counter ++;
  }
  cout << "SCALE = " << scale << endl;
  cout << "counter = " << counter << endl;
  CF->Scale(1./scale);
  return CF;

}

//____________________________
TList* buildAllcLamK0::Merge2Lists(TList* List1, TList* List2)
{
  TIter Iter1(List1);
  TIter Iter2(List2);
  TList *rList = new TList();
  TObject* obj;

  cout << "List 1 : " << endl;
  while(obj = (TObject*)Iter1.Next())
  {
    cout << obj->GetName() << endl;
    rList->Add(obj);
  }

  cout << "List 2 : " << endl;
  while(obj = (TObject*)Iter2.Next())
  {
    cout << obj->GetName() << endl;
    rList->Add(obj);
  }

  cout << "return List : " << endl;
  TIter rIter(rList);
  while(obj = (TObject*)rIter.Next())
  {
    cout << obj->GetName() << endl;
  }

  return rList;

}



//____________________________
void buildAllcLamK0::SetHistograms(TString aFileName)
{
  TString dirLamK0 = "LamK0";
  TString NumNameLamK0 = "NumKStarCF_LamK0";
  TString DenNameLamK0 = "DenKStarCF_LamK0";
  TString NumPosPosAvgSepCFNameLamK0 = "NumPosPosAvgSepCF_LamK0";
  TString DenPosPosAvgSepCFNameLamK0 = "DenPosPosAvgSepCF_LamK0";
  TString NumPosNegAvgSepCFNameLamK0 = "NumPosNegAvgSepCF_LamK0";
  TString DenPosNegAvgSepCFNameLamK0 = "DenPosNegAvgSepCF_LamK0";
  TString NumNegPosAvgSepCFNameLamK0 = "NumNegPosAvgSepCF_LamK0";
  TString DenNegPosAvgSepCFNameLamK0 = "DenNegPosAvgSepCF_LamK0";
  TString NumNegNegAvgSepCFNameLamK0 = "NumNegNegAvgSepCF_LamK0";
  TString DenNegNegAvgSepCFNameLamK0 = "DenNegNegAvgSepCF_LamK0";

  TString dirALamK0 = "ALamK0";
  TString NumNameALamK0 = "NumKStarCF_ALamK0";
  TString DenNameALamK0 = "DenKStarCF_ALamK0";
  TString NumPosPosAvgSepCFNameALamK0 = "NumPosPosAvgSepCF_ALamK0";
  TString DenPosPosAvgSepCFNameALamK0 = "DenPosPosAvgSepCF_ALamK0";
  TString NumPosNegAvgSepCFNameALamK0 = "NumPosNegAvgSepCF_ALamK0";
  TString DenPosNegAvgSepCFNameALamK0 = "DenPosNegAvgSepCF_ALamK0";
  TString NumNegPosAvgSepCFNameALamK0 = "NumNegPosAvgSepCF_ALamK0";
  TString DenNegPosAvgSepCFNameALamK0 = "DenNegPosAvgSepCF_ALamK0";
  TString NumNegNegAvgSepCFNameALamK0 = "NumNegNegAvgSepCF_ALamK0";
  TString DenNegNegAvgSepCFNameALamK0 = "DenNegNegAvgSepCF_ALamK0";

  TString LambdaPurity = "LambdaPurity";
  TString K0Short1Purity = "K0ShortPurity1";
  TString AntiLambdaPurity = "AntiLambdaPurity";
  TString K0Short2Purity = "K0ShortPurity1";

  if(aFileName.Contains("Bp1"))
  {
    //KStar CF------------------
    fNumLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,NumNameLamK0);
    fDenLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,DenNameLamK0);
    fCfLamK0Bp1 = buildCF("fCfLamK0Bp1","Lam-K0 (B+ 1)",fNumLamK0Bp1,fDenLamK0Bp1,fMinNormBinCF,fMaxNormBinCF);

    fNumALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,NumNameALamK0);
    fDenALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,DenNameALamK0);
    fCfALamK0Bp1 = buildCF("fCfALamK0Bp1","ALam-K0 (B+ 1)",fNumALamK0Bp1,fDenALamK0Bp1,fMinNormBinCF,fMaxNormBinCF);

    //Average Separation CF------------------
    //_____ ++ ___________
    fNumPosPosAvgSepCfLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,NumPosPosAvgSepCFNameLamK0);
    fDenPosPosAvgSepCfLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,DenPosPosAvgSepCFNameLamK0);
    fCfPosPosAvgSepCfLamK0Bp1 = buildCF("fCfPosPosAvgSepCfLamK0Bp1","Lam-K0 (B+ 1)",fNumPosPosAvgSepCfLamK0Bp1,fDenPosPosAvgSepCfLamK0Bp1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosPosAvgSepCfALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,NumPosPosAvgSepCFNameALamK0);
    fDenPosPosAvgSepCfALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,DenPosPosAvgSepCFNameALamK0);
    fCfPosPosAvgSepCfALamK0Bp1 = buildCF("fCfPosPosAvgSepCfALamK0Bp1","ALam-K0 (B+ 1)",fNumPosPosAvgSepCfALamK0Bp1,fDenPosPosAvgSepCfALamK0Bp1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ +- ___________
    fNumPosNegAvgSepCfLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,NumPosNegAvgSepCFNameLamK0);
    fDenPosNegAvgSepCfLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,DenPosNegAvgSepCFNameLamK0);
    fCfPosNegAvgSepCfLamK0Bp1 = buildCF("fCfPosNegAvgSepCfLamK0Bp1","Lam-K0 (B+ 1)",fNumPosNegAvgSepCfLamK0Bp1,fDenPosNegAvgSepCfLamK0Bp1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosNegAvgSepCfALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,NumPosNegAvgSepCFNameALamK0);
    fDenPosNegAvgSepCfALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,DenPosNegAvgSepCFNameALamK0);
    fCfPosNegAvgSepCfALamK0Bp1 = buildCF("fCfPosNegAvgSepCfALamK0Bp1","ALam-K0 (B+ 1)",fNumPosNegAvgSepCfALamK0Bp1,fDenPosNegAvgSepCfALamK0Bp1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -+ ___________
    fNumNegPosAvgSepCfLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,NumNegPosAvgSepCFNameLamK0);
    fDenNegPosAvgSepCfLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,DenNegPosAvgSepCFNameLamK0);
    fCfNegPosAvgSepCfLamK0Bp1 = buildCF("fCfNegPosAvgSepCfLamK0Bp1","Lam-K0 (B+ 1)",fNumNegPosAvgSepCfLamK0Bp1,fDenNegPosAvgSepCfLamK0Bp1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegPosAvgSepCfALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,NumNegPosAvgSepCFNameALamK0);
    fDenNegPosAvgSepCfALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,DenNegPosAvgSepCFNameALamK0);
    fCfNegPosAvgSepCfALamK0Bp1 = buildCF("fCfNegPosAvgSepCfALamK0Bp1","ALam-K0 (B+ 1)",fNumNegPosAvgSepCfALamK0Bp1,fDenNegPosAvgSepCfALamK0Bp1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -- ___________
    fNumNegNegAvgSepCfLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,NumNegNegAvgSepCFNameLamK0);
    fDenNegNegAvgSepCfLamK0Bp1 = GetHistoClone(aFileName,dirLamK0,DenNegNegAvgSepCFNameLamK0);
    fCfNegNegAvgSepCfLamK0Bp1 = buildCF("fCfNegNegAvgSepCfLamK0Bp1","Lam-K0 (B+ 1)",fNumNegNegAvgSepCfLamK0Bp1,fDenNegNegAvgSepCfLamK0Bp1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegNegAvgSepCfALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,NumNegNegAvgSepCFNameALamK0);
    fDenNegNegAvgSepCfALamK0Bp1 = GetHistoClone(aFileName,dirALamK0,DenNegNegAvgSepCFNameALamK0);
    fCfNegNegAvgSepCfALamK0Bp1 = buildCF("fCfNegNegAvgSepCfALamK0Bp1","ALam-K0 (B+ 1)",fNumNegNegAvgSepCfALamK0Bp1,fDenNegNegAvgSepCfALamK0Bp1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //Purity calculations------------------
    fLambdaPurityBp1 = GetHistoClone(aFileName,dirLamK0,LambdaPurity);
    fK0Short1PurityBp1 = GetHistoClone(aFileName,dirLamK0,K0Short1Purity);
    fAntiLambdaPurityBp1 = GetHistoClone(aFileName,dirALamK0,AntiLambdaPurity);
    fK0Short2PurityBp1 = GetHistoClone(aFileName,dirALamK0,K0Short2Purity);

  }
  else if(aFileName.Contains("Bp2"))
  {
    //KStar CF------------------
    fNumLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,NumNameLamK0);
    fDenLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,DenNameLamK0);
    fCfLamK0Bp2 = buildCF("fCfLamK0Bp2","Lam-K0 (B+ 2)",fNumLamK0Bp2,fDenLamK0Bp2,fMinNormBinCF,fMaxNormBinCF);

    fNumALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,NumNameALamK0);
    fDenALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,DenNameALamK0);
    fCfALamK0Bp2 = buildCF("fCfALamK0Bp2","ALam-K0 (B+ 2)",fNumALamK0Bp2,fDenALamK0Bp2,fMinNormBinCF,fMaxNormBinCF);

    //Average Separation CF------------------
    //_____ ++ ___________
    fNumPosPosAvgSepCfLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,NumPosPosAvgSepCFNameLamK0);
    fDenPosPosAvgSepCfLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,DenPosPosAvgSepCFNameLamK0);
    fCfPosPosAvgSepCfLamK0Bp2 = buildCF("fCfPosPosAvgSepCfLamK0Bp2","Lam-K0 (B+ 2)",fNumPosPosAvgSepCfLamK0Bp2,fDenPosPosAvgSepCfLamK0Bp2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosPosAvgSepCfALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,NumPosPosAvgSepCFNameALamK0);
    fDenPosPosAvgSepCfALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,DenPosPosAvgSepCFNameALamK0);
    fCfPosPosAvgSepCfALamK0Bp2 = buildCF("fCfPosPosAvgSepCfALamK0Bp2","ALam-K0 (B+ 2)",fNumPosPosAvgSepCfALamK0Bp2,fDenPosPosAvgSepCfALamK0Bp2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ +- ___________
    fNumPosNegAvgSepCfLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,NumPosNegAvgSepCFNameLamK0);
    fDenPosNegAvgSepCfLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,DenPosNegAvgSepCFNameLamK0);
    fCfPosNegAvgSepCfLamK0Bp2 = buildCF("fCfPosNegAvgSepCfLamK0Bp2","Lam-K0 (B+ 2)",fNumPosNegAvgSepCfLamK0Bp2,fDenPosNegAvgSepCfLamK0Bp2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosNegAvgSepCfALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,NumPosNegAvgSepCFNameALamK0);
    fDenPosNegAvgSepCfALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,DenPosNegAvgSepCFNameALamK0);
    fCfPosNegAvgSepCfALamK0Bp2 = buildCF("fCfPosNegAvgSepCfALamK0Bp2","ALam-K0 (B+ 2)",fNumPosNegAvgSepCfALamK0Bp2,fDenPosNegAvgSepCfALamK0Bp2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -+ ___________
    fNumNegPosAvgSepCfLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,NumNegPosAvgSepCFNameLamK0);
    fDenNegPosAvgSepCfLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,DenNegPosAvgSepCFNameLamK0);
    fCfNegPosAvgSepCfLamK0Bp2 = buildCF("fCfNegPosAvgSepCfLamK0Bp2","Lam-K0 (B+ 2)",fNumNegPosAvgSepCfLamK0Bp2,fDenNegPosAvgSepCfLamK0Bp2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegPosAvgSepCfALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,NumNegPosAvgSepCFNameALamK0);
    fDenNegPosAvgSepCfALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,DenNegPosAvgSepCFNameALamK0);
    fCfNegPosAvgSepCfALamK0Bp2 = buildCF("fCfNegPosAvgSepCfALamK0Bp2","ALam-K0 (B+ 2)",fNumNegPosAvgSepCfALamK0Bp2,fDenNegPosAvgSepCfALamK0Bp2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -- ___________
    fNumNegNegAvgSepCfLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,NumNegNegAvgSepCFNameLamK0);
    fDenNegNegAvgSepCfLamK0Bp2 = GetHistoClone(aFileName,dirLamK0,DenNegNegAvgSepCFNameLamK0);
    fCfNegNegAvgSepCfLamK0Bp2 = buildCF("fCfNegNegAvgSepCfLamK0Bp2","Lam-K0 (B+ 2)",fNumNegNegAvgSepCfLamK0Bp2,fDenNegNegAvgSepCfLamK0Bp2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegNegAvgSepCfALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,NumNegNegAvgSepCFNameALamK0);
    fDenNegNegAvgSepCfALamK0Bp2 = GetHistoClone(aFileName,dirALamK0,DenNegNegAvgSepCFNameALamK0);
    fCfNegNegAvgSepCfALamK0Bp2 = buildCF("fCfNegNegAvgSepCfALamK0Bp2","ALam-K0 (B+ 2)",fNumNegNegAvgSepCfALamK0Bp2,fDenNegNegAvgSepCfALamK0Bp2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //Purity calculations------------------
    fLambdaPurityBp2 = GetHistoClone(aFileName,dirLamK0,LambdaPurity);
    fK0Short1PurityBp2 = GetHistoClone(aFileName,dirLamK0,K0Short1Purity);
    fAntiLambdaPurityBp2 = GetHistoClone(aFileName,dirALamK0,AntiLambdaPurity);
    fK0Short2PurityBp2 = GetHistoClone(aFileName,dirALamK0,K0Short2Purity);

  }
  else if(aFileName.Contains("Bm1"))
  {
    //KStar CF------------------
    fNumLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,NumNameLamK0);
    fDenLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,DenNameLamK0);
    fCfLamK0Bm1 = buildCF("fCfLamK0Bm1","Lam-K0 (B- 1)",fNumLamK0Bm1,fDenLamK0Bm1,fMinNormBinCF,fMaxNormBinCF);

    fNumALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,NumNameALamK0);
    fDenALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,DenNameALamK0);
    fCfALamK0Bm1 = buildCF("fCfALamK0Bm1","ALam-K0 (B- 1)",fNumALamK0Bm1,fDenALamK0Bm1,fMinNormBinCF,fMaxNormBinCF);

    //Average Separation CF------------------
    //_____ ++ ___________
    fNumPosPosAvgSepCfLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,NumPosPosAvgSepCFNameLamK0);
    fDenPosPosAvgSepCfLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,DenPosPosAvgSepCFNameLamK0);
    fCfPosPosAvgSepCfLamK0Bm1 = buildCF("fCfPosPosAvgSepCfLamK0Bm1","Lam-K0 (B- 1)",fNumPosPosAvgSepCfLamK0Bm1,fDenPosPosAvgSepCfLamK0Bm1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosPosAvgSepCfALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,NumPosPosAvgSepCFNameALamK0);
    fDenPosPosAvgSepCfALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,DenPosPosAvgSepCFNameALamK0);
    fCfPosPosAvgSepCfALamK0Bm1 = buildCF("fCfPosPosAvgSepCfALamK0Bm1","ALam-K0 (B- 1)",fNumPosPosAvgSepCfALamK0Bm1,fDenPosPosAvgSepCfALamK0Bm1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ +- ___________
    fNumPosNegAvgSepCfLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,NumPosNegAvgSepCFNameLamK0);
    fDenPosNegAvgSepCfLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,DenPosNegAvgSepCFNameLamK0);
    fCfPosNegAvgSepCfLamK0Bm1 = buildCF("fCfPosNegAvgSepCfLamK0Bm1","Lam-K0 (B- 1)",fNumPosNegAvgSepCfLamK0Bm1,fDenPosNegAvgSepCfLamK0Bm1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosNegAvgSepCfALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,NumPosNegAvgSepCFNameALamK0);
    fDenPosNegAvgSepCfALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,DenPosNegAvgSepCFNameALamK0);
    fCfPosNegAvgSepCfALamK0Bm1 = buildCF("fCfPosNegAvgSepCfALamK0Bm1","ALam-K0 (B- 1)",fNumPosNegAvgSepCfALamK0Bm1,fDenPosNegAvgSepCfALamK0Bm1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -+ ___________
    fNumNegPosAvgSepCfLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,NumNegPosAvgSepCFNameLamK0);
    fDenNegPosAvgSepCfLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,DenNegPosAvgSepCFNameLamK0);
    fCfNegPosAvgSepCfLamK0Bm1 = buildCF("fCfNegPosAvgSepCfLamK0Bm1","Lam-K0 (B- 1)",fNumNegPosAvgSepCfLamK0Bm1,fDenNegPosAvgSepCfLamK0Bm1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegPosAvgSepCfALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,NumNegPosAvgSepCFNameALamK0);
    fDenNegPosAvgSepCfALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,DenNegPosAvgSepCFNameALamK0);
    fCfNegPosAvgSepCfALamK0Bm1 = buildCF("fCfNegPosAvgSepCfALamK0Bm1","ALam-K0 (B- 1)",fNumNegPosAvgSepCfALamK0Bm1,fDenNegPosAvgSepCfALamK0Bm1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -- ___________
    fNumNegNegAvgSepCfLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,NumNegNegAvgSepCFNameLamK0);
    fDenNegNegAvgSepCfLamK0Bm1 = GetHistoClone(aFileName,dirLamK0,DenNegNegAvgSepCFNameLamK0);
    fCfNegNegAvgSepCfLamK0Bm1 = buildCF("fCfNegNegAvgSepCfLamK0Bm1","Lam-K0 (B- 1)",fNumNegNegAvgSepCfLamK0Bm1,fDenNegNegAvgSepCfLamK0Bm1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegNegAvgSepCfALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,NumNegNegAvgSepCFNameALamK0);
    fDenNegNegAvgSepCfALamK0Bm1 = GetHistoClone(aFileName,dirALamK0,DenNegNegAvgSepCFNameALamK0);
    fCfNegNegAvgSepCfALamK0Bm1 = buildCF("fCfNegNegAvgSepCfALamK0Bm1","ALam-K0 (B- 1)",fNumNegNegAvgSepCfALamK0Bm1,fDenNegNegAvgSepCfALamK0Bm1,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //Purity calculations------------------
    fLambdaPurityBm1 = GetHistoClone(aFileName,dirLamK0,LambdaPurity);
    fK0Short1PurityBm1 = GetHistoClone(aFileName,dirLamK0,K0Short1Purity);
    fAntiLambdaPurityBm1 = GetHistoClone(aFileName,dirALamK0,AntiLambdaPurity);
    fK0Short2PurityBm1 = GetHistoClone(aFileName,dirALamK0,K0Short2Purity);

  }
  else if(aFileName.Contains("Bm2"))
  {
    //KStar CF------------------
    fNumLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,NumNameLamK0);
    fDenLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,DenNameLamK0);
    fCfLamK0Bm2 = buildCF("fCfLamK0Bm2","Lam-K0 (B- 2)",fNumLamK0Bm2,fDenLamK0Bm2,fMinNormBinCF,fMaxNormBinCF);

    fNumALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,NumNameALamK0);
    fDenALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,DenNameALamK0);
    fCfALamK0Bm2 = buildCF("fCfALamK0Bm2","ALam-K0 (B- 2)",fNumALamK0Bm2,fDenALamK0Bm2,fMinNormBinCF,fMaxNormBinCF);

    //Average Separation CF------------------
    //_____ ++ ___________
    fNumPosPosAvgSepCfLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,NumPosPosAvgSepCFNameLamK0);
    fDenPosPosAvgSepCfLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,DenPosPosAvgSepCFNameLamK0);
    fCfPosPosAvgSepCfLamK0Bm2 = buildCF("fCfPosPosAvgSepCfLamK0Bm2","Lam-K0 (B- 2)",fNumPosPosAvgSepCfLamK0Bm2,fDenPosPosAvgSepCfLamK0Bm2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosPosAvgSepCfALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,NumPosPosAvgSepCFNameALamK0);
    fDenPosPosAvgSepCfALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,DenPosPosAvgSepCFNameALamK0);
    fCfPosPosAvgSepCfALamK0Bm2 = buildCF("fCfPosPosAvgSepCfALamK0Bm2","ALam-K0 (B- 2)",fNumPosPosAvgSepCfALamK0Bm2,fDenPosPosAvgSepCfALamK0Bm2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ +- ___________
    fNumPosNegAvgSepCfLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,NumPosNegAvgSepCFNameLamK0);
    fDenPosNegAvgSepCfLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,DenPosNegAvgSepCFNameLamK0);
    fCfPosNegAvgSepCfLamK0Bm2 = buildCF("fCfPosNegAvgSepCfLamK0Bm2","Lam-K0 (B- 2)",fNumPosNegAvgSepCfLamK0Bm2,fDenPosNegAvgSepCfLamK0Bm2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosNegAvgSepCfALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,NumPosNegAvgSepCFNameALamK0);
    fDenPosNegAvgSepCfALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,DenPosNegAvgSepCFNameALamK0);
    fCfPosNegAvgSepCfALamK0Bm2 = buildCF("fCfPosNegAvgSepCfALamK0Bm2","ALam-K0 (B- 2)",fNumPosNegAvgSepCfALamK0Bm2,fDenPosNegAvgSepCfALamK0Bm2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -+ ___________
    fNumNegPosAvgSepCfLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,NumNegPosAvgSepCFNameLamK0);
    fDenNegPosAvgSepCfLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,DenNegPosAvgSepCFNameLamK0);
    fCfNegPosAvgSepCfLamK0Bm2 = buildCF("fCfNegPosAvgSepCfLamK0Bm2","Lam-K0 (B- 2)",fNumNegPosAvgSepCfLamK0Bm2,fDenNegPosAvgSepCfLamK0Bm2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegPosAvgSepCfALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,NumNegPosAvgSepCFNameALamK0);
    fDenNegPosAvgSepCfALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,DenNegPosAvgSepCFNameALamK0);
    fCfNegPosAvgSepCfALamK0Bm2 = buildCF("fCfNegPosAvgSepCfALamK0Bm2","ALam-K0 (B- 2)",fNumNegPosAvgSepCfALamK0Bm2,fDenNegPosAvgSepCfALamK0Bm2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -- ___________
    fNumNegNegAvgSepCfLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,NumNegNegAvgSepCFNameLamK0);
    fDenNegNegAvgSepCfLamK0Bm2 = GetHistoClone(aFileName,dirLamK0,DenNegNegAvgSepCFNameLamK0);
    fCfNegNegAvgSepCfLamK0Bm2 = buildCF("fCfNegNegAvgSepCfLamK0Bm2","Lam-K0 (B- 2)",fNumNegNegAvgSepCfLamK0Bm2,fDenNegNegAvgSepCfLamK0Bm2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegNegAvgSepCfALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,NumNegNegAvgSepCFNameALamK0);
    fDenNegNegAvgSepCfALamK0Bm2 = GetHistoClone(aFileName,dirALamK0,DenNegNegAvgSepCFNameALamK0);
    fCfNegNegAvgSepCfALamK0Bm2 = buildCF("fCfNegNegAvgSepCfALamK0Bm2","ALam-K0 (B- 2)",fNumNegNegAvgSepCfALamK0Bm2,fDenNegNegAvgSepCfALamK0Bm2,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //Purity calculations------------------
    fLambdaPurityBm2 = GetHistoClone(aFileName,dirLamK0,LambdaPurity);
    fK0Short1PurityBm2 = GetHistoClone(aFileName,dirLamK0,K0Short1Purity);
    fAntiLambdaPurityBm2 = GetHistoClone(aFileName,dirALamK0,AntiLambdaPurity);
    fK0Short2PurityBm2 = GetHistoClone(aFileName,dirALamK0,K0Short2Purity);

  }
  else if(aFileName.Contains("Bm3"))
  {
    //KStar CF------------------
    fNumLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,NumNameLamK0);
    fDenLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,DenNameLamK0);
    fCfLamK0Bm3 = buildCF("fCfLamK0Bm3","Lam-K0 (B- 3)",fNumLamK0Bm3,fDenLamK0Bm3,fMinNormBinCF,fMaxNormBinCF);

    fNumALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,NumNameALamK0);
    fDenALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,DenNameALamK0);
    fCfALamK0Bm3 = buildCF("fCfALamK0Bm3","ALam-K0 (B- 3)",fNumALamK0Bm3,fDenALamK0Bm3,fMinNormBinCF,fMaxNormBinCF);

    //Average Separation CF------------------
    //_____ ++ ___________
    fNumPosPosAvgSepCfLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,NumPosPosAvgSepCFNameLamK0);
    fDenPosPosAvgSepCfLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,DenPosPosAvgSepCFNameLamK0);
    fCfPosPosAvgSepCfLamK0Bm3 = buildCF("fCfPosPosAvgSepCfLamK0Bm3","Lam-K0 (B- 3)",fNumPosPosAvgSepCfLamK0Bm3,fDenPosPosAvgSepCfLamK0Bm3,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosPosAvgSepCfALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,NumPosPosAvgSepCFNameALamK0);
    fDenPosPosAvgSepCfALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,DenPosPosAvgSepCFNameALamK0);
    fCfPosPosAvgSepCfALamK0Bm3 = buildCF("fCfPosPosAvgSepCfALamK0Bm3","ALam-K0 (B- 3)",fNumPosPosAvgSepCfALamK0Bm3,fDenPosPosAvgSepCfALamK0Bm3,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ +- ___________
    fNumPosNegAvgSepCfLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,NumPosNegAvgSepCFNameLamK0);
    fDenPosNegAvgSepCfLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,DenPosNegAvgSepCFNameLamK0);
    fCfPosNegAvgSepCfLamK0Bm3 = buildCF("fCfPosNegAvgSepCfLamK0Bm3","Lam-K0 (B- 3)",fNumPosNegAvgSepCfLamK0Bm3,fDenPosNegAvgSepCfLamK0Bm3,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumPosNegAvgSepCfALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,NumPosNegAvgSepCFNameALamK0);
    fDenPosNegAvgSepCfALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,DenPosNegAvgSepCFNameALamK0);
    fCfPosNegAvgSepCfALamK0Bm3 = buildCF("fCfPosNegAvgSepCfALamK0Bm3","ALam-K0 (B- 3)",fNumPosNegAvgSepCfALamK0Bm3,fDenPosNegAvgSepCfALamK0Bm3,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -+ ___________
    fNumNegPosAvgSepCfLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,NumNegPosAvgSepCFNameLamK0);
    fDenNegPosAvgSepCfLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,DenNegPosAvgSepCFNameLamK0);
    fCfNegPosAvgSepCfLamK0Bm3 = buildCF("fCfNegPosAvgSepCfLamK0Bm3","Lam-K0 (B- 3)",fNumNegPosAvgSepCfLamK0Bm3,fDenNegPosAvgSepCfLamK0Bm3,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegPosAvgSepCfALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,NumNegPosAvgSepCFNameALamK0);
    fDenNegPosAvgSepCfALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,DenNegPosAvgSepCFNameALamK0);
    fCfNegPosAvgSepCfALamK0Bm3 = buildCF("fCfNegPosAvgSepCfALamK0Bm3","ALam-K0 (B- 3)",fNumNegPosAvgSepCfALamK0Bm3,fDenNegPosAvgSepCfALamK0Bm3,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //_____ -- ___________
    fNumNegNegAvgSepCfLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,NumNegNegAvgSepCFNameLamK0);
    fDenNegNegAvgSepCfLamK0Bm3 = GetHistoClone(aFileName,dirLamK0,DenNegNegAvgSepCFNameLamK0);
    fCfNegNegAvgSepCfLamK0Bm3 = buildCF("fCfNegNegAvgSepCfLamK0Bm3","Lam-K0 (B- 3)",fNumNegNegAvgSepCfLamK0Bm3,fDenNegNegAvgSepCfLamK0Bm3,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    fNumNegNegAvgSepCfALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,NumNegNegAvgSepCFNameALamK0);
    fDenNegNegAvgSepCfALamK0Bm3 = GetHistoClone(aFileName,dirALamK0,DenNegNegAvgSepCFNameALamK0);
    fCfNegNegAvgSepCfALamK0Bm3 = buildCF("fCfNegNegAvgSepCfALamK0Bm3","ALam-K0 (B- 3)",fNumNegNegAvgSepCfALamK0Bm3,fDenNegNegAvgSepCfALamK0Bm3,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

    //Purity calculations------------------
    fLambdaPurityBm3 = GetHistoClone(aFileName,dirLamK0,LambdaPurity);
    fK0Short1PurityBm3 = GetHistoClone(aFileName,dirLamK0,K0Short1Purity);
    fAntiLambdaPurityBm3 = GetHistoClone(aFileName,dirALamK0,AntiLambdaPurity);
    fK0Short2PurityBm3 = GetHistoClone(aFileName,dirALamK0,K0Short2Purity);

  }
  else
  {
    cout << "The input file name does not correspond to any Bp1, Bp2, Bm1, Bm2, or Bm3!!!!!!!!!!!!!!!!" << endl;
  }

}


//____________________________
void buildAllcLamK0::buildCorrCombined(bool BuildBpTot, bool BuildBmTot, bool BuildTot)
{

  //_____________________________________Lam-K0________________________________
  //KStar CF------------------
  if(BuildBpTot)
  {
    fNumList_LamK0_BpTot = new TList();
      fNumList_LamK0_BpTot->Add(fNumLamK0Bp1);
      fNumList_LamK0_BpTot->Add(fNumLamK0Bp2);
    fDenList_LamK0_BpTot = new TList();
      fDenList_LamK0_BpTot->Add(fDenLamK0Bp1);
      fDenList_LamK0_BpTot->Add(fDenLamK0Bp2);
    fCfList_LamK0_BpTot = new TList();
      fCfList_LamK0_BpTot->Add(fCfLamK0Bp1);
      fCfList_LamK0_BpTot->Add(fCfLamK0Bp2);
    fCfLamK0BpTot = CombineCFs("fCfLamK0BpTot","Lam-K0 (B+ Tot)",fCfList_LamK0_BpTot,fNumList_LamK0_BpTot,fMinNormBinCF,fMaxNormBinCF);
  }

  if(BuildBmTot)
  {
    fNumList_LamK0_BmTot = new TList();
      fNumList_LamK0_BmTot->Add(fNumLamK0Bm1);
      fNumList_LamK0_BmTot->Add(fNumLamK0Bm2);
      fNumList_LamK0_BmTot->Add(fNumLamK0Bm3);
    fDenList_LamK0_BmTot = new TList();
      fDenList_LamK0_BmTot->Add(fDenLamK0Bm1);
      fDenList_LamK0_BmTot->Add(fDenLamK0Bm2);
      fDenList_LamK0_BmTot->Add(fDenLamK0Bm3);
    fCfList_LamK0_BmTot = new TList();
      fCfList_LamK0_BmTot->Add(fCfLamK0Bm1);
      fCfList_LamK0_BmTot->Add(fCfLamK0Bm2);
      fCfList_LamK0_BmTot->Add(fCfLamK0Bm3);
    fCfLamK0BmTot = CombineCFs("fCfLamK0BmTot","Lam-K0 (B- Tot)",fCfList_LamK0_BmTot,fNumList_LamK0_BmTot,fMinNormBinCF,fMaxNormBinCF);
  }

  if(BuildTot)
  {
    fCfList_LamK0_Tot = Merge2Lists(fCfList_LamK0_BpTot,fCfList_LamK0_BmTot);
    fNumList_LamK0_Tot = Merge2Lists(fNumList_LamK0_BpTot,fNumList_LamK0_BmTot);
    fDenList_LamK0_Tot = Merge2Lists(fDenList_LamK0_BpTot,fDenList_LamK0_BmTot);
    fCfLamK0Tot = CombineCFs("fCfLamK0Tot","Lam-K0 (Tot)",fCfList_LamK0_Tot,fNumList_LamK0_Tot,fMinNormBinCF,fMaxNormBinCF);
  }

  //Average Separation CF------------------
  //_____ ++ ___________________
  if(BuildBpTot)
  {
    fNumPosPosAvgSepCfList_LamK0_BpTot = new TList();
      fNumPosPosAvgSepCfList_LamK0_BpTot->Add(fNumPosPosAvgSepCfLamK0Bp1);
      fNumPosPosAvgSepCfList_LamK0_BpTot->Add(fNumPosPosAvgSepCfLamK0Bp2);
    fDenPosPosAvgSepCfList_LamK0_BpTot = new TList();
      fDenPosPosAvgSepCfList_LamK0_BpTot->Add(fDenPosPosAvgSepCfLamK0Bp1);
      fDenPosPosAvgSepCfList_LamK0_BpTot->Add(fDenPosPosAvgSepCfLamK0Bp2);
    fCfPosPosAvgSepCfList_LamK0_BpTot = new TList();
      fCfPosPosAvgSepCfList_LamK0_BpTot->Add(fCfPosPosAvgSepCfLamK0Bp1);
      fCfPosPosAvgSepCfList_LamK0_BpTot->Add(fCfPosPosAvgSepCfLamK0Bp2);
    fCfPosPosAvgSepCfLamK0BpTot = CombineCFs("fCfPosPosAvgSepCfLamK0BpTot","Lam-K0 (B+ Tot)",fCfPosPosAvgSepCfList_LamK0_BpTot,fNumPosPosAvgSepCfList_LamK0_BpTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildBmTot)
  {
    fNumPosPosAvgSepCfList_LamK0_BmTot = new TList();
      fNumPosPosAvgSepCfList_LamK0_BmTot->Add(fNumPosPosAvgSepCfLamK0Bm1);
      fNumPosPosAvgSepCfList_LamK0_BmTot->Add(fNumPosPosAvgSepCfLamK0Bm2);
      fNumPosPosAvgSepCfList_LamK0_BmTot->Add(fNumPosPosAvgSepCfLamK0Bm3);
    fDenPosPosAvgSepCfList_LamK0_BmTot = new TList();
      fDenPosPosAvgSepCfList_LamK0_BmTot->Add(fDenPosPosAvgSepCfLamK0Bm1);
      fDenPosPosAvgSepCfList_LamK0_BmTot->Add(fDenPosPosAvgSepCfLamK0Bm2);
      fDenPosPosAvgSepCfList_LamK0_BmTot->Add(fDenPosPosAvgSepCfLamK0Bm3);
    fCfPosPosAvgSepCfList_LamK0_BmTot = new TList();
      fCfPosPosAvgSepCfList_LamK0_BmTot->Add(fCfPosPosAvgSepCfLamK0Bm1);
      fCfPosPosAvgSepCfList_LamK0_BmTot->Add(fCfPosPosAvgSepCfLamK0Bm2);
      fCfPosPosAvgSepCfList_LamK0_BmTot->Add(fCfPosPosAvgSepCfLamK0Bm3);
    fCfPosPosAvgSepCfLamK0BmTot = CombineCFs("fCfPosPosAvgSepCfLamK0BmTot","Lam-K0 (B- Tot)",fCfPosPosAvgSepCfList_LamK0_BmTot,fNumPosPosAvgSepCfList_LamK0_BmTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildTot)
  {
    fCfPosPosAvgSepCfList_LamK0_Tot = Merge2Lists(fCfPosPosAvgSepCfList_LamK0_BpTot,fCfPosPosAvgSepCfList_LamK0_BmTot);
    fNumPosPosAvgSepCfList_LamK0_Tot = Merge2Lists(fNumPosPosAvgSepCfList_LamK0_BpTot,fNumPosPosAvgSepCfList_LamK0_BmTot);
    fDenPosPosAvgSepCfList_LamK0_Tot = Merge2Lists(fDenPosPosAvgSepCfList_LamK0_BpTot,fDenPosPosAvgSepCfList_LamK0_BmTot);
    fCfPosPosAvgSepCfLamK0Tot = CombineCFs("fCfPosPosAvgSepCfLamK0Tot","Lam-K0 (Tot)",fCfPosPosAvgSepCfList_LamK0_Tot,fNumPosPosAvgSepCfList_LamK0_Tot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  //_____ +- ___________________
  if(BuildBpTot)
  {
    fNumPosNegAvgSepCfList_LamK0_BpTot = new TList();
      fNumPosNegAvgSepCfList_LamK0_BpTot->Add(fNumPosNegAvgSepCfLamK0Bp1);
      fNumPosNegAvgSepCfList_LamK0_BpTot->Add(fNumPosNegAvgSepCfLamK0Bp2);
    fDenPosNegAvgSepCfList_LamK0_BpTot = new TList();
      fDenPosNegAvgSepCfList_LamK0_BpTot->Add(fDenPosNegAvgSepCfLamK0Bp1);
      fDenPosNegAvgSepCfList_LamK0_BpTot->Add(fDenPosNegAvgSepCfLamK0Bp2);
    fCfPosNegAvgSepCfList_LamK0_BpTot = new TList();
      fCfPosNegAvgSepCfList_LamK0_BpTot->Add(fCfPosNegAvgSepCfLamK0Bp1);
      fCfPosNegAvgSepCfList_LamK0_BpTot->Add(fCfPosNegAvgSepCfLamK0Bp2);
    fCfPosNegAvgSepCfLamK0BpTot = CombineCFs("fCfPosNegAvgSepCfLamK0BpTot","Lam-K0 (B+ Tot)",fCfPosNegAvgSepCfList_LamK0_BpTot,fNumPosNegAvgSepCfList_LamK0_BpTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildBmTot)
  {
    fNumPosNegAvgSepCfList_LamK0_BmTot = new TList();
      fNumPosNegAvgSepCfList_LamK0_BmTot->Add(fNumPosNegAvgSepCfLamK0Bm1);
      fNumPosNegAvgSepCfList_LamK0_BmTot->Add(fNumPosNegAvgSepCfLamK0Bm2);
      fNumPosNegAvgSepCfList_LamK0_BmTot->Add(fNumPosNegAvgSepCfLamK0Bm3);
    fDenPosNegAvgSepCfList_LamK0_BmTot = new TList();
      fDenPosNegAvgSepCfList_LamK0_BmTot->Add(fDenPosNegAvgSepCfLamK0Bm1);
      fDenPosNegAvgSepCfList_LamK0_BmTot->Add(fDenPosNegAvgSepCfLamK0Bm2);
      fDenPosNegAvgSepCfList_LamK0_BmTot->Add(fDenPosNegAvgSepCfLamK0Bm3);
    fCfPosNegAvgSepCfList_LamK0_BmTot = new TList();
      fCfPosNegAvgSepCfList_LamK0_BmTot->Add(fCfPosNegAvgSepCfLamK0Bm1);
      fCfPosNegAvgSepCfList_LamK0_BmTot->Add(fCfPosNegAvgSepCfLamK0Bm2);
      fCfPosNegAvgSepCfList_LamK0_BmTot->Add(fCfPosNegAvgSepCfLamK0Bm3);
    fCfPosNegAvgSepCfLamK0BmTot = CombineCFs("fCfPosNegAvgSepCfLamK0BmTot","Lam-K0 (B- Tot)",fCfPosNegAvgSepCfList_LamK0_BmTot,fNumPosNegAvgSepCfList_LamK0_BmTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildTot)
  {
    fCfPosNegAvgSepCfList_LamK0_Tot = Merge2Lists(fCfPosNegAvgSepCfList_LamK0_BpTot,fCfPosNegAvgSepCfList_LamK0_BmTot);
    fNumPosNegAvgSepCfList_LamK0_Tot = Merge2Lists(fNumPosNegAvgSepCfList_LamK0_BpTot,fNumPosNegAvgSepCfList_LamK0_BmTot);
    fDenPosNegAvgSepCfList_LamK0_Tot = Merge2Lists(fDenPosNegAvgSepCfList_LamK0_BpTot,fDenPosNegAvgSepCfList_LamK0_BmTot);
    fCfPosNegAvgSepCfLamK0Tot = CombineCFs("fCfPosNegAvgSepCfLamK0Tot","Lam-K0 (Tot)",fCfPosNegAvgSepCfList_LamK0_Tot,fNumPosNegAvgSepCfList_LamK0_Tot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  //_____ -+ ___________________
  if(BuildBpTot)
  {
    fNumNegPosAvgSepCfList_LamK0_BpTot = new TList();
      fNumNegPosAvgSepCfList_LamK0_BpTot->Add(fNumNegPosAvgSepCfLamK0Bp1);
      fNumNegPosAvgSepCfList_LamK0_BpTot->Add(fNumNegPosAvgSepCfLamK0Bp2);
    fDenNegPosAvgSepCfList_LamK0_BpTot = new TList();
      fDenNegPosAvgSepCfList_LamK0_BpTot->Add(fDenNegPosAvgSepCfLamK0Bp1);
      fDenNegPosAvgSepCfList_LamK0_BpTot->Add(fDenNegPosAvgSepCfLamK0Bp2);
    fCfNegPosAvgSepCfList_LamK0_BpTot = new TList();
      fCfNegPosAvgSepCfList_LamK0_BpTot->Add(fCfNegPosAvgSepCfLamK0Bp1);
      fCfNegPosAvgSepCfList_LamK0_BpTot->Add(fCfNegPosAvgSepCfLamK0Bp2);
    fCfNegPosAvgSepCfLamK0BpTot = CombineCFs("fCfNegPosAvgSepCfLamK0BpTot","Lam-K0 (B+ Tot)",fCfNegPosAvgSepCfList_LamK0_BpTot,fNumNegPosAvgSepCfList_LamK0_BpTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildBmTot)
  {
    fNumNegPosAvgSepCfList_LamK0_BmTot = new TList();
      fNumNegPosAvgSepCfList_LamK0_BmTot->Add(fNumNegPosAvgSepCfLamK0Bm1);
      fNumNegPosAvgSepCfList_LamK0_BmTot->Add(fNumNegPosAvgSepCfLamK0Bm2);
      fNumNegPosAvgSepCfList_LamK0_BmTot->Add(fNumNegPosAvgSepCfLamK0Bm3);
    fDenNegPosAvgSepCfList_LamK0_BmTot = new TList();
      fDenNegPosAvgSepCfList_LamK0_BmTot->Add(fDenNegPosAvgSepCfLamK0Bm1);
      fDenNegPosAvgSepCfList_LamK0_BmTot->Add(fDenNegPosAvgSepCfLamK0Bm2);
      fDenNegPosAvgSepCfList_LamK0_BmTot->Add(fDenNegPosAvgSepCfLamK0Bm3);
    fCfNegPosAvgSepCfList_LamK0_BmTot = new TList();
      fCfNegPosAvgSepCfList_LamK0_BmTot->Add(fCfNegPosAvgSepCfLamK0Bm1);
      fCfNegPosAvgSepCfList_LamK0_BmTot->Add(fCfNegPosAvgSepCfLamK0Bm2);
      fCfNegPosAvgSepCfList_LamK0_BmTot->Add(fCfNegPosAvgSepCfLamK0Bm3);
    fCfNegPosAvgSepCfLamK0BmTot = CombineCFs("fCfNegPosAvgSepCfLamK0BmTot","Lam-K0 (B- Tot)",fCfNegPosAvgSepCfList_LamK0_BmTot,fNumNegPosAvgSepCfList_LamK0_BmTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildTot)
  {
    fCfNegPosAvgSepCfList_LamK0_Tot = Merge2Lists(fCfNegPosAvgSepCfList_LamK0_BpTot,fCfNegPosAvgSepCfList_LamK0_BmTot);
    fNumNegPosAvgSepCfList_LamK0_Tot = Merge2Lists(fNumNegPosAvgSepCfList_LamK0_BpTot,fNumNegPosAvgSepCfList_LamK0_BmTot);
    fDenNegPosAvgSepCfList_LamK0_Tot = Merge2Lists(fDenNegPosAvgSepCfList_LamK0_BpTot,fDenNegPosAvgSepCfList_LamK0_BmTot);
    fCfNegPosAvgSepCfLamK0Tot = CombineCFs("fCfNegPosAvgSepCfLamK0Tot","Lam-K0 (Tot)",fCfNegPosAvgSepCfList_LamK0_Tot,fNumNegPosAvgSepCfList_LamK0_Tot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  //_____ -- ___________________
  if(BuildBpTot)
  {
    fNumNegNegAvgSepCfList_LamK0_BpTot = new TList();
      fNumNegNegAvgSepCfList_LamK0_BpTot->Add(fNumNegNegAvgSepCfLamK0Bp1);
      fNumNegNegAvgSepCfList_LamK0_BpTot->Add(fNumNegNegAvgSepCfLamK0Bp2);
    fDenNegNegAvgSepCfList_LamK0_BpTot = new TList();
      fDenNegNegAvgSepCfList_LamK0_BpTot->Add(fDenNegNegAvgSepCfLamK0Bp1);
      fDenNegNegAvgSepCfList_LamK0_BpTot->Add(fDenNegNegAvgSepCfLamK0Bp2);
    fCfNegNegAvgSepCfList_LamK0_BpTot = new TList();
      fCfNegNegAvgSepCfList_LamK0_BpTot->Add(fCfNegNegAvgSepCfLamK0Bp1);
      fCfNegNegAvgSepCfList_LamK0_BpTot->Add(fCfNegNegAvgSepCfLamK0Bp2);
    fCfNegNegAvgSepCfLamK0BpTot = CombineCFs("fCfNegNegAvgSepCfLamK0BpTot","Lam-K0 (B+ Tot)",fCfNegNegAvgSepCfList_LamK0_BpTot,fNumNegNegAvgSepCfList_LamK0_BpTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
}

  if(BuildBmTot)
  {
    fNumNegNegAvgSepCfList_LamK0_BmTot = new TList();
      fNumNegNegAvgSepCfList_LamK0_BmTot->Add(fNumNegNegAvgSepCfLamK0Bm1);
      fNumNegNegAvgSepCfList_LamK0_BmTot->Add(fNumNegNegAvgSepCfLamK0Bm2);
      fNumNegNegAvgSepCfList_LamK0_BmTot->Add(fNumNegNegAvgSepCfLamK0Bm3);
    fDenNegNegAvgSepCfList_LamK0_BmTot = new TList();
      fDenNegNegAvgSepCfList_LamK0_BmTot->Add(fDenNegNegAvgSepCfLamK0Bm1);
      fDenNegNegAvgSepCfList_LamK0_BmTot->Add(fDenNegNegAvgSepCfLamK0Bm2);
      fDenNegNegAvgSepCfList_LamK0_BmTot->Add(fDenNegNegAvgSepCfLamK0Bm3);
    fCfNegNegAvgSepCfList_LamK0_BmTot = new TList();
      fCfNegNegAvgSepCfList_LamK0_BmTot->Add(fCfNegNegAvgSepCfLamK0Bm1);
      fCfNegNegAvgSepCfList_LamK0_BmTot->Add(fCfNegNegAvgSepCfLamK0Bm2);
      fCfNegNegAvgSepCfList_LamK0_BmTot->Add(fCfNegNegAvgSepCfLamK0Bm3);
    fCfNegNegAvgSepCfLamK0BmTot = CombineCFs("fCfNegNegAvgSepCfLamK0BmTot","Lam-K0 (B- Tot)",fCfNegNegAvgSepCfList_LamK0_BmTot,fNumNegNegAvgSepCfList_LamK0_BmTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildTot)
  {
    fCfNegNegAvgSepCfList_LamK0_Tot = Merge2Lists(fCfNegNegAvgSepCfList_LamK0_BpTot,fCfNegNegAvgSepCfList_LamK0_BmTot);
    fNumNegNegAvgSepCfList_LamK0_Tot = Merge2Lists(fNumNegNegAvgSepCfList_LamK0_BpTot,fNumNegNegAvgSepCfList_LamK0_BmTot);
    fDenNegNegAvgSepCfList_LamK0_Tot = Merge2Lists(fDenNegNegAvgSepCfList_LamK0_BpTot,fDenNegNegAvgSepCfList_LamK0_BmTot);
    fCfNegNegAvgSepCfLamK0Tot = CombineCFs("fCfNegNegAvgSepCfLamK0Tot","Lam-K0 (Tot)",fCfNegNegAvgSepCfList_LamK0_Tot,fNumNegNegAvgSepCfList_LamK0_Tot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }


  //_____________________________________ALam-K0________________________________
  //KStar CF------------------
  if(BuildBpTot)
  {
    fNumList_ALamK0_BpTot = new TList();
      fNumList_ALamK0_BpTot->Add(fNumALamK0Bp1);
      fNumList_ALamK0_BpTot->Add(fNumALamK0Bp2);
    fDenList_ALamK0_BpTot = new TList();
      fDenList_ALamK0_BpTot->Add(fDenALamK0Bp1);
      fDenList_ALamK0_BpTot->Add(fDenALamK0Bp2);
    fCfList_ALamK0_BpTot = new TList();
      fCfList_ALamK0_BpTot->Add(fCfALamK0Bp1);
      fCfList_ALamK0_BpTot->Add(fCfALamK0Bp2);
    fCfALamK0BpTot = CombineCFs("fCfALamK0BpTot","ALam-K0 (B+ Tot)",fCfList_ALamK0_BpTot,fNumList_ALamK0_BpTot,fMinNormBinCF,fMaxNormBinCF);
  }

  if(BuildBmTot)
  {
    fNumList_ALamK0_BmTot = new TList();
      fNumList_ALamK0_BmTot->Add(fNumALamK0Bm1);
      fNumList_ALamK0_BmTot->Add(fNumALamK0Bm2);
      fNumList_ALamK0_BmTot->Add(fNumALamK0Bm3);
    fDenList_ALamK0_BmTot = new TList();
      fDenList_ALamK0_BmTot->Add(fDenALamK0Bm1);
      fDenList_ALamK0_BmTot->Add(fDenALamK0Bm2);
      fDenList_ALamK0_BmTot->Add(fDenALamK0Bm3);
    fCfList_ALamK0_BmTot = new TList();
      fCfList_ALamK0_BmTot->Add(fCfALamK0Bm1);
      fCfList_ALamK0_BmTot->Add(fCfALamK0Bm2);
      fCfList_ALamK0_BmTot->Add(fCfALamK0Bm3);
    fCfALamK0BmTot = CombineCFs("fCfALamK0BmTot","ALam-K0 (B- Tot)",fCfList_ALamK0_BmTot,fNumList_ALamK0_BmTot,fMinNormBinCF,fMaxNormBinCF);
  }

  if(BuildTot)
  {
    fCfList_ALamK0_Tot = Merge2Lists(fCfList_ALamK0_BpTot,fCfList_ALamK0_BmTot);
    fNumList_ALamK0_Tot = Merge2Lists(fNumList_ALamK0_BpTot,fNumList_ALamK0_BmTot);
    fDenList_ALamK0_Tot = Merge2Lists(fDenList_ALamK0_BpTot,fDenList_ALamK0_BmTot);
    fCfALamK0Tot = CombineCFs("fCfALamK0Tot","ALam-K0 (Tot)",fCfList_ALamK0_Tot,fNumList_ALamK0_Tot,fMinNormBinCF,fMaxNormBinCF);
  }

  //Average Separation CF------------------
  //_____ ++ ___________________
  if(BuildBpTot)
  {
    fNumPosPosAvgSepCfList_ALamK0_BpTot = new TList();
      fNumPosPosAvgSepCfList_ALamK0_BpTot->Add(fNumPosPosAvgSepCfALamK0Bp1);
      fNumPosPosAvgSepCfList_ALamK0_BpTot->Add(fNumPosPosAvgSepCfALamK0Bp2);
    fDenPosPosAvgSepCfList_ALamK0_BpTot = new TList();
      fDenPosPosAvgSepCfList_ALamK0_BpTot->Add(fDenPosPosAvgSepCfALamK0Bp1);
      fDenPosPosAvgSepCfList_ALamK0_BpTot->Add(fDenPosPosAvgSepCfALamK0Bp2);
    fCfPosPosAvgSepCfList_ALamK0_BpTot = new TList();
      fCfPosPosAvgSepCfList_ALamK0_BpTot->Add(fCfPosPosAvgSepCfALamK0Bp1);
      fCfPosPosAvgSepCfList_ALamK0_BpTot->Add(fCfPosPosAvgSepCfALamK0Bp2);
    fCfPosPosAvgSepCfALamK0BpTot = CombineCFs("fCfPosPosAvgSepCfALamK0BpTot","ALam-K0 (B+ Tot)",fCfPosPosAvgSepCfList_ALamK0_BpTot,fNumPosPosAvgSepCfList_ALamK0_BpTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildBmTot)
  {
    fNumPosPosAvgSepCfList_ALamK0_BmTot = new TList();
      fNumPosPosAvgSepCfList_ALamK0_BmTot->Add(fNumPosPosAvgSepCfALamK0Bm1);
      fNumPosPosAvgSepCfList_ALamK0_BmTot->Add(fNumPosPosAvgSepCfALamK0Bm2);
      fNumPosPosAvgSepCfList_ALamK0_BmTot->Add(fNumPosPosAvgSepCfALamK0Bm3);
    fDenPosPosAvgSepCfList_ALamK0_BmTot = new TList();
      fDenPosPosAvgSepCfList_ALamK0_BmTot->Add(fDenPosPosAvgSepCfALamK0Bm1);
      fDenPosPosAvgSepCfList_ALamK0_BmTot->Add(fDenPosPosAvgSepCfALamK0Bm2);
      fDenPosPosAvgSepCfList_ALamK0_BmTot->Add(fDenPosPosAvgSepCfALamK0Bm3);
    fCfPosPosAvgSepCfList_ALamK0_BmTot = new TList();
      fCfPosPosAvgSepCfList_ALamK0_BmTot->Add(fCfPosPosAvgSepCfALamK0Bm1);
      fCfPosPosAvgSepCfList_ALamK0_BmTot->Add(fCfPosPosAvgSepCfALamK0Bm2);
      fCfPosPosAvgSepCfList_ALamK0_BmTot->Add(fCfPosPosAvgSepCfALamK0Bm3);
    fCfPosPosAvgSepCfALamK0BmTot = CombineCFs("fCfPosPosAvgSepCfALamK0BmTot","ALam-K0 (B- Tot)",fCfPosPosAvgSepCfList_ALamK0_BmTot,fNumPosPosAvgSepCfList_ALamK0_BmTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildTot)
  {
    fCfPosPosAvgSepCfList_ALamK0_Tot = Merge2Lists(fCfPosPosAvgSepCfList_ALamK0_BpTot,fCfPosPosAvgSepCfList_ALamK0_BmTot);
    fNumPosPosAvgSepCfList_ALamK0_Tot = Merge2Lists(fNumPosPosAvgSepCfList_ALamK0_BpTot,fNumPosPosAvgSepCfList_ALamK0_BmTot);
    fDenPosPosAvgSepCfList_ALamK0_Tot = Merge2Lists(fDenPosPosAvgSepCfList_ALamK0_BpTot,fDenPosPosAvgSepCfList_ALamK0_BmTot);
    fCfPosPosAvgSepCfALamK0Tot = CombineCFs("fCfPosPosAvgSepCfALamK0Tot","ALam-K0 (Tot)",fCfPosPosAvgSepCfList_ALamK0_Tot,fNumPosPosAvgSepCfList_ALamK0_Tot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  //_____ +- ___________________
  if(BuildBpTot)
  {
    fNumPosNegAvgSepCfList_ALamK0_BpTot = new TList();
      fNumPosNegAvgSepCfList_ALamK0_BpTot->Add(fNumPosNegAvgSepCfALamK0Bp1);
      fNumPosNegAvgSepCfList_ALamK0_BpTot->Add(fNumPosNegAvgSepCfALamK0Bp2);
    fDenPosNegAvgSepCfList_ALamK0_BpTot = new TList();
      fDenPosNegAvgSepCfList_ALamK0_BpTot->Add(fDenPosNegAvgSepCfALamK0Bp1);
      fDenPosNegAvgSepCfList_ALamK0_BpTot->Add(fDenPosNegAvgSepCfALamK0Bp2);
    fCfPosNegAvgSepCfList_ALamK0_BpTot = new TList();
      fCfPosNegAvgSepCfList_ALamK0_BpTot->Add(fCfPosNegAvgSepCfALamK0Bp1);
      fCfPosNegAvgSepCfList_ALamK0_BpTot->Add(fCfPosNegAvgSepCfALamK0Bp2);
    fCfPosNegAvgSepCfALamK0BpTot = CombineCFs("fCfPosNegAvgSepCfALamK0BpTot","ALam-K0 (B+ Tot)",fCfPosNegAvgSepCfList_ALamK0_BpTot,fNumPosNegAvgSepCfList_ALamK0_BpTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildBmTot)
  {
    fNumPosNegAvgSepCfList_ALamK0_BmTot = new TList();
      fNumPosNegAvgSepCfList_ALamK0_BmTot->Add(fNumPosNegAvgSepCfALamK0Bm1);
      fNumPosNegAvgSepCfList_ALamK0_BmTot->Add(fNumPosNegAvgSepCfALamK0Bm2);
      fNumPosNegAvgSepCfList_ALamK0_BmTot->Add(fNumPosNegAvgSepCfALamK0Bm3);
    fDenPosNegAvgSepCfList_ALamK0_BmTot = new TList();
      fDenPosNegAvgSepCfList_ALamK0_BmTot->Add(fDenPosNegAvgSepCfALamK0Bm1);
      fDenPosNegAvgSepCfList_ALamK0_BmTot->Add(fDenPosNegAvgSepCfALamK0Bm2);
      fDenPosNegAvgSepCfList_ALamK0_BmTot->Add(fDenPosNegAvgSepCfALamK0Bm3);
    fCfPosNegAvgSepCfList_ALamK0_BmTot = new TList();
      fCfPosNegAvgSepCfList_ALamK0_BmTot->Add(fCfPosNegAvgSepCfALamK0Bm1);
      fCfPosNegAvgSepCfList_ALamK0_BmTot->Add(fCfPosNegAvgSepCfALamK0Bm2);
      fCfPosNegAvgSepCfList_ALamK0_BmTot->Add(fCfPosNegAvgSepCfALamK0Bm3);
    fCfPosNegAvgSepCfALamK0BmTot = CombineCFs("fCfPosNegAvgSepCfALamK0BmTot","ALam-K0 (B- Tot)",fCfPosNegAvgSepCfList_ALamK0_BmTot,fNumPosNegAvgSepCfList_ALamK0_BmTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildTot)
  {
    fCfPosNegAvgSepCfList_ALamK0_Tot = Merge2Lists(fCfPosNegAvgSepCfList_ALamK0_BpTot,fCfPosNegAvgSepCfList_ALamK0_BmTot);
    fNumPosNegAvgSepCfList_ALamK0_Tot = Merge2Lists(fNumPosNegAvgSepCfList_ALamK0_BpTot,fNumPosNegAvgSepCfList_ALamK0_BmTot);
    fDenPosNegAvgSepCfList_ALamK0_Tot = Merge2Lists(fDenPosNegAvgSepCfList_ALamK0_BpTot,fDenPosNegAvgSepCfList_ALamK0_BmTot);
    fCfPosNegAvgSepCfALamK0Tot = CombineCFs("fCfPosNegAvgSepCfALamK0Tot","ALam-K0 (Tot)",fCfPosNegAvgSepCfList_ALamK0_Tot,fNumPosNegAvgSepCfList_ALamK0_Tot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  //_____ -+ ___________________
  if(BuildBpTot)
  {
    fNumNegPosAvgSepCfList_ALamK0_BpTot = new TList();
      fNumNegPosAvgSepCfList_ALamK0_BpTot->Add(fNumNegPosAvgSepCfALamK0Bp1);
      fNumNegPosAvgSepCfList_ALamK0_BpTot->Add(fNumNegPosAvgSepCfALamK0Bp2);
    fDenNegPosAvgSepCfList_ALamK0_BpTot = new TList();
      fDenNegPosAvgSepCfList_ALamK0_BpTot->Add(fDenNegPosAvgSepCfALamK0Bp1);
      fDenNegPosAvgSepCfList_ALamK0_BpTot->Add(fDenNegPosAvgSepCfALamK0Bp2);
    fCfNegPosAvgSepCfList_ALamK0_BpTot = new TList();
      fCfNegPosAvgSepCfList_ALamK0_BpTot->Add(fCfNegPosAvgSepCfALamK0Bp1);
      fCfNegPosAvgSepCfList_ALamK0_BpTot->Add(fCfNegPosAvgSepCfALamK0Bp2);
    fCfNegPosAvgSepCfALamK0BpTot = CombineCFs("fCfNegPosAvgSepCfALamK0BpTot","ALam-K0 (B+ Tot)",fCfNegPosAvgSepCfList_ALamK0_BpTot,fNumNegPosAvgSepCfList_ALamK0_BpTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildBmTot)
  {
    fNumNegPosAvgSepCfList_ALamK0_BmTot = new TList();
      fNumNegPosAvgSepCfList_ALamK0_BmTot->Add(fNumNegPosAvgSepCfALamK0Bm1);
      fNumNegPosAvgSepCfList_ALamK0_BmTot->Add(fNumNegPosAvgSepCfALamK0Bm2);
      fNumNegPosAvgSepCfList_ALamK0_BmTot->Add(fNumNegPosAvgSepCfALamK0Bm3);
    fDenNegPosAvgSepCfList_ALamK0_BmTot = new TList();
      fDenNegPosAvgSepCfList_ALamK0_BmTot->Add(fDenNegPosAvgSepCfALamK0Bm1);
      fDenNegPosAvgSepCfList_ALamK0_BmTot->Add(fDenNegPosAvgSepCfALamK0Bm2);
      fDenNegPosAvgSepCfList_ALamK0_BmTot->Add(fDenNegPosAvgSepCfALamK0Bm3);
    fCfNegPosAvgSepCfList_ALamK0_BmTot = new TList();
      fCfNegPosAvgSepCfList_ALamK0_BmTot->Add(fCfNegPosAvgSepCfALamK0Bm1);
      fCfNegPosAvgSepCfList_ALamK0_BmTot->Add(fCfNegPosAvgSepCfALamK0Bm2);
      fCfNegPosAvgSepCfList_ALamK0_BmTot->Add(fCfNegPosAvgSepCfALamK0Bm3);
    fCfNegPosAvgSepCfALamK0BmTot = CombineCFs("fCfNegPosAvgSepCfALamK0BmTot","ALam-K0 (B- Tot)",fCfNegPosAvgSepCfList_ALamK0_BmTot,fNumNegPosAvgSepCfList_ALamK0_BmTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildTot)
  {
    fCfNegPosAvgSepCfList_ALamK0_Tot = Merge2Lists(fCfNegPosAvgSepCfList_ALamK0_BpTot,fCfNegPosAvgSepCfList_ALamK0_BmTot);
    fNumNegPosAvgSepCfList_ALamK0_Tot = Merge2Lists(fNumNegPosAvgSepCfList_ALamK0_BpTot,fNumNegPosAvgSepCfList_ALamK0_BmTot);
    fDenNegPosAvgSepCfList_ALamK0_Tot = Merge2Lists(fDenNegPosAvgSepCfList_ALamK0_BpTot,fDenNegPosAvgSepCfList_ALamK0_BmTot);
    fCfNegPosAvgSepCfALamK0Tot = CombineCFs("fCfNegPosAvgSepCfALamK0Tot","ALam-K0 (Tot)",fCfNegPosAvgSepCfList_ALamK0_Tot,fNumNegPosAvgSepCfList_ALamK0_Tot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  //_____ -- ___________________
  if(BuildBpTot)
  {
    fNumNegNegAvgSepCfList_ALamK0_BpTot = new TList();
      fNumNegNegAvgSepCfList_ALamK0_BpTot->Add(fNumNegNegAvgSepCfALamK0Bp1);
      fNumNegNegAvgSepCfList_ALamK0_BpTot->Add(fNumNegNegAvgSepCfALamK0Bp2);
    fDenNegNegAvgSepCfList_ALamK0_BpTot = new TList();
      fDenNegNegAvgSepCfList_ALamK0_BpTot->Add(fDenNegNegAvgSepCfALamK0Bp1);
      fDenNegNegAvgSepCfList_ALamK0_BpTot->Add(fDenNegNegAvgSepCfALamK0Bp2);
    fCfNegNegAvgSepCfList_ALamK0_BpTot = new TList();
      fCfNegNegAvgSepCfList_ALamK0_BpTot->Add(fCfNegNegAvgSepCfALamK0Bp1);
      fCfNegNegAvgSepCfList_ALamK0_BpTot->Add(fCfNegNegAvgSepCfALamK0Bp2);
    fCfNegNegAvgSepCfALamK0BpTot = CombineCFs("fCfNegNegAvgSepCfALamK0BpTot","ALam-K0 (B+ Tot)",fCfNegNegAvgSepCfList_ALamK0_BpTot,fNumNegNegAvgSepCfList_ALamK0_BpTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildBmTot)
  {
    fNumNegNegAvgSepCfList_ALamK0_BmTot = new TList();
      fNumNegNegAvgSepCfList_ALamK0_BmTot->Add(fNumNegNegAvgSepCfALamK0Bm1);
      fNumNegNegAvgSepCfList_ALamK0_BmTot->Add(fNumNegNegAvgSepCfALamK0Bm2);
      fNumNegNegAvgSepCfList_ALamK0_BmTot->Add(fNumNegNegAvgSepCfALamK0Bm3);
    fDenNegNegAvgSepCfList_ALamK0_BmTot = new TList();
      fDenNegNegAvgSepCfList_ALamK0_BmTot->Add(fDenNegNegAvgSepCfALamK0Bm1);
      fDenNegNegAvgSepCfList_ALamK0_BmTot->Add(fDenNegNegAvgSepCfALamK0Bm2);
      fDenNegNegAvgSepCfList_ALamK0_BmTot->Add(fDenNegNegAvgSepCfALamK0Bm3);
    fCfNegNegAvgSepCfList_ALamK0_BmTot = new TList();
      fCfNegNegAvgSepCfList_ALamK0_BmTot->Add(fCfNegNegAvgSepCfALamK0Bm1);
      fCfNegNegAvgSepCfList_ALamK0_BmTot->Add(fCfNegNegAvgSepCfALamK0Bm2);
      fCfNegNegAvgSepCfList_ALamK0_BmTot->Add(fCfNegNegAvgSepCfALamK0Bm3);
    fCfNegNegAvgSepCfALamK0BmTot = CombineCFs("fCfNegNegAvgSepCfALamK0BmTot","ALam-K0 (B- Tot)",fCfNegNegAvgSepCfList_ALamK0_BmTot,fNumNegNegAvgSepCfList_ALamK0_BmTot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

  if(BuildTot)
  {
    fCfNegNegAvgSepCfList_ALamK0_Tot = Merge2Lists(fCfNegNegAvgSepCfList_ALamK0_BpTot,fCfNegNegAvgSepCfList_ALamK0_BmTot);
    fNumNegNegAvgSepCfList_ALamK0_Tot = Merge2Lists(fNumNegNegAvgSepCfList_ALamK0_BpTot,fNumNegNegAvgSepCfList_ALamK0_BmTot);
    fDenNegNegAvgSepCfList_ALamK0_Tot = Merge2Lists(fDenNegNegAvgSepCfList_ALamK0_BpTot,fDenNegNegAvgSepCfList_ALamK0_BmTot);
    fCfNegNegAvgSepCfALamK0Tot = CombineCFs("fCfNegNegAvgSepCfALamK0Tot","ALam-K0 (Tot)",fCfNegNegAvgSepCfList_ALamK0_Tot,fNumNegNegAvgSepCfList_ALamK0_Tot,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  }

}

//____________________________
TH1F* buildAllcLamK0::GetCf(TString aFileName, TString aAnalysis, TString aHistogram)
{
  if(aFileName.Contains("Bp1"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumLamK0Bp1)
        {
          cout << fNumLamK0Bp1->GetName() << endl;
          return fNumLamK0Bp1;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bp1/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenLamK0Bp1)
        {
          cout << fDenLamK0Bp1->GetName() << endl;
          return fDenLamK0Bp1;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bp1/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfLamK0Bp1)
        {
          cout << fCfLamK0Bp1->GetName() << endl;
          return fCfLamK0Bp1;
        }
        else
        {
          cout << "The Cf histogram for Bp1/LamK0 DNE!!!!!!!" << endl;
        }
      }
    }
    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumALamK0Bp1)
        {
          cout << fNumALamK0Bp1->GetName() << endl;
          return fNumALamK0Bp1;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bp1/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenALamK0Bp1)
        {
          cout << fDenALamK0Bp1->GetName() << endl;
          return fDenALamK0Bp1;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bp1/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfALamK0Bp1)
        {
          cout << fCfALamK0Bp1->GetName() << endl;
          return fCfALamK0Bp1;
        }
        else
        {
          cout << "The Cf histogram for Bp1/ALamK0 DNE!!!!!!!" << endl;
        }
      }
    }
  }

  else if(aFileName.Contains("Bp2"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumLamK0Bp2)
        {
          cout << fNumLamK0Bp2->GetName() << endl;
          return fNumLamK0Bp2;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bp2/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenLamK0Bp2)
        {
          cout << fDenLamK0Bp2->GetName() << endl;
          return fDenLamK0Bp2;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bp2/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfLamK0Bp2)
        {
          cout << fCfLamK0Bp2->GetName() << endl;
          return fCfLamK0Bp2;
        }
        else
        {
          cout << "The Cf histogram for Bp2/LamK0 DNE!!!!!!!" << endl;
        }
      }
    }
    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumALamK0Bp2)
        {
          cout << fNumALamK0Bp2->GetName() << endl;
          return fNumALamK0Bp2;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bp2/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenALamK0Bp2)
        {
          cout << fDenALamK0Bp2->GetName() << endl;
          return fDenALamK0Bp2;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bp2/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfALamK0Bp2)
        {
          cout << fCfALamK0Bp2->GetName() << endl;
          return fCfALamK0Bp2;
        }
        else
        {
          cout << "The Cf histogram for Bp2/ALamK0 DNE!!!!!!!" << endl;
        }
      }
    }
  }

  else if(aFileName.Contains("Bm1"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumLamK0Bm1)
        {
          cout << fNumLamK0Bm1->GetName() << endl;
          return fNumLamK0Bm1;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bm1/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenLamK0Bm1)
        {
          cout << fDenLamK0Bm1->GetName() << endl;
          return fDenLamK0Bm1;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bm1/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfLamK0Bm1)
        {
          cout << fCfLamK0Bm1->GetName() << endl;
          return fCfLamK0Bm1;
        }
        else
        {
          cout << "The Cf histogram for Bm1/LamK0 DNE!!!!!!!" << endl;
        }
      }
    }
    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumALamK0Bm1)
        {
          cout << fNumALamK0Bm1->GetName() << endl;
          return fNumALamK0Bm1;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bm1/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenALamK0Bm1)
        {
          cout << fDenALamK0Bm1->GetName() << endl;
          return fDenALamK0Bm1;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bm1/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfALamK0Bm1)
        {
          cout << fCfALamK0Bm1->GetName() << endl;
          return fCfALamK0Bm1;
        }
        else
        {
          cout << "The Cf histogram for Bm1/ALamK0 DNE!!!!!!!" << endl;
        }
      }
    }
  }

  else if(aFileName.Contains("Bm2"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumLamK0Bm2)
        {
          cout << fNumLamK0Bm2->GetName() << endl;
          return fNumLamK0Bm2;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bm2/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenLamK0Bm2)
        {
          cout << fDenLamK0Bm2->GetName() << endl;
          return fDenLamK0Bm2;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bm2/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfLamK0Bm2)
        {
          cout << fCfLamK0Bm2->GetName() << endl;
          return fCfLamK0Bm2;
        }
        else
        {
          cout << "The Cf histogram for Bm2/LamK0 DNE!!!!!!!" << endl;
        }
      }
    }
    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumALamK0Bm2)
        {
          cout << fNumALamK0Bm2->GetName() << endl;
          return fNumALamK0Bm2;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bm2/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenALamK0Bm2)
        {
          cout << fDenALamK0Bm2->GetName() << endl;
          return fDenALamK0Bm2;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bm2/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfALamK0Bm2)
        {
          cout << fCfALamK0Bm2->GetName() << endl;
          return fCfALamK0Bm2;
        }
        else
        {
          cout << "The Cf histogram for Bm2/ALamK0 DNE!!!!!!!" << endl;
        }
      }
    }
  }

  else if(aFileName.Contains("Bm3"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumLamK0Bm3)
        {
          cout << fNumLamK0Bm3->GetName() << endl;
          return fNumLamK0Bm3;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bm3/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenLamK0Bm3)
        {
          cout << fDenLamK0Bm3->GetName() << endl;
          return fDenLamK0Bm3;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bm3/LamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfLamK0Bm3)
        {
          cout << fCfLamK0Bm3->GetName() << endl;
          return fCfLamK0Bm3;
        }
        else
        {
          cout << "The Cf histogram for Bm3/LamK0 DNE!!!!!!!" << endl;
        }
      }
    }
    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aHistogram.EqualTo("Num"))
      {
        if(fNumALamK0Bm3)
        {
          cout << fNumALamK0Bm3->GetName() << endl;
          return fNumALamK0Bm3;
        }
        else
        {
          cout << "The Cf Numerator histogram for Bm3/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Den"))
      {
        if(fDenALamK0Bm3)
        {
          cout << fDenALamK0Bm3->GetName() << endl;
          return fDenALamK0Bm3;
        }
        else
        {
          cout << "The Cf Denominator histogram for Bm3/ALamK0 DNE!!!!!!!" << endl;
        }
      }
      else if(aHistogram.EqualTo("Cf"))
      {
        if(fCfALamK0Bm3)
        {
          cout << fCfALamK0Bm3->GetName() << endl;
          return fCfALamK0Bm3;
        }
        else
        {
          cout << "The Cf histogram for Bm3/ALamK0 DNE!!!!!!!" << endl;
        }
      }
    }
  }


}


//____________________________
TH1F* buildAllcLamK0::GetAvgSepCf(TString aFileName, TString aAnalysis, TString aDaughters, TString aHistogram)
{
  if(aFileName.Contains("Bp1"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfLamK0Bp1)
          {
            cout << fNumPosPosAvgSepCfLamK0Bp1->GetName() << endl;
            return fNumPosPosAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp1/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfLamK0Bp1)
          {
            cout << fDenPosPosAvgSepCfLamK0Bp1->GetName() << endl;
            return fDenPosPosAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp1/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfLamK0Bp1)
          {
            cout << fCfPosPosAvgSepCfLamK0Bp1->GetName() << endl;
            return fCfPosPosAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp1/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfLamK0Bp1)
          {
            cout << fNumPosNegAvgSepCfLamK0Bp1->GetName() << endl;
            return fNumPosNegAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp1/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfLamK0Bp1)
          {
            cout << fDenPosNegAvgSepCfLamK0Bp1->GetName() << endl;
            return fDenPosNegAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp1/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfLamK0Bp1)
          {
            cout << fCfPosNegAvgSepCfLamK0Bp1->GetName() << endl;
            return fCfPosNegAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp1/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfLamK0Bp1)
          {
            cout << fNumNegPosAvgSepCfLamK0Bp1->GetName() << endl;
            return fNumNegPosAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp1/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfLamK0Bp1)
          {
            cout << fDenNegPosAvgSepCfLamK0Bp1->GetName() << endl;
            return fDenNegPosAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp1/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfLamK0Bp1)
          {
            cout << fCfNegPosAvgSepCfLamK0Bp1->GetName() << endl;
            return fCfNegPosAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp1/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfLamK0Bp1)
          {
            cout << fNumNegNegAvgSepCfLamK0Bp1->GetName() << endl;
            return fNumNegNegAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp1/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfLamK0Bp1)
          {
            cout << fDenNegNegAvgSepCfLamK0Bp1->GetName() << endl;
            return fDenNegNegAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp1/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfLamK0Bp1)
          {
            cout << fCfNegNegAvgSepCfLamK0Bp1->GetName() << endl;
            return fCfNegNegAvgSepCfLamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp1/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }

    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfALamK0Bp1)
          {
            cout << fNumPosPosAvgSepCfALamK0Bp1->GetName() << endl;
            return fNumPosPosAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp1/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfALamK0Bp1)
          {
            cout << fDenPosPosAvgSepCfALamK0Bp1->GetName() << endl;
            return fDenPosPosAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp1/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfALamK0Bp1)
          {
            cout << fCfPosPosAvgSepCfALamK0Bp1->GetName() << endl;
            return fCfPosPosAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp1/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfALamK0Bp1)
          {
            cout << fNumPosNegAvgSepCfALamK0Bp1->GetName() << endl;
            return fNumPosNegAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp1/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfALamK0Bp1)
          {
            cout << fDenPosNegAvgSepCfALamK0Bp1->GetName() << endl;
            return fDenPosNegAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp1/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfALamK0Bp1)
          {
            cout << fCfPosNegAvgSepCfALamK0Bp1->GetName() << endl;
            return fCfPosNegAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp1/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfALamK0Bp1)
          {
            cout << fNumNegPosAvgSepCfALamK0Bp1->GetName() << endl;
            return fNumNegPosAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp1/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfALamK0Bp1)
          {
            cout << fDenNegPosAvgSepCfALamK0Bp1->GetName() << endl;
            return fDenNegPosAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp1/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfALamK0Bp1)
          {
            cout << fCfNegPosAvgSepCfALamK0Bp1->GetName() << endl;
            return fCfNegPosAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp1/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfALamK0Bp1)
          {
            cout << fNumNegNegAvgSepCfALamK0Bp1->GetName() << endl;
            return fNumNegNegAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp1/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfALamK0Bp1)
          {
            cout << fDenNegNegAvgSepCfALamK0Bp1->GetName() << endl;
            return fDenNegNegAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp1/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfALamK0Bp1)
          {
            cout << fCfNegNegAvgSepCfALamK0Bp1->GetName() << endl;
            return fCfNegNegAvgSepCfALamK0Bp1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp1/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }
  }


  if(aFileName.Contains("Bp2"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfLamK0Bp2)
          {
            cout << fNumPosPosAvgSepCfLamK0Bp2->GetName() << endl;
            return fNumPosPosAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp2/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfLamK0Bp2)
          {
            cout << fDenPosPosAvgSepCfLamK0Bp2->GetName() << endl;
            return fDenPosPosAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp2/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfLamK0Bp2)
          {
            cout << fCfPosPosAvgSepCfLamK0Bp2->GetName() << endl;
            return fCfPosPosAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp2/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfLamK0Bp2)
          {
            cout << fNumPosNegAvgSepCfLamK0Bp2->GetName() << endl;
            return fNumPosNegAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp2/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfLamK0Bp2)
          {
            cout << fDenPosNegAvgSepCfLamK0Bp2->GetName() << endl;
            return fDenPosNegAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp2/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfLamK0Bp2)
          {
            cout << fCfPosNegAvgSepCfLamK0Bp2->GetName() << endl;
            return fCfPosNegAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp2/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfLamK0Bp2)
          {
            cout << fNumNegPosAvgSepCfLamK0Bp2->GetName() << endl;
            return fNumNegPosAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp2/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfLamK0Bp2)
          {
            cout << fDenNegPosAvgSepCfLamK0Bp2->GetName() << endl;
            return fDenNegPosAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp2/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfLamK0Bp2)
          {
            cout << fCfNegPosAvgSepCfLamK0Bp2->GetName() << endl;
            return fCfNegPosAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp2/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfLamK0Bp2)
          {
            cout << fNumNegNegAvgSepCfLamK0Bp2->GetName() << endl;
            return fNumNegNegAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp2/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfLamK0Bp2)
          {
            cout << fDenNegNegAvgSepCfLamK0Bp2->GetName() << endl;
            return fDenNegNegAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp2/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfLamK0Bp2)
          {
            cout << fCfNegNegAvgSepCfLamK0Bp2->GetName() << endl;
            return fCfNegNegAvgSepCfLamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp2/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }

    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfALamK0Bp2)
          {
            cout << fNumPosPosAvgSepCfALamK0Bp2->GetName() << endl;
            return fNumPosPosAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp2/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfALamK0Bp2)
          {
            cout << fDenPosPosAvgSepCfALamK0Bp2->GetName() << endl;
            return fDenPosPosAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp2/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfALamK0Bp2)
          {
            cout << fCfPosPosAvgSepCfALamK0Bp2->GetName() << endl;
            return fCfPosPosAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp2/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfALamK0Bp2)
          {
            cout << fNumPosNegAvgSepCfALamK0Bp2->GetName() << endl;
            return fNumPosNegAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp2/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfALamK0Bp2)
          {
            cout << fDenPosNegAvgSepCfALamK0Bp2->GetName() << endl;
            return fDenPosNegAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp2/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfALamK0Bp2)
          {
            cout << fCfPosNegAvgSepCfALamK0Bp2->GetName() << endl;
            return fCfPosNegAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp2/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfALamK0Bp2)
          {
            cout << fNumNegPosAvgSepCfALamK0Bp2->GetName() << endl;
            return fNumNegPosAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp2/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfALamK0Bp2)
          {
            cout << fDenNegPosAvgSepCfALamK0Bp2->GetName() << endl;
            return fDenNegPosAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp2/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfALamK0Bp2)
          {
            cout << fCfNegPosAvgSepCfALamK0Bp2->GetName() << endl;
            return fCfNegPosAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp2/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfALamK0Bp2)
          {
            cout << fNumNegNegAvgSepCfALamK0Bp2->GetName() << endl;
            return fNumNegNegAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bp2/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfALamK0Bp2)
          {
            cout << fDenNegNegAvgSepCfALamK0Bp2->GetName() << endl;
            return fDenNegNegAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bp2/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfALamK0Bp2)
          {
            cout << fCfNegNegAvgSepCfALamK0Bp2->GetName() << endl;
            return fCfNegNegAvgSepCfALamK0Bp2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bp2/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }
  }


  if(aFileName.Contains("Bm1"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfLamK0Bm1)
          {
            cout << fNumPosPosAvgSepCfLamK0Bm1->GetName() << endl;
            return fNumPosPosAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm1/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfLamK0Bm1)
          {
            cout << fDenPosPosAvgSepCfLamK0Bm1->GetName() << endl;
            return fDenPosPosAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm1/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfLamK0Bm1)
          {
            cout << fCfPosPosAvgSepCfLamK0Bm1->GetName() << endl;
            return fCfPosPosAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm1/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfLamK0Bm1)
          {
            cout << fNumPosNegAvgSepCfLamK0Bm1->GetName() << endl;
            return fNumPosNegAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm1/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfLamK0Bm1)
          {
            cout << fDenPosNegAvgSepCfLamK0Bm1->GetName() << endl;
            return fDenPosNegAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm1/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfLamK0Bm1)
          {
            cout << fCfPosNegAvgSepCfLamK0Bm1->GetName() << endl;
            return fCfPosNegAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm1/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfLamK0Bm1)
          {
            cout << fNumNegPosAvgSepCfLamK0Bm1->GetName() << endl;
            return fNumNegPosAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm1/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfLamK0Bm1)
          {
            cout << fDenNegPosAvgSepCfLamK0Bm1->GetName() << endl;
            return fDenNegPosAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm1/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfLamK0Bm1)
          {
            cout << fCfNegPosAvgSepCfLamK0Bm1->GetName() << endl;
            return fCfNegPosAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm1/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfLamK0Bm1)
          {
            cout << fNumNegNegAvgSepCfLamK0Bm1->GetName() << endl;
            return fNumNegNegAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm1/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfLamK0Bm1)
          {
            cout << fDenNegNegAvgSepCfLamK0Bm1->GetName() << endl;
            return fDenNegNegAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm1/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfLamK0Bm1)
          {
            cout << fCfNegNegAvgSepCfLamK0Bm1->GetName() << endl;
            return fCfNegNegAvgSepCfLamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm1/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }

    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfALamK0Bm1)
          {
            cout << fNumPosPosAvgSepCfALamK0Bm1->GetName() << endl;
            return fNumPosPosAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm1/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfALamK0Bm1)
          {
            cout << fDenPosPosAvgSepCfALamK0Bm1->GetName() << endl;
            return fDenPosPosAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm1/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfALamK0Bm1)
          {
            cout << fCfPosPosAvgSepCfALamK0Bm1->GetName() << endl;
            return fCfPosPosAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm1/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfALamK0Bm1)
          {
            cout << fNumPosNegAvgSepCfALamK0Bm1->GetName() << endl;
            return fNumPosNegAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm1/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfALamK0Bm1)
          {
            cout << fDenPosNegAvgSepCfALamK0Bm1->GetName() << endl;
            return fDenPosNegAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm1/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfALamK0Bm1)
          {
            cout << fCfPosNegAvgSepCfALamK0Bm1->GetName() << endl;
            return fCfPosNegAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm1/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfALamK0Bm1)
          {
            cout << fNumNegPosAvgSepCfALamK0Bm1->GetName() << endl;
            return fNumNegPosAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm1/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfALamK0Bm1)
          {
            cout << fDenNegPosAvgSepCfALamK0Bm1->GetName() << endl;
            return fDenNegPosAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm1/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfALamK0Bm1)
          {
            cout << fCfNegPosAvgSepCfALamK0Bm1->GetName() << endl;
            return fCfNegPosAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm1/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfALamK0Bm1)
          {
            cout << fNumNegNegAvgSepCfALamK0Bm1->GetName() << endl;
            return fNumNegNegAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm1/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfALamK0Bm1)
          {
            cout << fDenNegNegAvgSepCfALamK0Bm1->GetName() << endl;
            return fDenNegNegAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm1/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfALamK0Bm1)
          {
            cout << fCfNegNegAvgSepCfALamK0Bm1->GetName() << endl;
            return fCfNegNegAvgSepCfALamK0Bm1;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm1/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }
  }


  if(aFileName.Contains("Bm2"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfLamK0Bm2)
          {
            cout << fNumPosPosAvgSepCfLamK0Bm2->GetName() << endl;
            return fNumPosPosAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm2/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfLamK0Bm2)
          {
            cout << fDenPosPosAvgSepCfLamK0Bm2->GetName() << endl;
            return fDenPosPosAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm2/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfLamK0Bm2)
          {
            cout << fCfPosPosAvgSepCfLamK0Bm2->GetName() << endl;
            return fCfPosPosAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm2/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfLamK0Bm2)
          {
            cout << fNumPosNegAvgSepCfLamK0Bm2->GetName() << endl;
            return fNumPosNegAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm2/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfLamK0Bm2)
          {
            cout << fDenPosNegAvgSepCfLamK0Bm2->GetName() << endl;
            return fDenPosNegAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm2/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfLamK0Bm2)
          {
            cout << fCfPosNegAvgSepCfLamK0Bm2->GetName() << endl;
            return fCfPosNegAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm2/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfLamK0Bm2)
          {
            cout << fNumNegPosAvgSepCfLamK0Bm2->GetName() << endl;
            return fNumNegPosAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm2/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfLamK0Bm2)
          {
            cout << fDenNegPosAvgSepCfLamK0Bm2->GetName() << endl;
            return fDenNegPosAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm2/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfLamK0Bm2)
          {
            cout << fCfNegPosAvgSepCfLamK0Bm2->GetName() << endl;
            return fCfNegPosAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm2/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfLamK0Bm2)
          {
            cout << fNumNegNegAvgSepCfLamK0Bm2->GetName() << endl;
            return fNumNegNegAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm2/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfLamK0Bm2)
          {
            cout << fDenNegNegAvgSepCfLamK0Bm2->GetName() << endl;
            return fDenNegNegAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm2/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfLamK0Bm2)
          {
            cout << fCfNegNegAvgSepCfLamK0Bm2->GetName() << endl;
            return fCfNegNegAvgSepCfLamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm2/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }

    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfALamK0Bm2)
          {
            cout << fNumPosPosAvgSepCfALamK0Bm2->GetName() << endl;
            return fNumPosPosAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm2/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfALamK0Bm2)
          {
            cout << fDenPosPosAvgSepCfALamK0Bm2->GetName() << endl;
            return fDenPosPosAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm2/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfALamK0Bm2)
          {
            cout << fCfPosPosAvgSepCfALamK0Bm2->GetName() << endl;
            return fCfPosPosAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm2/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfALamK0Bm2)
          {
            cout << fNumPosNegAvgSepCfALamK0Bm2->GetName() << endl;
            return fNumPosNegAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm2/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfALamK0Bm2)
          {
            cout << fDenPosNegAvgSepCfALamK0Bm2->GetName() << endl;
            return fDenPosNegAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm2/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfALamK0Bm2)
          {
            cout << fCfPosNegAvgSepCfALamK0Bm2->GetName() << endl;
            return fCfPosNegAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm2/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfALamK0Bm2)
          {
            cout << fNumNegPosAvgSepCfALamK0Bm2->GetName() << endl;
            return fNumNegPosAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm2/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfALamK0Bm2)
          {
            cout << fDenNegPosAvgSepCfALamK0Bm2->GetName() << endl;
            return fDenNegPosAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm2/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfALamK0Bm2)
          {
            cout << fCfNegPosAvgSepCfALamK0Bm2->GetName() << endl;
            return fCfNegPosAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm2/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfALamK0Bm2)
          {
            cout << fNumNegNegAvgSepCfALamK0Bm2->GetName() << endl;
            return fNumNegNegAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm2/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfALamK0Bm2)
          {
            cout << fDenNegNegAvgSepCfALamK0Bm2->GetName() << endl;
            return fDenNegNegAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm2/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfALamK0Bm2)
          {
            cout << fCfNegNegAvgSepCfALamK0Bm2->GetName() << endl;
            return fCfNegNegAvgSepCfALamK0Bm2;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm2/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }
  }


  if(aFileName.Contains("Bm3"))
  {
    if(aAnalysis.EqualTo("LamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfLamK0Bm3)
          {
            cout << fNumPosPosAvgSepCfLamK0Bm3->GetName() << endl;
            return fNumPosPosAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm3/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfLamK0Bm3)
          {
            cout << fDenPosPosAvgSepCfLamK0Bm3->GetName() << endl;
            return fDenPosPosAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm3/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfLamK0Bm3)
          {
            cout << fCfPosPosAvgSepCfLamK0Bm3->GetName() << endl;
            return fCfPosPosAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm3/LamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfLamK0Bm3)
          {
            cout << fNumPosNegAvgSepCfLamK0Bm3->GetName() << endl;
            return fNumPosNegAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm3/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfLamK0Bm3)
          {
            cout << fDenPosNegAvgSepCfLamK0Bm3->GetName() << endl;
            return fDenPosNegAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm3/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfLamK0Bm3)
          {
            cout << fCfPosNegAvgSepCfLamK0Bm3->GetName() << endl;
            return fCfPosNegAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm3/LamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfLamK0Bm3)
          {
            cout << fNumNegPosAvgSepCfLamK0Bm3->GetName() << endl;
            return fNumNegPosAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm3/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfLamK0Bm3)
          {
            cout << fDenNegPosAvgSepCfLamK0Bm3->GetName() << endl;
            return fDenNegPosAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm3/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfLamK0Bm3)
          {
            cout << fCfNegPosAvgSepCfLamK0Bm3->GetName() << endl;
            return fCfNegPosAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm3/LamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfLamK0Bm3)
          {
            cout << fNumNegNegAvgSepCfLamK0Bm3->GetName() << endl;
            return fNumNegNegAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm3/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfLamK0Bm3)
          {
            cout << fDenNegNegAvgSepCfLamK0Bm3->GetName() << endl;
            return fDenNegNegAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm3/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfLamK0Bm3)
          {
            cout << fCfNegNegAvgSepCfLamK0Bm3->GetName() << endl;
            return fCfNegNegAvgSepCfLamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm3/LamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }

    else if(aAnalysis.EqualTo("ALamK0"))
    {
      if(aDaughters.EqualTo("PosPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosPosAvgSepCfALamK0Bm3)
          {
            cout << fNumPosPosAvgSepCfALamK0Bm3->GetName() << endl;
            return fNumPosPosAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm3/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosPosAvgSepCfALamK0Bm3)
          {
            cout << fDenPosPosAvgSepCfALamK0Bm3->GetName() << endl;
            return fDenPosPosAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm3/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosPosAvgSepCfALamK0Bm3)
          {
            cout << fCfPosPosAvgSepCfALamK0Bm3->GetName() << endl;
            return fCfPosPosAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm3/ALamK0/PosPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("PosNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumPosNegAvgSepCfALamK0Bm3)
          {
            cout << fNumPosNegAvgSepCfALamK0Bm3->GetName() << endl;
            return fNumPosNegAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm3/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenPosNegAvgSepCfALamK0Bm3)
          {
            cout << fDenPosNegAvgSepCfALamK0Bm3->GetName() << endl;
            return fDenPosNegAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm3/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfPosNegAvgSepCfALamK0Bm3)
          {
            cout << fCfPosNegAvgSepCfALamK0Bm3->GetName() << endl;
            return fCfPosNegAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm3/ALamK0/PosNeg DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegPos"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegPosAvgSepCfALamK0Bm3)
          {
            cout << fNumNegPosAvgSepCfALamK0Bm3->GetName() << endl;
            return fNumNegPosAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm3/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegPosAvgSepCfALamK0Bm3)
          {
            cout << fDenNegPosAvgSepCfALamK0Bm3->GetName() << endl;
            return fDenNegPosAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm3/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegPosAvgSepCfALamK0Bm3)
          {
            cout << fCfNegPosAvgSepCfALamK0Bm3->GetName() << endl;
            return fCfNegPosAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm3/ALamK0/NegPos DNE!!!!!!!" << endl;
          }
        }
      }
      else if(aDaughters.EqualTo("NegNeg"))
      {
        if(aHistogram.EqualTo("Num"))
        {
          if(fNumNegNegAvgSepCfALamK0Bm3)
          {
            cout << fNumNegNegAvgSepCfALamK0Bm3->GetName() << endl;
            return fNumNegNegAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Numerator histogram for Bm3/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Den"))
        {
          if(fDenNegNegAvgSepCfALamK0Bm3)
          {
            cout << fDenNegNegAvgSepCfALamK0Bm3->GetName() << endl;
            return fDenNegNegAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf Denominator histogram for Bm3/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
        else if(aHistogram.EqualTo("Cf"))
        {
          if(fCfNegNegAvgSepCfALamK0Bm3)
          {
            cout << fCfNegNegAvgSepCfALamK0Bm3->GetName() << endl;
            return fCfNegNegAvgSepCfALamK0Bm3;
          }
          else
          {
            cout << "The AvgSepCf histogram for Bm3/ALamK0/NegNeg DNE!!!!!!!" << endl;
          }
        }
      }
    }
  }


}











//____________________________
TList* buildAllcLamK0::CalculatePurity(char* aName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2])
{
  char buffer[50];
  sprintf(buffer, "fitBgd_%s",aName);

  reject = kTRUE;
  ffBgFitLow[0] = aBgFitLow[0];
  ffBgFitLow[1] = aBgFitLow[1];
  ffBgFitHigh[0] = aBgFitHigh[0];
  ffBgFitHigh[1] = aBgFitHigh[1];

  TF1 *fitBgd = new TF1("fitBgd",PurityBgFitFunction,aPurityHisto->GetBinLowEdge(1),aPurityHisto->GetBinLowEdge(aPurityHisto->GetNbinsX()+1),5);
  aPurityHisto->Fit("fitBgd","0");

  reject = kFALSE;
  TF1 *fitBgd2 = new TF1(buffer,PurityBgFitFunction,aPurityHisto->GetBinLowEdge(1),aPurityHisto->GetBinLowEdge(aPurityHisto->GetNbinsX()+1),5);
  fitBgd2->SetParameters(fitBgd->GetParameters());

  //--------------------------------------------------------------------------------------------
  double bgd = fitBgd2->Integral(aROI[0],aROI[1]);
  bgd /= aPurityHisto->GetBinWidth(0);  //divide by bin size
  cout << aName << ": " << "bgd = " << bgd << endl;
  //-----
  double sigpbgd = aPurityHisto->Integral(aPurityHisto->FindBin(aROI[0]),aPurityHisto->FindBin(aROI[1]));
  cout << aName << ": " << "sig+bgd = " << sigpbgd << endl;
  //-----
  double sig = sigpbgd-bgd;
  cout << aName << ": " << "sig = " << sig << endl;
  //-----
  double pur = sig/sigpbgd;
  cout << aName << ": " << "Pur = " << pur << endl << endl;

  TVectorD *vInfo = new TVectorD(4);
    vInfo(0) = bgd;
    vInfo(1) = sigpbgd;
    vInfo(2) = sig;
    vInfo(3) = pur;

  TVectorD *vROI = new TVectorD(2);
    vROI(0) = aROI[0];
    vROI(1) = aROI[1];

  TVectorD *vBgFitLow = new TVectorD(2);
    vBgFitLow(0) = aBgFitLow[0];
    vBgFitLow(1) = aBgFitLow[1];

  TVectorD *vBgFitHigh = new TVectorD(2);
    vBgFitHigh(0) = aBgFitHigh[0];
    vBgFitHigh(1) = aBgFitHigh[1];
  //--------------------------------------------------------------------------------------------
  TList* temp = new TList();
  temp->Add(fitBgd2);
  temp->Add(vInfo);
  temp->Add(vROI);
  temp->Add(vBgFitLow);
  temp->Add(vBgFitHigh);
  return temp;

}

//____________________________
void buildAllcLamK0::PurityDrawAll(TH1F* PurityHisto, TList* FitList, bool ZoomBg)
{
  TIter iter(FitList);
  //-----
  TF1* fitBgd = *(iter.Begin());
    fitBgd->SetLineColor(4);
  //-----
  TVectorD* vInfo = (TVectorD*)iter.Next();
  //-----
  TVectorD* vROI = (TVectorD*)iter.Next();
  //-----
  TVectorD* vBgFitLow = (TVectorD*)iter.Next();
  //-----
  TVectorD* vBgFitHigh = (TVectorD*)iter.Next();
  //--------------------------------------------------------------------------------------------

  if(!ZoomBg)
  {
    double HistoMaxValue = PurityHisto->GetMaximum();
    TLine *lROImin = new TLine(vROI(0),0,vROI(0),HistoMaxValue);
    TLine *lROImax = new TLine(vROI(1),0,vROI(1),HistoMaxValue);
    //-----
    TLine *lBgFitLowMin = new TLine(vBgFitLow(0),0,vBgFitLow(0),HistoMaxValue);
    TLine *lBgFitLowMax = new TLine(vBgFitLow(1),0,vBgFitLow(1),HistoMaxValue);
    //-----
    TLine *lBgFitHighMin = new TLine(vBgFitHigh(0),0,vBgFitHigh(0),HistoMaxValue);
    TLine *lBgFitHighMax = new TLine(vBgFitHigh(1),0,vBgFitHigh(1),HistoMaxValue);
  }

  if(ZoomBg)
  {
    PurityHisto->GetXaxis()->SetRange(PurityHisto->FindBin(vBgFitLow(0)),PurityHisto->FindBin(vBgFitLow(1)));
      double maxLow = PurityHisto->GetMaximum();
      double minLow = PurityHisto->GetMinimum();
    PurityHisto->GetXaxis()->SetRange(PurityHisto->FindBin(vBgFitHigh(0)),PurityHisto->FindBin(vBgFitHigh(1))-1);
      double maxHigh = PurityHisto->GetMaximum();
      double minHigh = PurityHisto->GetMinimum();
    double maxBg;
      if(maxLow>maxHigh) maxBg = maxLow;
      else maxBg = maxHigh;
      //cout << "Background max = " << maxBg << endl;
    double minBg;
      if(minLow<minHigh) minBg = minLow;
      else minBg = minHigh;
      //cout << "Background min = " << minBg << endl;
    //--Extend the y-range that I plot

    double rangeBg = maxBg-minBg;
    maxBg+=rangeBg/10.;
    minBg-=rangeBg/10.;

    PurityHisto->GetXaxis()->SetRange(1,PurityHisto->GetNbinsX());
    PurityHisto->GetYaxis()->SetRangeUser(minBg,maxBg);
    //--------------------------------------------------------------------------------------------
    TLine *lROImin = new TLine(vROI(0),minBg,vROI(0),maxBg);
    TLine *lROImax = new TLine(vROI(1),minBg,vROI(1),maxBg);
    //-----
    TLine *lBgFitLowMin = new TLine(vBgFitLow(0),minBg,vBgFitLow(0),maxBg);
    TLine *lBgFitLowMax = new TLine(vBgFitLow(1),minBg,vBgFitLow(1),maxBg);
    //-----
    TLine *lBgFitHighMin = new TLine(vBgFitHigh(0),minBg,vBgFitHigh(0),maxBg);
    TLine *lBgFitHighMax = new TLine(vBgFitHigh(1),minBg,vBgFitHigh(1),maxBg);
  }

  //--------------------------------------------------------------------------------------------
  PurityHisto->SetLineColor(1);
  PurityHisto->SetLineWidth(3);

  lROImin->SetLineColor(3);
  lROImax->SetLineColor(3);

  lBgFitLowMin->SetLineColor(2);
  lBgFitLowMax->SetLineColor(2);

  lBgFitHighMin->SetLineColor(2);
  lBgFitHighMax->SetLineColor(2);


  //--------------------------------------------------------------------------------------------
  PurityHisto->DrawCopy("Ehist");
  fitBgd->Draw("same");
  lROImin->Draw();
  lROImax->Draw();
  lBgFitLowMin->Draw();
  lBgFitLowMax->Draw();
  lBgFitHighMin->Draw();
  lBgFitHighMax->Draw();

  if(!ZoomBg)
  {
    TPaveText *myText = new TPaveText(0.12,0.65,0.42,0.85,"NDC");
    char buffer[50];
    double purity = vInfo(3);
    char title[20] = PurityHisto->GetName();
    sprintf(buffer, "%s = %.2f\%",title, 100.*purity);
    myText->AddText(buffer);
    myText->Draw();
  }
}
