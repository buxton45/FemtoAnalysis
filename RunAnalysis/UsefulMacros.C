//A collection of different macros that I use all the time in my fitting etc.
//_________________________________________________________________________________________
TH1F* GetHisto(TString FileName, TString ArrayName, TString HistoName)
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

  TH1F *ReturnHistoClone = (TH1F*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}

//________________________________________________________________________________________________________________
TH2F* Get2dHisto(TString FileName, TString ArrayName, TString HistoName)
{
  TFile f1(FileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  if(ArrayName)
    {
      TObjArray *array = (TObjArray*)femtolist->FindObject(ArrayName);
      TH2F *ReturnHisto = (TH2F*)array->FindObject(HistoName);
    }
  else
    {
      TH1F *ReturnHisto = (TH2F*)femtolist->FindObject(HistoName);
    }

  TH2F *ReturnHistoClone = (TH2F*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}

//_________________________________________________________________________________________
TH1F* buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, double aMinNorm, double aMaxNorm, int aRebinFactor=1)
{
//  cout << "For hist "<< Num->GetName()<<", num pair = " << Num->Integral(1,Num->FindBin(.05)) << endl;
  TH1F* tNum = Num->Clone("tNum");
  TH1F* tDenom = Denom->Clone("tDenom");  

  tNum->Rebin(aRebinFactor);
  tDenom->Rebin(aRebinFactor);

  for(int i=1; i<=tDenom->GetNbinsX(); i++) tDenom->SetBinError(i,0.);

  int tMinNormBin = tNum->FindBin(aMinNorm);
  int tMaxNormBin = tNum->FindBin(aMaxNorm);

  double NumScale = tNum->Integral(tMinNormBin,tMaxNormBin);
  double DenScale = tDenom->Integral(tMinNormBin,tMaxNormBin);

  //tNum->Scale(1./NumScale);
  //tDenom->Scale(1./DenScale);

  TH1F* CF = tNum->Clone(name);
  CF->Divide(tDenom);
  CF->Scale(DenScale/NumScale);
  CF->SetTitle(title);

  return CF;
}

//_________________________________________________________________________________________
TH1F* CombineCFs(TString name, TString title, TList* CfList, TList* NumList, double aMinNorm, double aMaxNorm)
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

  int tMinNormBin = Num1->FindBin(aMinNorm);
  int tMaxNormBin = Num1->FindBin(aMaxNorm);

  temp = Num1->Integral(tMinNormBin,tMaxNormBin);
    cout << "Name: " << Num1->GetName() << "  NumScale: " << Num1->Integral(tMinNormBin,tMaxNormBin) << endl;
  scale+=temp;
  counter++;

  CF->Scale(temp);

  while( (tempCF = (TH1F*)CfIter.Next()) && (tempNum = (TH1F*)NumIter.Next()) )
  {
    cout << "Name: " << tempCF->GetName() << endl;
    temp = tempNum->Integral(tMinNormBin,tMaxNormBin);
    cout << "Name: " << tempNum->GetName() << "  NumScale: " << tempNum->Integral(tMinNormBin,tMaxNormBin) << endl;
    CF->Add(tempCF,temp);
    scale += temp;
    counter ++;
  }
  cout << "SCALE = " << scale << endl;
  cout << "counter = " << counter << endl;
  CF->Scale(1./scale);
  return CF;

}



//_________________________________________________________________________________________
TList* Merge2Lists(TList* List1, TList* List2)
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
