//_________________________________________________________________________________________
TH1F* GetHisto(TString FileName, TString ListName, TString ArrayName, TString HistoName)
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
TH1F* buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int fMinNormBin, int fMaxNormBin)
{
//  cout << "For hist "<< Num->GetName()<<", num pair = " << Num->Integral(1,Num->FindBin(.05)) << endl;
  double NumScale = Num->Integral(fMinNormBin,fMaxNormBin);
  double DenScale = Denom->Integral(fMinNormBin,fMaxNormBin);

  TH1F* CF = Num->Clone(name);
  CF->Divide(Denom);
  CF->Scale(DenScale/NumScale);
  CF->SetTitle(title);

  return CF;
}

//_________________________________________________________________________________________
TH1F* Combine(TString name, TString title, TList* CfList, TList* NumList, int fMinNormBin, int fMaxNormBin)
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

//_________________________________________________________________________________________
TMultiGraph* GetLednicky()
{
/*
  TFile f1("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_3_0.root");
  TGraph* graph1 = (TGraph*)f1.Get("Graph");
  graph1->SetLineColor(1);

  TFile f2("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_3_3.root");
  TGraph* graph2 = (TGraph*)f2.Get("Graph");
  graph2->SetLineColor(2);

  TFile f3("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_3_5.root");
  TGraph* graph3 = (TGraph*)f3.Get("Graph");
  graph3->SetLineColor(3);

  TFile f4("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_6_0.root");
  TGraph* graph4 = (TGraph*)f4.Get("Graph");
  graph4->SetLineColor(4);

  TFile f5("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_6_3.root");
  TGraph* graph5 = (TGraph*)f5.Get("Graph");
  graph5->SetLineColor(5);

  TFile f6("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_6_5.root");
  TGraph* graph6 = (TGraph*)f6.Get("Graph");
  graph6->SetLineColor(6);

  TFile f7("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_9_0.root");
  TGraph* graph7 = (TGraph*)f7.Get("Graph");
  graph7->SetLineColor(7);

  TFile f8("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_9_3.root");
  TGraph* graph8 = (TGraph*)f8.Get("Graph");
  graph8->SetLineColor(8);

  TFile f9("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_1_9_5.root");
  TGraph* graph9 = (TGraph*)f9.Get("Graph");
  graph9->SetLineColor(9);
*/

  TFile f10("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_3_0.root");
  TGraph* graph10 = (TGraph*)f10.Get("Graph");
  graph10->SetLineColor(1);

  TFile f11("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_3_3.root");
  TGraph* graph11 = (TGraph*)f11.Get("Graph");
  graph11->SetLineColor(2);

  TFile f12("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_3_5.root");
  TGraph* graph12 = (TGraph*)f12.Get("Graph");
  graph12->SetLineColor(3);

  TFile f13("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_6_0.root");
  TGraph* graph13 = (TGraph*)f13.Get("Graph");
  graph13->SetLineColor(4);

  TFile f14("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_6_3.root");
  TGraph* graph14 = (TGraph*)f14.Get("Graph");
  graph14->SetLineColor(5);

  TFile f15("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_6_5.root");
  TGraph* graph15 = (TGraph*)f15.Get("Graph");
  graph15->SetLineColor(6);

  TFile f16("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_9_0.root");
  TGraph* graph16 = (TGraph*)f16.Get("Graph");
  graph16->SetLineColor(7);

  TFile f17("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_9_3.root");
  TGraph* graph17 = (TGraph*)f17.Get("Graph");
  graph17->SetLineColor(8);

  TFile f18("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_05_9_5.root");
  TGraph* graph18 = (TGraph*)f18.Get("Graph");
  graph18->SetLineColor(9);

  TFile f19("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_3_0.root");
  TGraph* graph19 = (TGraph*)f19.Get("Graph");
  graph19->SetLineColor(41);

  TFile f20("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_3_3.root");
  TGraph* graph20 = (TGraph*)f20.Get("Graph");
  graph20->SetLineColor(42);

  TFile f21("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_3_5.root");
  TGraph* graph21 = (TGraph*)f21.Get("Graph");
  graph21->SetLineColor(43);

  TFile f22("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_6_0.root");
  TGraph* graph22 = (TGraph*)f22.Get("Graph");
  graph22->SetLineColor(44);

  TFile f23("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_6_3.root");
  TGraph* graph23 = (TGraph*)f23.Get("Graph");
  graph23->SetLineColor(45);

  TFile f24("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_6_5.root");
  TGraph* graph24 = (TGraph*)f24.Get("Graph");
  graph24->SetLineColor(46);

  TFile f25("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_9_0.root");
  TGraph* graph25 = (TGraph*)f25.Get("Graph");
  graph25->SetLineColor(47);

  TFile f26("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_9_3.root");
  TGraph* graph26 = (TGraph*)f26.Get("Graph");
  graph26->SetLineColor(48);

  TFile f27("~/Analysis/SimpleLednickyEqn/Variation/LednickyEqn_02_9_5.root");
  TGraph* graph27 = (TGraph*)f27.Get("Graph");
  graph27->SetLineColor(49);


  TMultiGraph *mg = new TMultiGraph();
/*
  mg->Add(graph1);
  mg->Add(graph2);
  mg->Add(graph3);
  mg->Add(graph4);
  mg->Add(graph5);
  mg->Add(graph6);
  mg->Add(graph7);
  mg->Add(graph8);
  mg->Add(graph9);
*/
  mg->Add(graph10);
  mg->Add(graph11);
  mg->Add(graph12);
  mg->Add(graph13);
  mg->Add(graph14);
  mg->Add(graph15);
  mg->Add(graph16);
  mg->Add(graph17);
  mg->Add(graph18);
  mg->Add(graph19);
  mg->Add(graph20);
  mg->Add(graph21);
  mg->Add(graph22);
  mg->Add(graph23);
  mg->Add(graph24);
  mg->Add(graph25);
  mg->Add(graph26);
  mg->Add(graph27);

  return mg;
}

//_________________________________________________________________________________________
void compCorrK0Kch_LedVar()
{
  const Int_t MinNormBin = 60;
  const Int_t MaxNormBin = 75;

 //___________________________________________________________________Lam-K0 Analysis__________________________________________________________
  //_____________________________________BP1________________________________
  TString File_Bp1 = "Resultsgrid_cLamK0_Bp1.root";
  //-------------Lam-K0------------------
  TH1F *NumLamK0Bp1 = GetHisto(File_Bp1, "femtolist", "LamK0", "NumLamK0KStarCF1");
    NumLamK0Bp1->SetLineColor(1);
  TH1F *DenLamK0Bp1 = GetHisto(File_Bp1, "femtolist", "LamK0", "DenLamK0KStarCF1");
    DenLamK0Bp1->SetLineColor(1);
  TH1F *CfLamK0Bp1 = buildCF("CfLamK0Bp1","Lam-K0 (B+ 1)",NumLamK0Bp1,DenLamK0Bp1,MinNormBin,MaxNormBin);
    CfLamK0Bp1->SetLineColor(1);
  //-------------ALam-K0----------------
  TH1F *NumALamK0Bp1 = GetHisto(File_Bp1, "femtolist", "ALamK0", "NumALamK0KStarCF2");
    NumALamK0Bp1->SetLineColor(1);
  TH1F *DenALamK0Bp1 = GetHisto(File_Bp1, "femtolist", "ALamK0", "DenALamK0KStarCF2");
    DenALamK0Bp1->SetLineColor(1);
  TH1F *CfALamK0Bp1 = buildCF("CfALamK0Bp1","ALam-K0 (B+ 1)",NumALamK0Bp1,DenALamK0Bp1,MinNormBin,MaxNormBin);
    CfALamK0Bp1->SetLineColor(1);

  //_____________________________________BP2________________________________
  TString File_Bp2 = "Resultsgrid_cLamK0_Bp2.root";
  //-------------Lam-K0------------------
  TH1F *NumLamK0Bp2 = GetHisto(File_Bp2, "femtolist", "LamK0", "NumLamK0KStarCF1");
    NumLamK0Bp2->SetLineColor(1);
  TH1F *DenLamK0Bp2 = GetHisto(File_Bp2, "femtolist", "LamK0", "DenLamK0KStarCF1");
    DenLamK0Bp2->SetLineColor(1);
  TH1F *CfLamK0Bp2 = buildCF("CfLamK0Bp2","Lam-K0 (B+ 2)",NumLamK0Bp2,DenLamK0Bp2,MinNormBin,MaxNormBin);
    CfLamK0Bp2->SetLineColor(1);
  //-------------ALam-K0----------------
  TH1F *NumALamK0Bp2 = GetHisto(File_Bp2, "femtolist", "ALamK0", "NumALamK0KStarCF2");
    NumALamK0Bp2->SetLineColor(1);
  TH1F *DenALamK0Bp2 = GetHisto(File_Bp2, "femtolist", "ALamK0", "DenALamK0KStarCF2");
    DenALamK0Bp2->SetLineColor(1);
  TH1F *CfALamK0Bp2 = buildCF("CfALamK0Bp2","ALam-K0 (B+ 2)",NumALamK0Bp2,DenALamK0Bp2,MinNormBin,MaxNormBin);
    CfALamK0Bp2->SetLineColor(1);

  //_____________________________________BM1________________________________
  TString File_Bm1 = "Resultsgrid_cLamK0_Bm1.root";
  //-------------Lam-K0------------------
  TH1F *NumLamK0Bm1 = GetHisto(File_Bm1, "femtolist", "LamK0", "NumLamK0KStarCF1");
    NumLamK0Bm1->SetLineColor(1);
  TH1F *DenLamK0Bm1 = GetHisto(File_Bm1, "femtolist", "LamK0", "DenLamK0KStarCF1");
    DenLamK0Bm1->SetLineColor(1);
  TH1F *CfLamK0Bm1 = buildCF("CfLamK0Bm1","Lam-K0 (B- 1)",NumLamK0Bm1,DenLamK0Bm1,MinNormBin,MaxNormBin);
    CfLamK0Bm1->SetLineColor(1);
  //-------------ALam-K0----------------
  TH1F *NumALamK0Bm1 = GetHisto(File_Bm1, "femtolist", "ALamK0", "NumALamK0KStarCF2");
    NumALamK0Bm1->SetLineColor(1);
  TH1F *DenALamK0Bm1 = GetHisto(File_Bm1, "femtolist", "ALamK0", "DenALamK0KStarCF2");
    DenALamK0Bm1->SetLineColor(1);
  TH1F *CfALamK0Bm1 = buildCF("CfALamK0Bm1","ALam-K0 (B- 1)",NumALamK0Bm1,DenALamK0Bm1,MinNormBin,MaxNormBin);
    CfALamK0Bm1->SetLineColor(1);

  //_____________________________________BM2________________________________
  TString File_Bm2 = "Resultsgrid_cLamK0_Bm2.root";
  //-------------Lam-K0------------------
  TH1F *NumLamK0Bm2 = GetHisto(File_Bm2, "femtolist", "LamK0", "NumLamK0KStarCF1");
    NumLamK0Bm2->SetLineColor(1);
  TH1F *DenLamK0Bm2 = GetHisto(File_Bm2, "femtolist", "LamK0", "DenLamK0KStarCF1");
    DenLamK0Bm2->SetLineColor(1);
  TH1F *CfLamK0Bm2 = buildCF("CfLamK0Bm2","Lam-K0 (B- 2)",NumLamK0Bm2,DenLamK0Bm2,MinNormBin,MaxNormBin);
    CfLamK0Bm2->SetLineColor(1);
  //-------------ALam-K0----------------
  TH1F *NumALamK0Bm2 = GetHisto(File_Bm2, "femtolist", "ALamK0", "NumALamK0KStarCF2");
    NumALamK0Bm2->SetLineColor(1);
  TH1F *DenALamK0Bm2 = GetHisto(File_Bm2, "femtolist", "ALamK0", "DenALamK0KStarCF2");
    DenALamK0Bm2->SetLineColor(1);
  TH1F *CfALamK0Bm2 = buildCF("CfALamK0Bm2","ALam-K0 (B- 2)",NumALamK0Bm2,DenALamK0Bm2,MinNormBin,MaxNormBin);
    CfALamK0Bm2->SetLineColor(1);

  //_____________________________________BM3________________________________
  TString File_Bm3 = "Resultsgrid_cLamK0_Bm3.root";
  //-------------Lam-K0------------------
  TH1F *NumLamK0Bm3 = GetHisto(File_Bm3, "femtolist", "LamK0", "NumLamK0KStarCF1");
    NumLamK0Bm3->SetLineColor(1);
  TH1F *DenLamK0Bm3 = GetHisto(File_Bm3, "femtolist", "LamK0", "DenLamK0KStarCF1");
    DenLamK0Bm3->SetLineColor(1);
  TH1F *CfLamK0Bm3 = buildCF("CfLamK0Bm3","Lam-K0 (B- 3)",NumLamK0Bm3,DenLamK0Bm3,MinNormBin,MaxNormBin);
    CfLamK0Bm3->SetLineColor(1);
  //-------------ALam-K0----------------
  TH1F *NumALamK0Bm3 = GetHisto(File_Bm3, "femtolist", "ALamK0", "NumALamK0KStarCF2");
    NumALamK0Bm3->SetLineColor(1);
  TH1F *DenALamK0Bm3 = GetHisto(File_Bm3, "femtolist", "ALamK0", "DenALamK0KStarCF2");
    DenALamK0Bm3->SetLineColor(1);
  TH1F *CfALamK0Bm3 = buildCF("CfALamK0Bm3","ALam-K0 (B- 3)",NumALamK0Bm3,DenALamK0Bm3,MinNormBin,MaxNormBin);
    CfALamK0Bm3->SetLineColor(1);

//__________________________________________________________________________________________________________________
  TList* CfList_LamK0_BpTot = new TList();
    CfList_LamK0_BpTot->Add(CfLamK0Bp1);
    CfList_LamK0_BpTot->Add(CfLamK0Bp2);
  TList* NumList_LamK0_BpTot = new TList();
    NumList_LamK0_BpTot->Add(NumLamK0Bp1);
    NumList_LamK0_BpTot->Add(NumLamK0Bp2);
  TH1F* CfLamK0BpTot = Combine("CfLamK0BpTot","Lam-K0 (B+ Tot)",CfList_LamK0_BpTot,NumList_LamK0_BpTot,MinNormBin,MaxNormBin);

  TList* CfList_ALamK0_BpTot = new TList();
    CfList_ALamK0_BpTot->Add(CfALamK0Bp1);
    CfList_ALamK0_BpTot->Add(CfALamK0Bp2);
  TList* NumList_ALamK0_BpTot = new TList();
    NumList_ALamK0_BpTot->Add(NumALamK0Bp1);
    NumList_ALamK0_BpTot->Add(NumALamK0Bp2);
  TH1F* CfALamK0BpTot = Combine("CfALamK0BpTot","ALam-K0 (B+ Tot)",CfList_ALamK0_BpTot,NumList_ALamK0_BpTot,MinNormBin,MaxNormBin);

  TList* CfList_LamK0_BmTot = new TList();
    CfList_LamK0_BmTot->Add(CfLamK0Bm1);
    CfList_LamK0_BmTot->Add(CfLamK0Bm2);
    CfList_LamK0_BmTot->Add(CfLamK0Bm3);
  TList* NumList_LamK0_BmTot = new TList();
    NumList_LamK0_BmTot->Add(NumLamK0Bm1);
    NumList_LamK0_BmTot->Add(NumLamK0Bm2);
    NumList_LamK0_BmTot->Add(NumLamK0Bm3);
  TH1F* CfLamK0BmTot = Combine("CfLamK0BmTot","Lam-K0 (B- Tot)",CfList_LamK0_BmTot,NumList_LamK0_BmTot,MinNormBin,MaxNormBin);

  TList* CfList_ALamK0_BmTot = new TList();
    CfList_ALamK0_BmTot->Add(CfALamK0Bm1);
    CfList_ALamK0_BmTot->Add(CfALamK0Bm2);
    CfList_ALamK0_BmTot->Add(CfALamK0Bm3);
  TList* NumList_ALamK0_BmTot = new TList();
    NumList_ALamK0_BmTot->Add(NumALamK0Bm1);
    NumList_ALamK0_BmTot->Add(NumALamK0Bm2);
    NumList_ALamK0_BmTot->Add(NumALamK0Bm3);
  TH1F* CfALamK0BmTot = Combine("CfALamK0BmTot","ALam-K0 (B- Tot)",CfList_ALamK0_BmTot,NumList_ALamK0_BmTot,MinNormBin,MaxNormBin);

  TList* CfList_LamK0_Tot = Merge2Lists(CfList_LamK0_BpTot,CfList_LamK0_BmTot);
  TList* NumList_LamK0_Tot = Merge2Lists(NumList_LamK0_BpTot,NumList_LamK0_BmTot);
  TH1F* CfLamK0Tot = Combine("CfLamK0Tot","Lam-K0 (Tot)",CfList_LamK0_Tot,NumList_LamK0_Tot,MinNormBin,MaxNormBin);
  
  TList* CfList_ALamK0_Tot = Merge2Lists(CfList_ALamK0_BpTot,CfList_ALamK0_BmTot);
  TList* NumList_ALamK0_Tot = Merge2Lists(NumList_ALamK0_BpTot,NumList_ALamK0_BmTot);
  TH1F* CfALamK0Tot = Combine("CfALamK0Tot","ALam-K0 (Tot)",CfList_ALamK0_Tot,NumList_ALamK0_Tot,MinNormBin,MaxNormBin);


 //___________________________________________________________________Lam-Kch Analysis__________________________________________________________
  //_____________________________________BP1________________________________
  TString File_Bp1 = "~/Analysis/KchLam/Resultsgrid_cLamcKch_Bp1.root";
  //------------- Lam-K+ ------------------
  TH1F *NumLamKchPBp1 = GetHisto(File_Bp1, "femtolist", "LamKchP", "NumLamKchPKStarCF");
    NumLamKchPBp1->SetLineColor(1);
  TH1F *DenLamKchPBp1 = GetHisto(File_Bp1, "femtolist", "LamKchP", "DenLamKchPKStarCF");
    DenLamKchPBp1->SetLineColor(1);
  TH1F *CfLamKchPBp1 = buildCF("CfLamKchPBp1","Lam-K+ (B+ 1)",NumLamKchPBp1,DenLamKchPBp1,MinNormBin,MaxNormBin);
    CfLamKchPBp1->SetLineColor(1);
  //------------- Lam-K- ------------------
  TH1F *NumLamKchMBp1 = GetHisto(File_Bp1, "femtolist", "LamKchM", "NumLamKchMKStarCF");
    NumLamKchMBp1->SetLineColor(1);
  TH1F *DenLamKchMBp1 = GetHisto(File_Bp1, "femtolist", "LamKchM", "DenLamKchMKStarCF");
    DenLamKchMBp1->SetLineColor(1);
  TH1F *CfLamKchMBp1 = buildCF("CfLamKchMBp1","Lam-K- (B+ 1)",NumLamKchMBp1,DenLamKchMBp1,MinNormBin,MaxNormBin);
    CfLamKchMBp1->SetLineColor(1);
  //------------- ALam-K+ ------------------
  TH1F *NumALamKchPBp1 = GetHisto(File_Bp1, "femtolist", "ALamKchP", "NumALamKchPKStarCF");
    NumALamKchPBp1->SetLineColor(1);
  TH1F *DenALamKchPBp1 = GetHisto(File_Bp1, "femtolist", "ALamKchP", "DenALamKchPKStarCF");
    DenALamKchPBp1->SetLineColor(1);
  TH1F *CfALamKchPBp1 = buildCF("CfALamKchPBp1","ALam-K+ (B+ 1)",NumALamKchPBp1,DenALamKchPBp1,MinNormBin,MaxNormBin);
    CfALamKchPBp1->SetLineColor(1);
  //------------- ALam-K- ------------------
  TH1F *NumALamKchMBp1 = GetHisto(File_Bp1, "femtolist", "ALamKchM", "NumALamKchMKStarCF");
    NumALamKchMBp1->SetLineColor(1);
  TH1F *DenALamKchMBp1 = GetHisto(File_Bp1, "femtolist", "ALamKchM", "DenALamKchMKStarCF");
    DenALamKchMBp1->SetLineColor(1);
  TH1F *CfALamKchMBp1 = buildCF("CfALamKchMBp1","ALam-K- (B+ 1)",NumALamKchMBp1,DenALamKchMBp1,MinNormBin,MaxNormBin);
    CfALamKchMBp1->SetLineColor(1);

  //_____________________________________BP2________________________________
  TString File_Bp2 = "~/Analysis/KchLam/Resultsgrid_cLamcKch_Bp2.root";
  //------------- Lam-K+ ------------------
  TH1F *NumLamKchPBp2 = GetHisto(File_Bp2, "femtolist", "LamKchP", "NumLamKchPKStarCF");
    NumLamKchPBp2->SetLineColor(1);
  TH1F *DenLamKchPBp2 = GetHisto(File_Bp2, "femtolist", "LamKchP", "DenLamKchPKStarCF");
    DenLamKchPBp2->SetLineColor(1);
  TH1F *CfLamKchPBp2 = buildCF("CfLamKchPBp2","Lam-K+ (B+ 1)",NumLamKchPBp2,DenLamKchPBp2,MinNormBin,MaxNormBin);
    CfLamKchPBp2->SetLineColor(1);
  //------------- Lam-K- ------------------
  TH1F *NumLamKchMBp2 = GetHisto(File_Bp2, "femtolist", "LamKchM", "NumLamKchMKStarCF");
    NumLamKchMBp2->SetLineColor(1);
  TH1F *DenLamKchMBp2 = GetHisto(File_Bp2, "femtolist", "LamKchM", "DenLamKchMKStarCF");
    DenLamKchMBp2->SetLineColor(1);
  TH1F *CfLamKchMBp2 = buildCF("CfLamKchMBp2","Lam-K- (B+ 1)",NumLamKchMBp2,DenLamKchMBp2,MinNormBin,MaxNormBin);
    CfLamKchMBp2->SetLineColor(1);
  //------------- ALam-K+ ------------------
  TH1F *NumALamKchPBp2 = GetHisto(File_Bp2, "femtolist", "ALamKchP", "NumALamKchPKStarCF");
    NumALamKchPBp2->SetLineColor(1);
  TH1F *DenALamKchPBp2 = GetHisto(File_Bp2, "femtolist", "ALamKchP", "DenALamKchPKStarCF");
    DenALamKchPBp2->SetLineColor(1);
  TH1F *CfALamKchPBp2 = buildCF("CfALamKchPBp2","ALam-K+ (B+ 1)",NumALamKchPBp2,DenALamKchPBp2,MinNormBin,MaxNormBin);
    CfALamKchPBp2->SetLineColor(1);
  //------------- ALam-K- ------------------
  TH1F *NumALamKchMBp2 = GetHisto(File_Bp2, "femtolist", "ALamKchM", "NumALamKchMKStarCF");
    NumALamKchMBp2->SetLineColor(1);
  TH1F *DenALamKchMBp2 = GetHisto(File_Bp2, "femtolist", "ALamKchM", "DenALamKchMKStarCF");
    DenALamKchMBp2->SetLineColor(1);
  TH1F *CfALamKchMBp2 = buildCF("CfALamKchMBp2","ALam-K- (B+ 1)",NumALamKchMBp2,DenALamKchMBp2,MinNormBin,MaxNormBin);
    CfALamKchMBp2->SetLineColor(1);

  //_____________________________________BM1________________________________
  TString File_Bm1 = "~/Analysis/KchLam/Resultsgrid_cLamcKch_Bm1.root";
  //------------- Lam-K+ ------------------
  TH1F *NumLamKchPBm1 = GetHisto(File_Bm1, "femtolist", "LamKchP", "NumLamKchPKStarCF");
    NumLamKchPBm1->SetLineColor(1);
  TH1F *DenLamKchPBm1 = GetHisto(File_Bm1, "femtolist", "LamKchP", "DenLamKchPKStarCF");
    DenLamKchPBm1->SetLineColor(1);
  TH1F *CfLamKchPBm1 = buildCF("CfLamKchPBm1","Lam-K+ (B+ 1)",NumLamKchPBm1,DenLamKchPBm1,MinNormBin,MaxNormBin);
    CfLamKchPBm1->SetLineColor(1);
  //------------- Lam-K- ------------------
  TH1F *NumLamKchMBm1 = GetHisto(File_Bm1, "femtolist", "LamKchM", "NumLamKchMKStarCF");
    NumLamKchMBm1->SetLineColor(1);
  TH1F *DenLamKchMBm1 = GetHisto(File_Bm1, "femtolist", "LamKchM", "DenLamKchMKStarCF");
    DenLamKchMBm1->SetLineColor(1);
  TH1F *CfLamKchMBm1 = buildCF("CfLamKchMBm1","Lam-K- (B+ 1)",NumLamKchMBm1,DenLamKchMBm1,MinNormBin,MaxNormBin);
    CfLamKchMBm1->SetLineColor(1);
  //------------- ALam-K+ ------------------
  TH1F *NumALamKchPBm1 = GetHisto(File_Bm1, "femtolist", "ALamKchP", "NumALamKchPKStarCF");
    NumALamKchPBm1->SetLineColor(1);
  TH1F *DenALamKchPBm1 = GetHisto(File_Bm1, "femtolist", "ALamKchP", "DenALamKchPKStarCF");
    DenALamKchPBm1->SetLineColor(1);
  TH1F *CfALamKchPBm1 = buildCF("CfALamKchPBm1","ALam-K+ (B+ 1)",NumALamKchPBm1,DenALamKchPBm1,MinNormBin,MaxNormBin);
    CfALamKchPBm1->SetLineColor(1);
  //------------- ALam-K- ------------------
  TH1F *NumALamKchMBm1 = GetHisto(File_Bm1, "femtolist", "ALamKchM", "NumALamKchMKStarCF");
    NumALamKchMBm1->SetLineColor(1);
  TH1F *DenALamKchMBm1 = GetHisto(File_Bm1, "femtolist", "ALamKchM", "DenALamKchMKStarCF");
    DenALamKchMBm1->SetLineColor(1);
  TH1F *CfALamKchMBm1 = buildCF("CfALamKchMBm1","ALam-K- (B+ 1)",NumALamKchMBm1,DenALamKchMBm1,MinNormBin,MaxNormBin);
    CfALamKchMBm1->SetLineColor(1);

  //_____________________________________BM2________________________________
  TString File_Bm2 = "~/Analysis/KchLam/Resultsgrid_cLamcKch_Bm2.root";
  //------------- Lam-K+ ------------------
  TH1F *NumLamKchPBm2 = GetHisto(File_Bm2, "femtolist", "LamKchP", "NumLamKchPKStarCF");
    NumLamKchPBm2->SetLineColor(1);
  TH1F *DenLamKchPBm2 = GetHisto(File_Bm2, "femtolist", "LamKchP", "DenLamKchPKStarCF");
    DenLamKchPBm2->SetLineColor(1);
  TH1F *CfLamKchPBm2 = buildCF("CfLamKchPBm2","Lam-K+ (B+ 1)",NumLamKchPBm2,DenLamKchPBm2,MinNormBin,MaxNormBin);
    CfLamKchPBm2->SetLineColor(1);
  //------------- Lam-K- ------------------
  TH1F *NumLamKchMBm2 = GetHisto(File_Bm2, "femtolist", "LamKchM", "NumLamKchMKStarCF");
    NumLamKchMBm2->SetLineColor(1);
  TH1F *DenLamKchMBm2 = GetHisto(File_Bm2, "femtolist", "LamKchM", "DenLamKchMKStarCF");
    DenLamKchMBm2->SetLineColor(1);
  TH1F *CfLamKchMBm2 = buildCF("CfLamKchMBm2","Lam-K- (B+ 1)",NumLamKchMBm2,DenLamKchMBm2,MinNormBin,MaxNormBin);
    CfLamKchMBm2->SetLineColor(1);
  //------------- ALam-K+ ------------------
  TH1F *NumALamKchPBm2 = GetHisto(File_Bm2, "femtolist", "ALamKchP", "NumALamKchPKStarCF");
    NumALamKchPBm2->SetLineColor(1);
  TH1F *DenALamKchPBm2 = GetHisto(File_Bm2, "femtolist", "ALamKchP", "DenALamKchPKStarCF");
    DenALamKchPBm2->SetLineColor(1);
  TH1F *CfALamKchPBm2 = buildCF("CfALamKchPBm2","ALam-K+ (B+ 1)",NumALamKchPBm2,DenALamKchPBm2,MinNormBin,MaxNormBin);
    CfALamKchPBm2->SetLineColor(1);
  //------------- ALam-K- ------------------
  TH1F *NumALamKchMBm2 = GetHisto(File_Bm2, "femtolist", "ALamKchM", "NumALamKchMKStarCF");
    NumALamKchMBm2->SetLineColor(1);
  TH1F *DenALamKchMBm2 = GetHisto(File_Bm2, "femtolist", "ALamKchM", "DenALamKchMKStarCF");
    DenALamKchMBm2->SetLineColor(1);
  TH1F *CfALamKchMBm2 = buildCF("CfALamKchMBm2","ALam-K- (B+ 1)",NumALamKchMBm2,DenALamKchMBm2,MinNormBin,MaxNormBin);
    CfALamKchMBm2->SetLineColor(1);

  //_____________________________________BM3________________________________
  TString File_Bm3 = "~/Analysis/KchLam/Resultsgrid_cLamcKch_Bm3.root";
  //------------- Lam-K+ ------------------
  TH1F *NumLamKchPBm3 = GetHisto(File_Bm3, "femtolist", "LamKchP", "NumLamKchPKStarCF");
    NumLamKchPBm3->SetLineColor(1);
  TH1F *DenLamKchPBm3 = GetHisto(File_Bm3, "femtolist", "LamKchP", "DenLamKchPKStarCF");
    DenLamKchPBm3->SetLineColor(1);
  TH1F *CfLamKchPBm3 = buildCF("CfLamKchPBm3","Lam-K+ (B+ 1)",NumLamKchPBm3,DenLamKchPBm3,MinNormBin,MaxNormBin);
    CfLamKchPBm3->SetLineColor(1);
  //------------- Lam-K- ------------------
  TH1F *NumLamKchMBm3 = GetHisto(File_Bm3, "femtolist", "LamKchM", "NumLamKchMKStarCF");
    NumLamKchMBm3->SetLineColor(1);
  TH1F *DenLamKchMBm3 = GetHisto(File_Bm3, "femtolist", "LamKchM", "DenLamKchMKStarCF");
    DenLamKchMBm3->SetLineColor(1);
  TH1F *CfLamKchMBm3 = buildCF("CfLamKchMBm3","Lam-K- (B+ 1)",NumLamKchMBm3,DenLamKchMBm3,MinNormBin,MaxNormBin);
    CfLamKchMBm3->SetLineColor(1);
  //------------- ALam-K+ ------------------
  TH1F *NumALamKchPBm3 = GetHisto(File_Bm3, "femtolist", "ALamKchP", "NumALamKchPKStarCF");
    NumALamKchPBm3->SetLineColor(1);
  TH1F *DenALamKchPBm3 = GetHisto(File_Bm3, "femtolist", "ALamKchP", "DenALamKchPKStarCF");
    DenALamKchPBm3->SetLineColor(1);
  TH1F *CfALamKchPBm3 = buildCF("CfALamKchPBm3","ALam-K+ (B+ 1)",NumALamKchPBm3,DenALamKchPBm3,MinNormBin,MaxNormBin);
    CfALamKchPBm3->SetLineColor(1);
  //------------- ALam-K- ------------------
  TH1F *NumALamKchMBm3 = GetHisto(File_Bm3, "femtolist", "ALamKchM", "NumALamKchMKStarCF");
    NumALamKchMBm3->SetLineColor(1);
  TH1F *DenALamKchMBm3 = GetHisto(File_Bm3, "femtolist", "ALamKchM", "DenALamKchMKStarCF");
    DenALamKchMBm3->SetLineColor(1);
  TH1F *CfALamKchMBm3 = buildCF("CfALamKchMBm3","ALam-K- (B+ 1)",NumALamKchMBm3,DenALamKchMBm3,MinNormBin,MaxNormBin);
    CfALamKchMBm3->SetLineColor(1);

//__________________________________________________________________________________________________________________
  TList* CfList_LamKchP_BpTot = new TList();
    CfList_LamKchP_BpTot->Add(CfLamKchPBp1);
    CfList_LamKchP_BpTot->Add(CfLamKchPBp2);
  TList* NumList_LamKchP_BpTot = new TList();
    NumList_LamKchP_BpTot->Add(NumLamKchPBp1);
    NumList_LamKchP_BpTot->Add(NumLamKchPBp2);
  TH1F* CfLamKchPBpTot = Combine("CfLamKchPBpTot","Lam-K+ (B+ Tot)",CfList_LamKchP_BpTot,NumList_LamKchP_BpTot,MinNormBin,MaxNormBin);

  TList* CfList_LamKchM_BpTot = new TList();
    CfList_LamKchM_BpTot->Add(CfLamKchMBp1);
    CfList_LamKchM_BpTot->Add(CfLamKchMBp2);
  TList* NumList_LamKchM_BpTot = new TList();
    NumList_LamKchM_BpTot->Add(NumLamKchMBp1);
    NumList_LamKchM_BpTot->Add(NumLamKchMBp2);
  TH1F* CfLamKchMBpTot = Combine("CfLamKchMBpTot","Lam-K- (B+ Tot)",CfList_LamKchM_BpTot,NumList_LamKchM_BpTot,MinNormBin,MaxNormBin);

  TList* CfList_ALamKchP_BpTot = new TList();
    CfList_ALamKchP_BpTot->Add(CfALamKchPBp1);
    CfList_ALamKchP_BpTot->Add(CfALamKchPBp2);
  TList* NumList_ALamKchP_BpTot = new TList();
    NumList_ALamKchP_BpTot->Add(NumALamKchPBp1);
    NumList_ALamKchP_BpTot->Add(NumALamKchPBp2);
  TH1F* CfALamKchPBpTot = Combine("CfALamKchPBpTot","ALam-K+ (B+ Tot)",CfList_ALamKchP_BpTot,NumList_ALamKchP_BpTot,MinNormBin,MaxNormBin);

  TList* CfList_ALamKchM_BpTot = new TList();
    CfList_ALamKchM_BpTot->Add(CfALamKchMBp1);
    CfList_ALamKchM_BpTot->Add(CfALamKchMBp2);
  TList* NumList_ALamKchM_BpTot = new TList();
    NumList_ALamKchM_BpTot->Add(NumALamKchMBp1);
    NumList_ALamKchM_BpTot->Add(NumALamKchMBp2);
  TH1F* CfALamKchMBpTot = Combine("CfALamKchMBpTot","ALam-K- (B+ Tot)",CfList_ALamKchM_BpTot,NumList_ALamKchM_BpTot,MinNormBin,MaxNormBin);

  //--------------------------------------
  TList* CfList_LamKchP_BmTot = new TList();
    CfList_LamKchP_BmTot->Add(CfLamKchPBm1);
    CfList_LamKchP_BmTot->Add(CfLamKchPBm2);
    CfList_LamKchP_BmTot->Add(CfLamKchPBm3);
  TList* NumList_LamKchP_BmTot = new TList();
    NumList_LamKchP_BmTot->Add(NumLamKchPBm1);
    NumList_LamKchP_BmTot->Add(NumLamKchPBm2);
    NumList_LamKchP_BmTot->Add(NumLamKchPBm3);
  TH1F* CfLamKchPBmTot = Combine("CfLamKchPBmTot","Lam-K+ (B- Tot)",CfList_LamKchP_BmTot,NumList_LamKchP_BmTot,MinNormBin,MaxNormBin);

  TList* CfList_LamKchM_BmTot = new TList();
    CfList_LamKchM_BmTot->Add(CfLamKchMBm1);
    CfList_LamKchM_BmTot->Add(CfLamKchMBm2);
    CfList_LamKchM_BmTot->Add(CfLamKchMBm3);
  TList* NumList_LamKchM_BmTot = new TList();
    NumList_LamKchM_BmTot->Add(NumLamKchMBm1);
    NumList_LamKchM_BmTot->Add(NumLamKchMBm2);
    NumList_LamKchM_BmTot->Add(NumLamKchMBm3);
  TH1F* CfLamKchMBmTot = Combine("CfLamKchMBmTot","Lam-K- (B- Tot)",CfList_LamKchM_BmTot,NumList_LamKchM_BmTot,MinNormBin,MaxNormBin);

  TList* CfList_ALamKchP_BmTot = new TList();
    CfList_ALamKchP_BmTot->Add(CfALamKchPBm1);
    CfList_ALamKchP_BmTot->Add(CfALamKchPBm2);
    CfList_ALamKchP_BmTot->Add(CfALamKchPBm3);
  TList* NumList_ALamKchP_BmTot = new TList();
    NumList_ALamKchP_BmTot->Add(NumALamKchPBm1);
    NumList_ALamKchP_BmTot->Add(NumALamKchPBm2);
    NumList_ALamKchP_BmTot->Add(NumALamKchPBm3);
  TH1F* CfALamKchPBmTot = Combine("CfALamKchPBmTot","ALam-K+ (B- Tot)",CfList_ALamKchP_BmTot,NumList_ALamKchP_BmTot,MinNormBin,MaxNormBin);

  TList* CfList_ALamKchM_BmTot = new TList();
    CfList_ALamKchM_BmTot->Add(CfALamKchMBm1);
    CfList_ALamKchM_BmTot->Add(CfALamKchMBm2);
    CfList_ALamKchM_BmTot->Add(CfALamKchMBm3);
  TList* NumList_ALamKchM_BmTot = new TList();
    NumList_ALamKchM_BmTot->Add(NumALamKchMBm1);
    NumList_ALamKchM_BmTot->Add(NumALamKchMBm2);
    NumList_ALamKchM_BmTot->Add(NumALamKchMBm3);
  TH1F* CfALamKchMBmTot = Combine("CfALamKchMBmTot","ALam-K- (B- Tot)",CfList_ALamKchM_BmTot,NumList_ALamKchM_BmTot,MinNormBin,MaxNormBin);

  //--------------------------------------
  TList* CfList_LamKchP_Tot = Merge2Lists(CfList_LamKchP_BpTot,CfList_LamKchP_BmTot);
  TList* NumList_LamKchP_Tot = Merge2Lists(NumList_LamKchP_BpTot,NumList_LamKchP_BmTot);
  TH1F* CfLamKchPTot = Combine("CfLamKchPTot","Lam-K+ (Tot)",CfList_LamKchP_Tot,NumList_LamKchP_Tot,MinNormBin,MaxNormBin);

  TList* CfList_LamKchM_Tot = Merge2Lists(CfList_LamKchM_BpTot,CfList_LamKchM_BmTot);
  TList* NumList_LamKchM_Tot = Merge2Lists(NumList_LamKchM_BpTot,NumList_LamKchM_BmTot);
  TH1F* CfLamKchMTot = Combine("CfLamKchMTot","Lam-K- (Tot)",CfList_LamKchM_Tot,NumList_LamKchM_Tot,MinNormBin,MaxNormBin);

  TList* CfList_ALamKchP_Tot = Merge2Lists(CfList_ALamKchP_BpTot,CfList_ALamKchP_BmTot);
  TList* NumList_ALamKchP_Tot = Merge2Lists(NumList_ALamKchP_BpTot,NumList_ALamKchP_BmTot);
  TH1F* CfALamKchPTot = Combine("CfALamKchPTot","ALam-K+ (Tot)",CfList_ALamKchP_Tot,NumList_ALamKchP_Tot,MinNormBin,MaxNormBin);

  TList* CfList_ALamKchM_Tot = Merge2Lists(CfList_ALamKchM_BpTot,CfList_ALamKchM_BmTot);
  TList* NumList_ALamKchM_Tot = Merge2Lists(NumList_ALamKchM_BpTot,NumList_ALamKchM_BmTot);
  TH1F* CfALamKchMTot = Combine("CfALamKchMTot","ALam-K- (Tot)",CfList_ALamKchM_Tot,NumList_ALamKchM_Tot,MinNormBin,MaxNormBin);



//__________________________________________________________________________________________________________________

  TList* CfList_LamKchPM_Tot = Merge2Lists(CfList_LamKchP_Tot,CfList_LamKchM_Tot);
  TList* NumList_LamKchPM_Tot = Merge2Lists(NumList_LamKchP_Tot,NumList_LamKchM_Tot);
  TH1F* CfLamKchPMTot = Combine("CfLamKchPMTot","Lam-K+- (Tot)",CfList_LamKchPM_Tot,NumList_LamKchPM_Tot,MinNormBin,MaxNormBin);
  CfLamKchPMTot->SetLineColor(6);

  TList* CfList_ALamKchPM_Tot = Merge2Lists(CfList_ALamKchP_Tot,CfList_ALamKchM_Tot);
  TList* NumList_ALamKchPM_Tot = Merge2Lists(NumList_ALamKchP_Tot,NumList_ALamKchM_Tot);
  TH1F* CfALamKchPMTot = Combine("CfALamKchPMTot","ALam-K+- (Tot)",CfList_ALamKchPM_Tot,NumList_ALamKchPM_Tot,MinNormBin,MaxNormBin);
  CfALamKchPMTot->SetLineColor(6);

  TMultiGraph* graphLednicky = GetLednicky();
  Bool_t plotLednicky = kTRUE;

  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(2,2);
  gStyle->SetOptStat(0);
  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);

  //-----------------------------------------------
  c1->cd(1);
  TAxis *xax1 = CfLamKchPTot->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
  TAxis *yax1 = CfLamKchPTot->GetYaxis();
    yax1->SetRangeUser(0.9,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();
  CfLamKchPTot->SetLineColor(2);
  CfLamKchPTot->SetMarkerColor(2);
  CfLamKchPTot->SetMarkerStyle(20);
  CfLamKchPTot->SetMarkerSize(0.50);
  CfLamKchPTot->SetTitle("#LambdaK+ & #LambdaK-");
  CfLamKchPTot->Draw();

  CfLamKchMTot->SetLineColor(4);
  CfLamKchMTot->SetMarkerColor(4);
  CfLamKchMTot->SetMarkerStyle(20);
  CfLamKchMTot->SetMarkerSize(0.50);
  CfLamKchMTot->Draw("same");
  line->Draw();

  if(plotLednicky) graphLednicky->Draw();
  CfLamKchPTot->Draw("same");
  CfLamKchMTot->Draw("same");

  leg1 = new TLegend(0.60,0.12,0.89,0.32);
  leg1->SetFillColor(0);
  leg1->AddEntry(CfLamKchPTot, "#LambdaK+","lp");
  leg1->AddEntry(CfLamKchMTot, "#LambdaK-","lp");
  leg1->Draw();

  //-----------------------------------------------
  c1->cd(2);
  TAxis *xax2 = CfALamKchPTot->GetXaxis();
    xax2->SetTitle("k* (GeV/c)");
    xax2->SetTitleSize(0.05);
    xax2->SetTitleOffset(1.0);
  TAxis *yax2 = CfALamKchPTot->GetYaxis();
    yax2->SetRangeUser(0.9,1.1);
    yax2->SetTitle("C(k*)");
    yax2->SetTitleSize(0.05);
    yax2->SetTitleOffset(1.0);
    yax2->CenterTitle();
  CfALamKchPTot->SetLineColor(4);
  CfALamKchPTot->SetMarkerColor(4);
  CfALamKchPTot->SetMarkerStyle(20);
  CfALamKchPTot->SetMarkerSize(0.50);
  CfALamKchPTot->SetTitle("#bar{#Lambda}K+ & #bar{#Lambda}K-");
  CfALamKchPTot->Draw();

  CfALamKchMTot->SetLineColor(2);
  CfALamKchMTot->SetMarkerColor(2);
  CfALamKchMTot->SetMarkerStyle(20);
  CfALamKchMTot->SetMarkerSize(0.50);
  CfALamKchMTot->Draw("same");
  line->Draw();
  if(plotLednicky) graphLednicky->Draw();

  leg2 = new TLegend(0.60,0.12,0.89,0.32);
  leg2->SetFillColor(0);
  leg2->AddEntry(CfALamKchMTot, "#bar{#Lambda}K-","lp");
  leg2->AddEntry(CfALamKchPTot, "#bar{#Lambda}K+","lp");
  leg2->Draw();

  //-----------------------------------------------
  c1->cd(3);
  TAxis *xax3 = CfLamK0Tot->GetXaxis();
    xax3->SetTitle("k* (GeV/c)");
    xax3->SetTitleSize(0.05);
    xax3->SetTitleOffset(1.0);
    //xax3->SetRangeUser(0.0,0.2);
  TAxis *yax3 = CfLamK0Tot->GetYaxis();
    yax3->SetRangeUser(0.9,1.1);
    yax3->SetTitle("C(k*)");
    yax3->SetTitleSize(0.05);
    yax3->SetTitleOffset(1.0);
    yax3->CenterTitle();
  CfLamK0Tot->SetLineColor(1);
  CfLamK0Tot->SetMarkerColor(1);
  CfLamK0Tot->SetMarkerStyle(20);
  CfLamK0Tot->SetMarkerSize(0.50);
  CfLamK0Tot->SetTitle("#LambdaK^{0} & #LambdaK+-");
  CfLamK0Tot->Draw();

  CfLamKchPMTot->SetLineColor(6);
  CfLamKchPMTot->SetMarkerColor(6);
  CfLamKchPMTot->SetMarkerStyle(20);
  CfLamKchPMTot->SetMarkerSize(0.50);
  CfLamKchPMTot->Draw("same");
  line->Draw();
  if(plotLednicky) graphLednicky->Draw();

  //---Clones so I can control the interval over which the Chi2Test occurs
  //---while still plotting the full range
  TH1F* CfLamK0TotCLONE = CfLamK0Tot->Clone("CfLamK0TotCLONE");
  CfLamK0TotCLONE->GetXaxis()->SetRangeUser(0.01,0.2);
  TH1F* CfLamKchPMTotCLONE = CfLamKchPMTot->Clone("CfLamKchPMTotCLONE");
  double pLamK0 = CfLamK0TotCLONE->Chi2Test(CfLamKchPMTotCLONE,"NORM");
  cout << "pLamK0 = " << pLamK0 << endl;

  leg3 = new TLegend(0.50,0.12,0.89,0.32);
  leg3->SetFillColor(0);
  leg3->AddEntry(CfLamK0Tot, "#LambdaK^{0}", "lp");
  leg3->AddEntry(CfLamKchPMTot, "Combined #LambdaK+ & #LambdaK-","lp");
  leg3->Draw();

  text3 = new TPaveText(0.70,0.60,0.89,0.75,"NDC");
  char buffer[50];
  sprintf(buffer, "p = %.9f",pLamK0);
  text3->AddText(buffer);
  text3->Draw();

  //-----------------------------------------------
  c1->cd(4);
  TAxis *xax4 = CfALamK0Tot->GetXaxis();
    xax4->SetTitle("k* (GeV/c)");
    xax4->SetTitleSize(0.05);
    xax4->SetTitleOffset(1.0);
    //xax4->SetRangeUser(0.0,0.2);
  TAxis *yax4 = CfALamK0Tot->GetYaxis();
    yax4->SetRangeUser(0.9,1.1);
    yax4->SetTitle("C(k*)");
    yax4->SetTitleSize(0.05);
    yax4->SetTitleOffset(1.0);
    yax4->CenterTitle();
  CfALamK0Tot->SetLineColor(1);
  CfALamK0Tot->SetMarkerColor(1);
  CfALamK0Tot->SetMarkerStyle(20);
  CfALamK0Tot->SetMarkerSize(0.50);
  CfALamK0Tot->SetTitle("#bar{#Lambda}K^{0} & #bar{#Lambda}K+-");
  CfALamK0Tot->Draw();

  CfALamKchPMTot->SetLineColor(6);
  CfALamKchPMTot->SetMarkerColor(6);
  CfALamKchPMTot->SetMarkerStyle(20);
  CfALamKchPMTot->SetMarkerSize(0.50);
  CfALamKchPMTot->Draw("same");
  line->Draw();
  if(plotLednicky) graphLednicky->Draw();

  //---Clones so I can control the interval over which the Chi2Test occurs
  //---while still plotting the full range
  TH1F* CfALamK0TotCLONE = CfALamK0Tot->Clone("CfALamK0TotCLONE");
  CfALamK0TotCLONE->GetXaxis()->SetRangeUser(0.01,0.2);
  TH1F* CfALamKchPMTotCLONE = CfALamKchPMTot->Clone("CfALamKchPMTotCLONE");
  double pALamK0 = CfALamK0TotCLONE->Chi2Test(CfALamKchPMTotCLONE,"NORM");
  cout << "pALamK0 = " << pALamK0 << endl;

  leg4 = new TLegend(0.50,0.12,0.89,0.32);
  leg4->SetFillColor(0);
  leg4->AddEntry(CfALamK0Tot, "#bar{#Lambda}K^{0}", "lp");
  leg4->AddEntry(CfLamKchPMTot, "Combined #bar{#Lambda}K- & #bar{#Lambda}K+","lp");
  leg4->Draw();

  text4 = new TPaveText(0.70,0.60,0.89,0.75,"NDC");
  char buffer[50];
  sprintf(buffer, "p = %.9f",pALamK0);
  text4->AddText(buffer);
  text4->Draw();
  

  c1->SaveAs("compCorrK0Kch_LedVar.pdf");


}
