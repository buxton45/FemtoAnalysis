#include "buildAllcLamK0.cxx"

void AnalyzeAnalysiscLamK0()
{
  buildAllcLamK0 *cLamK0Analysis = new buildAllcLamK0();

  cLamK0Analysis->SetMinNormBinCF(60);
  cLamK0Analysis->SetMaxNormBinCF(75);

  cLamK0Analysis->SetMinNormBinAvgSepCF(150);
  cLamK0Analysis->SetMaxNormBinAvgSepCF(200);

  cLamK0Analysis->SetHistograms("Resultsgrid_cLamK0_CentBins_Bp1NEW.root");
  cLamK0Analysis->SetHistograms("Resultsgrid_cLamK0_CentBins_Bp2NEW.root");
  cLamK0Analysis->SetHistograms("Resultsgrid_cLamK0_CentBins_Bm1NEW.root");
  cLamK0Analysis->SetHistograms("Resultsgrid_cLamK0_CentBins_Bm2NEW.root");
  cLamK0Analysis->SetHistograms("Resultsgrid_cLamK0_CentBins_Bm3NEW.root");

//________________________________________________________________________________________________________________

/*
  cLamK0Analysis->buildCorrCombined(true,false,false);

  cout << "I AM HERE (1)!!!!!!!" << endl;
  TList* CfList_LamK0_BpTot = cLamK0Analysis->GetCfList_LamK0_BpTot();
  cout << CfList_LamK0_BpTot->GetEntries() << endl;

  TIter iter(CfList_LamK0_BpTot);
  TObject* obj;
  while(obj = (TObject*)iter.Next())
  {
    cout << obj->GetName() << endl;
    cout << "WORKING(1)" << endl;
  }
  cout << "NOW I AM HERE(1)!!!!" << endl << endl;

  //-----
  cout << "I AM HERE(2)!!!!!!!" << endl;
  TList* CfList_ALamK0_BpTot = cLamK0Analysis->GetCfList_ALamK0_BpTot();
  cout << CfList_ALamK0_BpTot->GetEntries() << endl;

  TIter iter2(CfList_ALamK0_BpTot);
  TObject* obj2;
  while(obj2 = (TObject*)iter2.Next())
  {
    cout << obj2->GetName() << endl;
    cout << "WORKING(2)" << endl;
  }
  cout << "NOW I AM HERE(2)!!!!" << endl << endl;

  //----------

  cout << "I AM HERE (1)!!!!!!!" << endl;
  TList* NumList_LamK0_BpTot = cLamK0Analysis->GetNumList_LamK0_BpTot();
  cout << NumList_LamK0_BpTot->GetEntries() << endl;

  TIter iter(NumList_LamK0_BpTot);
  TObject* obj;
  while(obj = (TObject*)iter.Next())
  {
    cout << obj->GetName() << endl;
    cout << "WORKING(1)" << endl;
  }
  cout << "NOW I AM HERE(1)!!!!" << endl << endl;

  //-----
  cout << "I AM HERE(2)!!!!!!!" << endl;
  TList* NumList_ALamK0_BpTot = cLamK0Analysis->GetNumList_ALamK0_BpTot();
  cout << NumList_ALamK0_BpTot->GetEntries() << endl;

  TIter iter2(NumList_ALamK0_BpTot);
  TObject* obj2;
  while(obj2 = (TObject*)iter2.Next())
  {
    cout << obj2->GetName() << endl;
    cout << "WORKING(2)" << endl;
  }
  cout << "NOW I AM HERE(2)!!!!" << endl << endl;

  //----------

  cout << "I AM HERE (1)!!!!!!!" << endl;
  TList* DenList_LamK0_BpTot = cLamK0Analysis->GetDenList_LamK0_BpTot();
  cout << DenList_LamK0_BpTot->GetEntries() << endl;

  TIter iter(DenList_LamK0_BpTot);
  TObject* obj;
  while(obj = (TObject*)iter.Next())
  {
    cout << obj->GetName() << endl;
    cout << "WORKING(1)" << endl;
  }
  cout << "NOW I AM HERE(1)!!!!" << endl << endl;

  //-----
  cout << "I AM HERE(2)!!!!!!!" << endl;
  TList* DenList_ALamK0_BpTot = cLamK0Analysis->GetDenList_ALamK0_BpTot();
  cout << DenList_ALamK0_BpTot->GetEntries() << endl;

  TIter iter2(DenList_ALamK0_BpTot);
  TObject* obj2;
  while(obj2 = (TObject*)iter2.Next())
  {
    cout << obj2->GetName() << endl;
    cout << "WORKING(2)" << endl;
  }
  cout << "NOW I AM HERE(2)!!!!" << endl << endl;
*/

//________________________________________________________________________________________________________________

/*
  TH1F* Numerator2 = cLamK0Analysis->GetCf("Bp1","ALamK0","Num");
  TH1F* Denominator2 = cLamK0Analysis->GetCf("Bp1","ALamK0","Den");
  TH1F* CF2 = cLamK0Analysis->GetCf("Bp1","ALamK0","Cf");
*/

//________________________________________________________________________________________________________________

/*
  TH1F* PosPosAvgSepCF_LamK0 = cLamK0Analysis->GetAvgSepCf("Bp1","LamK0","PosPos","Cf");
  TH1F* PosNegAvgSepCF_LamK0 = cLamK0Analysis->GetAvgSepCf("Bp1","LamK0","PosNeg","Cf");
  TH1F* NegPosAvgSepCF_LamK0 = cLamK0Analysis->GetAvgSepCf("Bp1","LamK0","NegPos","Cf");
  TH1F* NegNegAvgSepCF_LamK0 = cLamK0Analysis->GetAvgSepCf("Bp1","LamK0","NegNeg","Cf");

  TH1F* PosPosAvgSepCF_ALamK0 = cLamK0Analysis->GetAvgSepCf("Bp1","ALamK0","PosPos","Cf");
  TH1F* PosNegAvgSepCF_ALamK0 = cLamK0Analysis->GetAvgSepCf("Bp1","ALamK0","PosNeg","Cf");
  TH1F* NegPosAvgSepCF_ALamK0 = cLamK0Analysis->GetAvgSepCf("Bp1","ALamK0","NegPos","Cf");
  TH1F* NegNegAvgSepCF_ALamK0 = cLamK0Analysis->GetAvgSepCf("Bp1","ALamK0","NegNeg","Cf");


  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(2,2);

  c1->cd(1);
  PosPosAvgSepCF_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  PosPosAvgSepCF_LamK0->Draw();

  c1->cd(2);
  PosNegAvgSepCF_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  PosNegAvgSepCF_LamK0->Draw();

  c1->cd(3);
  NegPosAvgSepCF_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  NegPosAvgSepCF_LamK0->Draw();

  c1->cd(4);
  NegNegAvgSepCF_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  NegNegAvgSepCF_LamK0->Draw();

  TCanvas* c2 = new TCanvas("c2","Plotting Canvas");
  c2->Divide(2,2);

  c2->cd(1);
  PosPosAvgSepCF_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  PosPosAvgSepCF_ALamK0->Draw();

  c2->cd(2);
  PosNegAvgSepCF_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  PosNegAvgSepCF_ALamK0->Draw();

  c2->cd(3);
  NegPosAvgSepCF_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  NegPosAvgSepCF_ALamK0->Draw();

  c2->cd(4);
  NegNegAvgSepCF_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  NegNegAvgSepCF_ALamK0->Draw();
*/

//________________________________________________________________________________________________________________

/*
  TH1F* LambdaPurity = cLamK0Analysis->GetHistoClone("Resultsgrid_cLamK0_CentBins_Bp1NEW.root","LamK0","LambdaPurity");
  TH1F* K0Short1Purity = cLamK0Analysis->GetHistoClone("Resultsgrid_cLamK0_CentBins_Bp1NEW.root","LamK0","K0ShortPurity1");
  TH1F* AntiLambdaPurity = cLamK0Analysis->GetHistoClone("Resultsgrid_cLamK0_CentBins_Bp1NEW.root","ALamK0","AntiLambdaPurity");
  TH1F* K0Short2Purity = cLamK0Analysis->GetHistoClone("Resultsgrid_cLamK0_CentBins_Bp1NEW.root","ALamK0","K0ShortPurity1");

  const double LambdaMass = 1.115683, KaonMass = 0.493677;

  double LamBgFitLow[2];
    LamBgFitLow[0] = 1.09;
    LamBgFitLow[1] = 1.102;
  double LamBgFitHigh[2];
    LamBgFitHigh[0] = 1.130;
    LamBgFitHigh[1] = LambdaPurity->GetBinLowEdge(LambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  double LamROI[2];
    LamROI[0] = LambdaMass-0.0038;
    LamROI[1] = LambdaMass+0.0038;

  double K0Short1BgFitLow[2];
    K0Short1BgFitLow[0] = K0Short1Purity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
    K0Short1BgFitLow[1] = 0.452;
  double K0Short1BgFitHigh[2];
    K0Short1BgFitHigh[0] = 0.536;
    K0Short1BgFitHigh[1] = K0Short1Purity->GetBinLowEdge(K0Short1Purity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)
  double K0Short1ROI[2];
    K0Short1ROI[0] = KaonMass-0.013677;
    K0Short1ROI[1] = KaonMass+0.020323;

  TList* LamList = cLamK0Analysis->CalculatePurity("LambdaPurity",LambdaPurity,LamBgFitLow,LamBgFitHigh,LamROI);
  TList* K0Short1List = cLamK0Analysis->CalculatePurity("K0Short1Purity",K0Short1Purity,K0Short1BgFitLow,K0Short1BgFitHigh,K0Short1ROI);
  TList* ALamList = cLamK0Analysis->CalculatePurity("AntiLambdaPurity",AntiLambdaPurity,LamBgFitLow,LamBgFitHigh,LamROI);
  TList* K0Short2List = cLamK0Analysis->CalculatePurity("K0Short2Purity",K0Short2Purity,K0Short1BgFitLow,K0Short1BgFitHigh,K0Short1ROI);

  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(2,4);
  //-----
  c1->cd(1);
  cLamK0Analysis->PurityDrawAll(LambdaPurity,LamList,false);
  c1->cd(2);
  cLamK0Analysis->PurityDrawAll(LambdaPurity,LamList,true);
  //-----
  c1->cd(3);
  cLamK0Analysis->PurityDrawAll(K0Short1Purity,K0Short1List,false);
  c1->cd(4);
  cLamK0Analysis->PurityDrawAll(K0Short1Purity,K0Short1List,true);
  //-----
  c1->cd(5);
  cLamK0Analysis->PurityDrawAll(AntiLambdaPurity,ALamList,false);
  c1->cd(6);
  cLamK0Analysis->PurityDrawAll(AntiLambdaPurity,ALamList,true);
  //-----
  c1->cd(7);
  cLamK0Analysis->PurityDrawAll(K0Short2Purity,K0Short2List,false);
  c1->cd(8);
  cLamK0Analysis->PurityDrawAll(K0Short2Purity,K0Short2List,true);
*/

//________________________________________________________________________________________________________________


  cLamK0Analysis->buildCorrCombined(true,true,true);

  TH1F* CfPosPosAvgSepCfLamK0Tot = cLamK0Analysis->GetCfPosPosAvgSepCfLamK0Tot();
  TH1F* CfPosNegAvgSepCfLamK0Tot = cLamK0Analysis->GetCfPosNegAvgSepCfLamK0Tot();
  TH1F* CfNegPosAvgSepCfLamK0Tot = cLamK0Analysis->GetCfNegPosAvgSepCfLamK0Tot();
  TH1F* CfNegNegAvgSepCfLamK0Tot = cLamK0Analysis->GetCfNegNegAvgSepCfLamK0Tot();

  TH1F* CfPosPosAvgSepCfALamK0Tot = cLamK0Analysis->GetCfPosPosAvgSepCfALamK0Tot();
  TH1F* CfPosNegAvgSepCfALamK0Tot = cLamK0Analysis->GetCfPosNegAvgSepCfALamK0Tot();
  TH1F* CfNegPosAvgSepCfALamK0Tot = cLamK0Analysis->GetCfNegPosAvgSepCfALamK0Tot();
  TH1F* CfNegNegAvgSepCfALamK0Tot = cLamK0Analysis->GetCfNegNegAvgSepCfALamK0Tot();

  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(2,2);

  c1->cd(1);
  CfPosPosAvgSepCfLamK0Tot->GetYaxis()->SetRangeUser(-0.5,5.);
  CfPosPosAvgSepCfLamK0Tot->Draw();
  c1->cd(2);
  CfPosNegAvgSepCfLamK0Tot->GetYaxis()->SetRangeUser(-0.5,5.);
  CfPosNegAvgSepCfLamK0Tot->Draw();
  c1->cd(3);
  CfNegPosAvgSepCfLamK0Tot->GetYaxis()->SetRangeUser(-0.5,5.);
  CfNegPosAvgSepCfLamK0Tot->Draw();
  c1->cd(4);
  CfNegNegAvgSepCfLamK0Tot->GetYaxis()->SetRangeUser(-0.5,5.);
  CfNegNegAvgSepCfLamK0Tot->Draw();


  TCanvas* c2 = new TCanvas("c2","Plotting Canvas");
  c2->Divide(2,2);

  c2->cd(1);
  CfPosPosAvgSepCfALamK0Tot->GetYaxis()->SetRangeUser(-0.5,5.);
  CfPosPosAvgSepCfALamK0Tot->Draw();
  c2->cd(2);
  CfPosNegAvgSepCfALamK0Tot->GetYaxis()->SetRangeUser(-0.5,5.);
  CfPosNegAvgSepCfALamK0Tot->Draw();
  c2->cd(3);
  CfNegPosAvgSepCfALamK0Tot->GetYaxis()->SetRangeUser(-0.5,5.);
  CfNegPosAvgSepCfALamK0Tot->Draw();
  c2->cd(4);
  CfNegNegAvgSepCfALamK0Tot->GetYaxis()->SetRangeUser(-0.5,5.);
  CfNegNegAvgSepCfALamK0Tot->Draw();


//________________________________________________________________________________________________________________

}
