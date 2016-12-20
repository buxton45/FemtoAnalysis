///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamK02                                                        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "buildAllcLamK02.h"

#ifdef __ROOT__
ClassImp(buildAllcLamK02)
#endif

//________________________________________________________________________________________________________________
buildAllcLamK02::buildAllcLamK02(vector<TString> &aVectorOfFileNames, TString aDirNameLamK0, TString aDirNameALamK0):
  //General stuff----------------
  fDirNameLamK0(aDirNameLamK0),
  fDirNameALamK0(aDirNameALamK0),

  //KStar CF------------------
  fMinNormBinCF(60),
  fMaxNormBinCF(75),

  //Average Separation CF------------------
  fMinNormBinAvgSepCF(150),
  fMaxNormBinAvgSepCF(200)



{
  SetVectorOfFileNames(aVectorOfFileNames);
  SetAnalysisDirectories();


  //Purity calculations------------------
  fLamBgFitLow[0] = 1.09;
  fLamBgFitLow[1] = 1.102;
  fLamBgFitHigh[0] = 1.130;
  fLamBgFitHigh[1] = 1.15068;
  fLamROI[0] = LambdaMass-0.0038;
  fLamROI[1] = LambdaMass+0.0038;

  fK0Short1BgFitLow[0] = 0.423677;
  fK0Short1BgFitLow[1] = 0.452;
  fK0Short1BgFitHigh[0] = 0.536;
  fK0Short1BgFitHigh[1] = 0.563677;
  fK0Short1ROI[0] = KaonMass-0.013677;
  fK0Short1ROI[1] = KaonMass+0.020323;


}

//________________________________________________________________________________________________________________
TH1F* buildAllcLamK02::buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int aMinNormBin, int aMaxNormBin)
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
TH1F* buildAllcLamK02::CombineCFs(TString aReturnName, TString aReturnTitle, vector<TH1F*> &aCfCollection, vector<TH1F*> &aNumCollection, int aMinNormBin, int aMaxNormBin)
{
  double scale = 0.;
  int counter = 0;
  double temp = 0.;

  int SizeOfCfCollection = aCfCollection.size();
  int SizeOfNumCollection = aNumCollection.size();

  if(SizeOfCfCollection != SizeOfNumCollection) {cout << "ERROR: In CombineCFs, the CfCollection and NumCollection ARE NOT EQUAL IN SIZE!!!!" << endl;}

  TH1F* ReturnCf = (TH1F*)aCfCollection[0]->Clone(aReturnName);
  ReturnCf->Sumw2();
  ReturnCf->SetTitle(aReturnTitle);
    cout << "Name: " << ReturnCf->GetName() << endl;

  TH1F* Num1 = aNumCollection[0];
  temp = Num1->Integral(aMinNormBin,aMaxNormBin);
    cout << "Name: " << Num1->GetName() << "  NumScale: " << Num1->Integral(aMinNormBin,aMaxNormBin) << endl;
  scale+=temp;
  counter++;

  ReturnCf->Scale(temp);

  for(int i=1; i<SizeOfCfCollection; i++)
  {
    cout << "Name: " << aCfCollection[i]->GetName() << endl;
    temp = aNumCollection[i]->Integral(aMinNormBin,aMaxNormBin);
    cout << "Name: " << aNumCollection[i]->GetName() << " NumScale: " << aNumCollection[i]->Integral(aMinNormBin,aMaxNormBin) << endl;
    ReturnCf->Add(aCfCollection[i],temp);
    scale += temp;
    counter ++;
  }
  cout << "SCALE = " << scale << endl;
  cout << "counter = " << counter << endl;
  ReturnCf->Scale(1./scale);

  return ReturnCf;

}



//________________________________________________________________________________________________________________
void buildAllcLamK02::SetVectorOfFileNames(vector<TString> &aVectorOfNames)
{
  int InputVectorSize = aVectorOfNames.size();
  if(InputVectorSize != 5) {cout << "MISSING FILE(S)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl << endl <<  "INPUT VECTOR HAS SIZE: " << InputVectorSize << endl;}

  for(int i=0; i<InputVectorSize; i++)
  {
    fVectorOfFileNames.push_back(aVectorOfNames[i]);
  }

  //Double check that all names were loaded correctly
  int OutputVectorSize = fVectorOfFileNames.size();
  cout << "fVectorOfFileNames now has size " << OutputVectorSize << endl;
  cout << "And contains..." << endl;
  for(int i=0; i<OutputVectorSize; i++)
  {
    cout << fVectorOfFileNames[i] << endl;
  }

}


//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK02::ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName)
{
  TFile f1(aFileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  TObjArray *ReturnArray = (TObjArray*)femtolist->FindObject(aDirectoryName);

  return ReturnArray;
}

//________________________________________________________________________________________________________________
void buildAllcLamK02::SetAnalysisDirectories()
{
  TString FileNameBp1, FileNameBp2, FileNameBm1, FileNameBm2, FileNameBm3;
  //The files should be ordered Bp1, Bp2, Bm1, Bm2, Bm3...but if they are not, the following for loop
  // will pick out the correct file names to associate with the directories

  for(unsigned int i=0; i<fVectorOfFileNames.size(); i++)
  {
    if(fVectorOfFileNames[i].Contains("Bp1")){FileNameBp1 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bp2")){FileNameBp2 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm1")){FileNameBm1 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm2")){FileNameBm2 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm3")){FileNameBm3 = fVectorOfFileNames[i];}
    else{cout << "ERROR: In SetAnalysisDirectories....a file name was found to NOT contain Bp1,Bp2,Bm1,Bm2,or Bm3!!!!!" << endl;}
  }

  fDirLamK0Bp1 = ConnectAnalysisDirectory(FileNameBp1,fDirNameLamK0);
  fDirALamK0Bp1 = ConnectAnalysisDirectory(FileNameBp1,fDirNameALamK0);

  fDirLamK0Bp2 = ConnectAnalysisDirectory(FileNameBp2,fDirNameLamK0);
  fDirALamK0Bp2 = ConnectAnalysisDirectory(FileNameBp2,fDirNameALamK0);

  fDirLamK0Bm1 = ConnectAnalysisDirectory(FileNameBm1,fDirNameLamK0);
  fDirALamK0Bm1 = ConnectAnalysisDirectory(FileNameBm1,fDirNameALamK0);

  fDirLamK0Bm2 = ConnectAnalysisDirectory(FileNameBm2,fDirNameLamK0);
  fDirALamK0Bm2 = ConnectAnalysisDirectory(FileNameBm2,fDirNameALamK0);

  fDirLamK0Bm3 = ConnectAnalysisDirectory(FileNameBm3,fDirNameLamK0);
  fDirALamK0Bm3 = ConnectAnalysisDirectory(FileNameBm3,fDirNameALamK0);
}

//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK02::GetAnalysisDirectory(TString aFile, TString aDirectoryName)
{
  if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bp1;}
  else if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bp1;}

  else if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bp2;}
  else if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bp2;}

  else if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bm1;}
  else if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bm1;}

  else if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bm2;}
  else if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bm2;}

  else if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bm3;}
  else if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bm3;}

  else
  {
    cout << "ERROR in GetAnalysisDirectory:  No directory to return!!!!!" << endl;
    return 0;
  }

}


//________________________________________________________________________________________________________________
TH1F* buildAllcLamK02::GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName)
{
  TH1F *ReturnHisto = (TH1F*)aAnalysisDirectory->FindObject(aHistoName);
  TH1F *ReturnHistoClone = (TH1F*)ReturnHisto->Clone(aCloneHistoName);
  ReturnHistoClone->SetDirectory(0);
  ReturnHistoClone->Sumw2();

  return (TH1F*)ReturnHistoClone;
}


//________________________________________________________________________________________________________________
vector<TH1F*> buildAllcLamK02::LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName)
{
  vector<TH1F*> ReturnCollection;
  ReturnCollection.clear();  //make sure initiall empty


  if(aDirectoryName.EqualTo(fDirNameLamK0))
  {
    ReturnCollection.push_back(GetHistoClone(fDirLamK0Bp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection.push_back(GetHistoClone(fDirLamK0Bp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection.push_back(GetHistoClone(fDirLamK0Bm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection.push_back(GetHistoClone(fDirLamK0Bm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection.push_back(GetHistoClone(fDirLamK0Bm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamK0))
  {
    ReturnCollection.push_back(GetHistoClone(fDirALamK0Bp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection.push_back(GetHistoClone(fDirALamK0Bp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection.push_back(GetHistoClone(fDirALamK0Bm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection.push_back(GetHistoClone(fDirALamK0Bm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection.push_back(GetHistoClone(fDirALamK0Bm3,aHistoName,aHistoName+"_Bm3"));
  }

  else{cout << "ERROR IN LoadCollectionOfHistograms!!!!!!!!!!!!!!!!" << endl;}

  return ReturnCollection;

}

//________________________________________________________________________________________________________________
vector<TH1F*> buildAllcLamK02::BuildCollectionOfCfs(vector<TH1F*> &aNumCollection, vector<TH1F*> &aDenCollection, int aMinNormBin, int aMaxNormBin)
{
  vector<TH1F*> ReturnCollection;
  ReturnCollection.clear();  //make sure initiall empty

  int SizeOfNumCollection = aNumCollection.size();
  int SizeOfDenCollection = aDenCollection.size();

  if( (SizeOfNumCollection != 5) && (SizeOfDenCollection != 5) ) {cout << "In BuildCollectionOfCfs, the input collection sizes DO NOT EQUAL 5!!!!!" << endl;}
  if( SizeOfNumCollection != SizeOfDenCollection ) {cout << "In BuildCollectionOfCfs, the NumCollection and DenCollection DO NOT HAVE EQUAL SIZES!!!!!!!" << endl;}

  for(int i=0; i<SizeOfNumCollection; i++)
  {
    ReturnCollection.push_back(buildCF("CF","CF",aNumCollection[i],aDenCollection[i],aMinNormBin,aMaxNormBin));
  }

  return ReturnCollection;


}

//________________________________________________________________________________________________________________
void buildAllcLamK02::BuildCFCollections()
{
  TString NumName_LamK0 = "NumKStarCF_LamK0";
  TString DenName_LamK0 = "DenKStarCF_LamK0";

  TString NumName_ALamK0 = "NumKStarCF_ALamK0";
  TString DenName_ALamK0 = "DenKStarCF_ALamK0";

  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamK0.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamK0.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamK0.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  NumName_LamK0+=CentralityTag;
  DenName_LamK0+=CentralityTag;
  NumName_ALamK0+=CentralityTag;
  DenName_ALamK0+=CentralityTag;
  //----------------------------------------------------------------------

  fNumCollection_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_LamK0);
  fDenCollection_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_LamK0);
  fCfCollection_LamK0 = BuildCollectionOfCfs(fNumCollection_LamK0,fDenCollection_LamK0,fMinNormBinCF,fMaxNormBinCF);
  fCf_LamK0_Tot = CombineCFs("fCf_LamK0_Tot","Lam-K0 (Tot)",fCfCollection_LamK0,fNumCollection_LamK0,fMinNormBinCF,fMaxNormBinCF);

  fNumCollection_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_ALamK0);
  fDenCollection_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_ALamK0);
  fCfCollection_ALamK0 = BuildCollectionOfCfs(fNumCollection_ALamK0,fDenCollection_ALamK0,fMinNormBinCF,fMaxNormBinCF);
  fCf_ALamK0_Tot = CombineCFs("fCf_ALamK0_Tot","ALam-K0 (Tot)",fCfCollection_ALamK0,fNumCollection_ALamK0,fMinNormBinCF,fMaxNormBinCF);

  //--------------BpTot-----------------------------------------------------
  vector<TH1F*> fNumCollection_LamK0_BpTot;
    fNumCollection_LamK0_BpTot.push_back(fNumCollection_LamK0[0]);
    fNumCollection_LamK0_BpTot.push_back(fNumCollection_LamK0[1]);
  vector<TH1F*> fCfCollection_LamK0_BpTot;
    fCfCollection_LamK0_BpTot.push_back(fCfCollection_LamK0[0]);
    fCfCollection_LamK0_BpTot.push_back(fCfCollection_LamK0[1]);
  fCf_LamK0_BpTot = CombineCFs("fCf_LamK0_BpTot","Lam-K0 (B+ Tot)",fCfCollection_LamK0_BpTot,fNumCollection_LamK0_BpTot,fMinNormBinCF,fMaxNormBinCF);

  vector<TH1F*> fNumCollection_ALamK0_BpTot;
    fNumCollection_ALamK0_BpTot.push_back(fNumCollection_ALamK0[0]);
    fNumCollection_ALamK0_BpTot.push_back(fNumCollection_ALamK0[1]);
  vector<TH1F*> fCfCollection_ALamK0_BpTot;
    fCfCollection_ALamK0_BpTot.push_back(fCfCollection_ALamK0[0]);
    fCfCollection_ALamK0_BpTot.push_back(fCfCollection_ALamK0[1]);
  fCf_ALamK0_BpTot = CombineCFs("fCf_ALamK0_BpTot","ALam-K0 (B+ Tot)",fCfCollection_ALamK0_BpTot,fNumCollection_ALamK0_BpTot,fMinNormBinCF,fMaxNormBinCF);


  //--------------BmTot-----------------------------------------------------
  vector<TH1F*> fNumCollection_LamK0_BmTot;
    fNumCollection_LamK0_BmTot.push_back(fNumCollection_LamK0[2]);
    fNumCollection_LamK0_BmTot.push_back(fNumCollection_LamK0[3]);
    fNumCollection_LamK0_BmTot.push_back(fNumCollection_LamK0[4]);
  vector<TH1F*> fCfCollection_LamK0_BmTot;
    fCfCollection_LamK0_BmTot.push_back(fCfCollection_LamK0[2]);
    fCfCollection_LamK0_BmTot.push_back(fCfCollection_LamK0[3]);
    fCfCollection_LamK0_BmTot.push_back(fCfCollection_LamK0[4]);
  fCf_LamK0_BmTot = CombineCFs("fCf_LamK0_BmTot","Lam-K0 (B- Tot)",fCfCollection_LamK0_BmTot,fNumCollection_LamK0_BmTot,fMinNormBinCF,fMaxNormBinCF);

  vector<TH1F*> fNumCollection_ALamK0_BmTot;
    fNumCollection_ALamK0_BmTot.push_back(fNumCollection_ALamK0[2]);
    fNumCollection_ALamK0_BmTot.push_back(fNumCollection_ALamK0[3]);
    fNumCollection_ALamK0_BmTot.push_back(fNumCollection_ALamK0[4]);
  vector<TH1F*> fCfCollection_ALamK0_BmTot;
    fCfCollection_ALamK0_BmTot.push_back(fCfCollection_ALamK0[2]);
    fCfCollection_ALamK0_BmTot.push_back(fCfCollection_ALamK0[3]);
    fCfCollection_ALamK0_BmTot.push_back(fCfCollection_ALamK0[4]);
  fCf_ALamK0_BmTot = CombineCFs("fCf_ALamK0_BmTot","ALam-K0 (B- Tot)",fCfCollection_ALamK0_BmTot,fNumCollection_ALamK0_BmTot,fMinNormBinCF,fMaxNormBinCF);

}

//________________________________________________________________________________________________________________
void buildAllcLamK02::DrawFinalCFs(TCanvas *aCanvas)
{
  aCanvas->cd();
  gStyle->SetOptStat(0);

  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aCfLamK0 = (TH1F*)fCf_LamK0_Tot->Clone();
  TH1F *aCfALamK0 = (TH1F*)fCf_ALamK0_Tot->Clone();

  TAxis *xax1 = aCfLamK0->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
  TAxis *yax1 = aCfLamK0->GetYaxis();
    yax1->SetRangeUser(0.8,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();

  aCfLamK0->SetMarkerStyle(20);
  aCfLamK0->SetMarkerSize(1);
  aCfLamK0->SetMarkerColor(1);
  aCfLamK0->SetLineColor(1);
  aCfLamK0->SetTitle("#Lambda(#bar{#Lambda})-K^{0}");

  aCfALamK0->SetMarkerStyle(20);
  aCfALamK0->SetMarkerSize(1);
  aCfALamK0->SetMarkerColor(2);
  aCfALamK0->SetLineColor(2);

  aCfLamK0->Draw();
  aCfALamK0->Draw("same");

  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);
  line->Draw();

  TLegend *leg1 = new TLegend(0.65,0.15,0.85,0.35);
    leg1->SetFillColor(0);
    leg1->AddEntry(aCfLamK0, "#Lambda-K^{0}", "p");
    leg1->AddEntry(aCfALamK0, "#bar{#Lambda}-K^{0}", "p");
    leg1->Draw();

}

//________________________________________________________________________________________________________________
void buildAllcLamK02::BuildAvgSepCollections()
{
  TString NumName_AvgSepPosPos_LamK0 = "NumPosPosAvgSepCF_LamK0";
  TString DenName_AvgSepPosPos_LamK0 = "DenPosPosAvgSepCF_LamK0";
  TString NumName_AvgSepPosNeg_LamK0 = "NumPosNegAvgSepCF_LamK0";
  TString DenName_AvgSepPosNeg_LamK0 = "DenPosNegAvgSepCF_LamK0";
  TString NumName_AvgSepNegPos_LamK0 = "NumNegPosAvgSepCF_LamK0";
  TString DenName_AvgSepNegPos_LamK0 = "DenNegPosAvgSepCF_LamK0";
  TString NumName_AvgSepNegNeg_LamK0 = "NumNegNegAvgSepCF_LamK0";
  TString DenName_AvgSepNegNeg_LamK0 = "DenNegNegAvgSepCF_LamK0";

  TString NumName_AvgSepPosPos_ALamK0 = "NumPosPosAvgSepCF_ALamK0";
  TString DenName_AvgSepPosPos_ALamK0 = "DenPosPosAvgSepCF_ALamK0";
  TString NumName_AvgSepPosNeg_ALamK0 = "NumPosNegAvgSepCF_ALamK0";
  TString DenName_AvgSepPosNeg_ALamK0 = "DenPosNegAvgSepCF_ALamK0";
  TString NumName_AvgSepNegPos_ALamK0 = "NumNegPosAvgSepCF_ALamK0";
  TString DenName_AvgSepNegPos_ALamK0 = "DenNegPosAvgSepCF_ALamK0";
  TString NumName_AvgSepNegNeg_ALamK0 = "NumNegNegAvgSepCF_ALamK0";
  TString DenName_AvgSepNegNeg_ALamK0 = "DenNegNegAvgSepCF_ALamK0";

  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamK0.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamK0.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamK0.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  NumName_AvgSepPosPos_LamK0 += CentralityTag;
  DenName_AvgSepPosPos_LamK0 += CentralityTag;
  NumName_AvgSepPosNeg_LamK0 += CentralityTag;
  DenName_AvgSepPosNeg_LamK0 += CentralityTag;
  NumName_AvgSepNegPos_LamK0 += CentralityTag;
  DenName_AvgSepNegPos_LamK0 += CentralityTag;
  NumName_AvgSepNegNeg_LamK0 += CentralityTag;
  DenName_AvgSepNegNeg_LamK0 += CentralityTag;

  NumName_AvgSepPosPos_ALamK0 += CentralityTag;
  DenName_AvgSepPosPos_ALamK0 += CentralityTag;
  NumName_AvgSepPosNeg_ALamK0 += CentralityTag;
  DenName_AvgSepPosNeg_ALamK0 += CentralityTag;
  NumName_AvgSepNegPos_ALamK0 += CentralityTag;
  DenName_AvgSepNegPos_ALamK0 += CentralityTag;
  NumName_AvgSepNegNeg_ALamK0 += CentralityTag;
  DenName_AvgSepNegNeg_ALamK0 += CentralityTag;
  //----------------------------------------------------------------------

  //--LamK0
  fAvgSepNumCollection_PosPos_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_AvgSepPosPos_LamK0);
  fAvgSepDenCollection_PosPos_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_AvgSepPosPos_LamK0);
  fAvgSepCfCollection_PosPos_LamK0 = BuildCollectionOfCfs(fAvgSepNumCollection_PosPos_LamK0,fAvgSepDenCollection_PosPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_PosPos_LamK0_Tot = CombineCFs("fAvgSepCf_PosPos_LamK0_Tot","Lam-K0 (Tot)",fAvgSepCfCollection_PosPos_LamK0,fAvgSepNumCollection_PosPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_PosNeg_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_AvgSepPosNeg_LamK0);
  fAvgSepDenCollection_PosNeg_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_AvgSepPosNeg_LamK0);
  fAvgSepCfCollection_PosNeg_LamK0 = BuildCollectionOfCfs(fAvgSepNumCollection_PosNeg_LamK0,fAvgSepDenCollection_PosNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_PosNeg_LamK0_Tot = CombineCFs("fAvgSepCf_PosNeg_LamK0_Tot","Lam-K0 (Tot)",fAvgSepCfCollection_PosNeg_LamK0,fAvgSepNumCollection_PosNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_NegPos_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_AvgSepNegPos_LamK0);
  fAvgSepDenCollection_NegPos_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_AvgSepNegPos_LamK0);
  fAvgSepCfCollection_NegPos_LamK0 = BuildCollectionOfCfs(fAvgSepNumCollection_NegPos_LamK0,fAvgSepDenCollection_NegPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_NegPos_LamK0_Tot = CombineCFs("fAvgSepCf_NegPos_LamK0_Tot","Lam-K0 (Tot)",fAvgSepCfCollection_NegPos_LamK0,fAvgSepNumCollection_NegPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_NegNeg_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_AvgSepNegNeg_LamK0);
  fAvgSepDenCollection_NegNeg_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_AvgSepNegNeg_LamK0);
  fAvgSepCfCollection_NegNeg_LamK0 = BuildCollectionOfCfs(fAvgSepNumCollection_NegNeg_LamK0,fAvgSepDenCollection_NegNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_NegNeg_LamK0_Tot = CombineCFs("fAvgSepCf_NegNeg_LamK0_Tot","Lam-K0 (Tot)",fAvgSepCfCollection_NegNeg_LamK0,fAvgSepNumCollection_NegNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);


  //--ALam-K0
  fAvgSepNumCollection_PosPos_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_AvgSepPosPos_ALamK0);
  fAvgSepDenCollection_PosPos_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_AvgSepPosPos_ALamK0);
  fAvgSepCfCollection_PosPos_ALamK0 = BuildCollectionOfCfs(fAvgSepNumCollection_PosPos_ALamK0,fAvgSepDenCollection_PosPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_PosPos_ALamK0_Tot = CombineCFs("fAvgSepCf_PosPos_ALamK0_Tot","ALam-K0 (Tot)",fAvgSepCfCollection_PosPos_ALamK0,fAvgSepNumCollection_PosPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_PosNeg_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_AvgSepPosNeg_ALamK0);
  fAvgSepDenCollection_PosNeg_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_AvgSepPosNeg_ALamK0);
  fAvgSepCfCollection_PosNeg_ALamK0 = BuildCollectionOfCfs(fAvgSepNumCollection_PosNeg_ALamK0,fAvgSepDenCollection_PosNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_PosNeg_ALamK0_Tot = CombineCFs("fAvgSepCf_PosNeg_ALamK0_Tot","ALam-K0 (Tot)",fAvgSepCfCollection_PosNeg_ALamK0,fAvgSepNumCollection_PosNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_NegPos_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_AvgSepNegPos_ALamK0);
  fAvgSepDenCollection_NegPos_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_AvgSepNegPos_ALamK0);
  fAvgSepCfCollection_NegPos_ALamK0 = BuildCollectionOfCfs(fAvgSepNumCollection_NegPos_ALamK0,fAvgSepDenCollection_NegPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_NegPos_ALamK0_Tot = CombineCFs("fAvgSepCf_NegPos_ALamK0_Tot","ALam-K0 (Tot)",fAvgSepCfCollection_NegPos_ALamK0,fAvgSepNumCollection_NegPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_NegNeg_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_AvgSepNegNeg_ALamK0);
  fAvgSepDenCollection_NegNeg_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_AvgSepNegNeg_ALamK0);
  fAvgSepCfCollection_NegNeg_ALamK0 = BuildCollectionOfCfs(fAvgSepNumCollection_NegNeg_ALamK0,fAvgSepDenCollection_NegNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_NegNeg_ALamK0_Tot = CombineCFs("fAvgSepCf_NegNeg_ALamK0_Tot","ALam-K0 (Tot)",fAvgSepCfCollection_NegNeg_ALamK0,fAvgSepNumCollection_NegNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

}

//________________________________________________________________________________________________________________
void buildAllcLamK02::DrawFinalAvgSepCFs(TCanvas *aCanvasLamK0, TCanvas *aCanvasALamK0)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aAvgSepCf_PosPos_LamK0 = (TH1F*)fAvgSepCf_PosPos_LamK0_Tot->Clone();
  TH1F *aAvgSepCf_PosNeg_LamK0 = (TH1F*)fAvgSepCf_PosNeg_LamK0_Tot->Clone();
  TH1F *aAvgSepCf_NegPos_LamK0 = (TH1F*)fAvgSepCf_NegPos_LamK0_Tot->Clone();
  TH1F *aAvgSepCf_NegNeg_LamK0 = (TH1F*)fAvgSepCf_NegNeg_LamK0_Tot->Clone();
  //
  TH1F *aAvgSepCf_PosPos_ALamK0 = (TH1F*)fAvgSepCf_PosPos_ALamK0_Tot->Clone();
  TH1F *aAvgSepCf_PosNeg_ALamK0 = (TH1F*)fAvgSepCf_PosNeg_ALamK0_Tot->Clone();
  TH1F *aAvgSepCf_NegPos_ALamK0 = (TH1F*)fAvgSepCf_NegPos_ALamK0_Tot->Clone();
  TH1F *aAvgSepCf_NegNeg_ALamK0 = (TH1F*)fAvgSepCf_NegNeg_ALamK0_Tot->Clone();

  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);


  aCanvasLamK0->cd();
  aCanvasLamK0->Divide(2,2);

  aCanvasLamK0->cd(1);
  aAvgSepCf_PosPos_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_PosPos_LamK0->SetTitle("p(#Lambda) - #pi^{+}(K^{0})");
  aAvgSepCf_PosPos_LamK0->Draw();
  line->Draw();

  aCanvasLamK0->cd(2);
  aAvgSepCf_PosNeg_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_PosNeg_LamK0->SetTitle("p(#Lambda) - #pi^{-}(K^{0})");
  aAvgSepCf_PosNeg_LamK0->Draw();
  line->Draw();

  aCanvasLamK0->cd(3);
  aAvgSepCf_NegPos_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_NegPos_LamK0->SetTitle("#pi^{-}(#Lambda) - #pi^{+}(K^{0})");
  aAvgSepCf_NegPos_LamK0->Draw();
  line->Draw();

  aCanvasLamK0->cd(4);
  aAvgSepCf_NegNeg_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_NegNeg_LamK0->SetTitle("#pi^{-}(#Lambda) - #pi^{-}(K^{0})");
  aAvgSepCf_NegNeg_LamK0->Draw();
  line->Draw();




  aCanvasALamK0->cd();
  aCanvasALamK0->Divide(2,2);

  aCanvasALamK0->cd(1);
  aAvgSepCf_PosPos_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_PosPos_ALamK0->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  aAvgSepCf_PosPos_ALamK0->Draw();
  line->Draw();

  aCanvasALamK0->cd(2);
  aAvgSepCf_PosNeg_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_PosNeg_ALamK0->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  aAvgSepCf_PosNeg_ALamK0->Draw();
  line->Draw();

  aCanvasALamK0->cd(3);
  aAvgSepCf_NegPos_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_NegPos_ALamK0->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  aAvgSepCf_NegPos_ALamK0->Draw();
  line->Draw();

  aCanvasALamK0->cd(4);
  aAvgSepCf_NegNeg_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_NegNeg_ALamK0->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  aAvgSepCf_NegNeg_ALamK0->Draw();
  line->Draw();

}













//________________________________________________________________________________________________________________
void buildAllcLamK02::SetPurityRegimes(TH1F* aLambdaPurity, TH1F* aK0Short1Purity)
{
  fLamBgFitLow[0] = 1.09;
  fLamBgFitLow[1] = 1.102;

  fLamBgFitHigh[0] = 1.130;
  fLamBgFitHigh[1] = aLambdaPurity->GetBinLowEdge(aLambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)

  fLamROI[0] = LambdaMass-0.0038;
  fLamROI[1] = LambdaMass+0.0038;


  fK0Short1BgFitLow[0] = aK0Short1Purity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
  fK0Short1BgFitLow[1] = 0.452;

  fK0Short1BgFitHigh[0] = 0.536;
  fK0Short1BgFitHigh[1] = aK0Short1Purity->GetBinLowEdge(aK0Short1Purity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)

  fK0Short1ROI[0] = KaonMass-0.013677;
  fK0Short1ROI[1] = KaonMass+0.020323;

}




//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK02::CalculatePurity(const char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2])
{
  char buffer[50];
  sprintf(buffer, "fitBgd_%s",aReturnFitName);

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
  cout << aReturnFitName << ": " << "bgd = " << bgd << endl;
  //-----
  double sigpbgd = aPurityHisto->Integral(aPurityHisto->FindBin(aROI[0]),aPurityHisto->FindBin(aROI[1]));
  cout << aReturnFitName << ": " << "sig+bgd = " << sigpbgd << endl;
  //-----
  double sig = sigpbgd-bgd;
  cout << aReturnFitName << ": " << "sig = " << sig << endl;
  //-----
  double pur = sig/sigpbgd;
  cout << aReturnFitName << ": " << "Pur = " << pur << endl << endl;

  TVectorD *vInfo = new TVectorD(4);
    (*vInfo)(0) = bgd;
    (*vInfo)(1) = sigpbgd;
    (*vInfo)(2) = sig;
    (*vInfo)(3) = pur;

  TVectorD *vROI = new TVectorD(2);
    (*vROI)(0) = aROI[0];
    (*vROI)(1) = aROI[1];

  TVectorD *vBgFitLow = new TVectorD(2);
    (*vBgFitLow)(0) = aBgFitLow[0];
    (*vBgFitLow)(1) = aBgFitLow[1];

  TVectorD *vBgFitHigh = new TVectorD(2);
    (*vBgFitHigh)(0) = aBgFitHigh[0];
    (*vBgFitHigh)(1) = aBgFitHigh[1];
  //--------------------------------------------------------------------------------------------
  TObjArray* temp = new TObjArray();
  temp->Add(fitBgd2);
  temp->Add(vInfo);
  temp->Add(vROI);
  temp->Add(vBgFitLow);
  temp->Add(vBgFitHigh);
  return temp;

}

//________________________________________________________________________________________________________________
TH1F* buildAllcLamK02::CombineCollectionOfHistograms(TString aReturnHistoName, vector<TH1F*> &aCollectionOfHistograms)
{
  TH1F* ReturnHisto = (TH1F*)aCollectionOfHistograms[0]->Clone(aReturnHistoName);
  ReturnHisto->Sumw2();

  for(unsigned int i=1; i<aCollectionOfHistograms.size(); i++)
  {
    ReturnHisto->Add(aCollectionOfHistograms[i]);
  }

  return ReturnHisto;

}


//________________________________________________________________________________________________________________
void buildAllcLamK02::BuildPurityCollections()
{
  TString Name_LambdaPurity = "LambdaPurity";
  TString Name_K0Short1Purity = "K0ShortPurity1";
  TString Name_AntiLambdaPurity = "AntiLambdaPurity";
  TString Name_K0Short2Purity = "K0ShortPurity1";
  //----------------------------------------------------------
  fLambdaPurityHistogramCollection = LoadCollectionOfHistograms(fDirNameLamK0,Name_LambdaPurity);
  fK0Short1PurityHistogramCollection = LoadCollectionOfHistograms(fDirNameLamK0,Name_K0Short1Purity);

  fAntiLambdaPurityHistogramCollection = LoadCollectionOfHistograms(fDirNameALamK0,Name_AntiLambdaPurity);
  fK0Short2PurityHistogramCollection = LoadCollectionOfHistograms(fDirNameALamK0,Name_K0Short2Purity);
  //----------------------------------------------------------
  SetPurityRegimes(fLambdaPurityHistogramCollection[0],fK0Short1PurityHistogramCollection[0]);
  //----------------------------------------------------------
  for(unsigned int i=0; i<fLambdaPurityHistogramCollection.size(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("LambdaPurity",fLambdaPurityHistogramCollection[i],fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fLambdaPurityListCollection.push_back(TempList);
  }

  for(unsigned int i=0; i<fK0Short1PurityHistogramCollection.size(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("K0Short1Purity",fK0Short1PurityHistogramCollection[i],fK0Short1BgFitLow,fK0Short1BgFitHigh,fK0Short1ROI);
    fK0Short1PurityListCollection.push_back(TempList);
  }

  for(unsigned int i=0; i<fAntiLambdaPurityHistogramCollection.size(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("AntiLambdaPurity",fAntiLambdaPurityHistogramCollection[i],fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fAntiLambdaPurityListCollection.push_back(TempList);
  }

  for(unsigned int i=0; i<fK0Short2PurityHistogramCollection.size(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("K0Short2Purity",fK0Short2PurityHistogramCollection[i],fK0Short1BgFitLow,fK0Short1BgFitHigh,fK0Short1ROI);
    fK0Short2PurityListCollection.push_back(TempList);
  }
  //----------------------------------------------------------
  fLambdaPurityTot = CombineCollectionOfHistograms("fLambdaPurityTot",fLambdaPurityHistogramCollection);
  fK0Short1PurityTot = CombineCollectionOfHistograms("fK0Short1PurityTot",fK0Short1PurityHistogramCollection);
  fAntiLambdaPurityTot = CombineCollectionOfHistograms("fAntiLambdaPurityTot",fAntiLambdaPurityHistogramCollection);
  fK0Short2PurityTot = CombineCollectionOfHistograms("fK0Short2PurityTot",fK0Short2PurityHistogramCollection);
  //----------------------------------------------------------
  fLambdaPurityListTot = CalculatePurity("LambdaPurity",fLambdaPurityTot,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  fK0Short1PurityListTot = CalculatePurity("K0Short1Purity",fK0Short1PurityTot,fK0Short1BgFitLow,fK0Short1BgFitHigh,fK0Short1ROI);
  fAntiLambdaPurityListTot = CalculatePurity("AntiLambdaPurity",fAntiLambdaPurityTot,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  fK0Short2PurityListTot = CalculatePurity("K0Short2Purity",fK0Short2PurityTot,fK0Short1BgFitLow,fK0Short1BgFitHigh,fK0Short1ROI);

}



//________________________________________________________________________________________________________________
void buildAllcLamK02::DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg)
{
  TF1* fitBgd = (TF1*)aFitList->At(0);
    fitBgd->SetLineColor(4);
  //-----
  TVectorD* vInfo = (TVectorD*)aFitList->At(1);
  //-----
  TVectorD* vROI = (TVectorD*)aFitList->At(2);
  //-----
  TVectorD* vBgFitLow = (TVectorD*)aFitList->At(3);
  //-----
  TVectorD* vBgFitHigh = (TVectorD*)aFitList->At(4);
  //--------------------------------------------------------------------------------------------

  TLine *lROImin, *lROImax, *lBgFitLowMin, *lBgFitLowMax, *lBgFitHighMin, *lBgFitHighMax;

  if(!ZoomBg)
  {
    double HistoMaxValue = aPurityHisto->GetMaximum();
    lROImin = new TLine((*vROI)(0),0,(*vROI)(0),HistoMaxValue);
    lROImax = new TLine((*vROI)(1),0,(*vROI)(1),HistoMaxValue);
    //-----
    lBgFitLowMin = new TLine((*vBgFitLow)(0),0,(*vBgFitLow)(0),HistoMaxValue);
    lBgFitLowMax = new TLine((*vBgFitLow)(1),0,(*vBgFitLow)(1),HistoMaxValue);
    //-----
    lBgFitHighMin = new TLine((*vBgFitHigh)(0),0,(*vBgFitHigh)(0),HistoMaxValue);
    lBgFitHighMax = new TLine((*vBgFitHigh)(1),0,(*vBgFitHigh)(1),HistoMaxValue);
  }

  if(ZoomBg)
  {
    aPurityHisto->GetXaxis()->SetRange(aPurityHisto->FindBin((*vBgFitLow)(0)),aPurityHisto->FindBin((*vBgFitLow)(1)));
      double maxLow = aPurityHisto->GetMaximum();
      double minLow = aPurityHisto->GetMinimum();
    aPurityHisto->GetXaxis()->SetRange(aPurityHisto->FindBin((*vBgFitHigh)(0)),aPurityHisto->FindBin((*vBgFitHigh)(1))-1);
      double maxHigh = aPurityHisto->GetMaximum();
      double minHigh = aPurityHisto->GetMinimum();
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

    aPurityHisto->GetXaxis()->SetRange(1,aPurityHisto->GetNbinsX());
    aPurityHisto->GetYaxis()->SetRangeUser(minBg,maxBg);
    //--------------------------------------------------------------------------------------------
    lROImin = new TLine((*vROI)(0),minBg,(*vROI)(0),maxBg);
    lROImax = new TLine((*vROI)(1),minBg,(*vROI)(1),maxBg);
    //-----
    lBgFitLowMin = new TLine((*vBgFitLow)(0),minBg,(*vBgFitLow)(0),maxBg);
    lBgFitLowMax = new TLine((*vBgFitLow)(1),minBg,(*vBgFitLow)(1),maxBg);
    //-----
    lBgFitHighMin = new TLine((*vBgFitHigh)(0),minBg,(*vBgFitHigh)(0),maxBg);
    lBgFitHighMax = new TLine((*vBgFitHigh)(1),minBg,(*vBgFitHigh)(1),maxBg);
  }

  //--------------------------------------------------------------------------------------------
  aPurityHisto->SetLineColor(1);
  aPurityHisto->SetLineWidth(3);

  lROImin->SetLineColor(3);
  lROImax->SetLineColor(3);

  lBgFitLowMin->SetLineColor(2);
  lBgFitLowMax->SetLineColor(2);

  lBgFitHighMin->SetLineColor(2);
  lBgFitHighMax->SetLineColor(2);


  //--------------------------------------------------------------------------------------------
  aPurityHisto->DrawCopy("Ehist");
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
    double purity = (*vInfo)(3);
    const char* title = aPurityHisto->GetName();
    //char title[20] = aPurityHisto->GetName();
    sprintf(buffer, "%s = %.2f%%",title, 100.*purity);
    myText->AddText(buffer);
    myText->Draw();
  }
}


//________________________________________________________________________________________________________________
void buildAllcLamK02::DrawFinalPurity(TCanvas *aCanvas)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aLambdaPurity = (TH1F*)fLambdaPurityTot->Clone();
  TH1F *aK0Short1Purity = (TH1F*)fK0Short1PurityTot->Clone();
  TH1F *aAntiLambdaPurity = (TH1F*)fAntiLambdaPurityTot->Clone();
  TH1F *aK0Short2Purity = (TH1F*)fK0Short2PurityTot->Clone();

  TObjArray *aLambdaPurityList = (TObjArray*)fLambdaPurityListTot->Clone();
  TObjArray *aK0Short1PurityList = (TObjArray*)fK0Short1PurityListTot->Clone();
  TObjArray *aAntiLambdaPurityList = (TObjArray*)fAntiLambdaPurityListTot->Clone();
  TObjArray *aK0Short2PurityList = (TObjArray*)fK0Short2PurityListTot->Clone();


  aCanvas->cd();
  aCanvas->Divide(2,4);
  //-----
  aCanvas->cd(1);
  DrawPurity(aLambdaPurity,aLambdaPurityList,false);
  aCanvas->cd(2);
  DrawPurity(aLambdaPurity,aLambdaPurityList,true);
  //-----
  aCanvas->cd(3);
  DrawPurity(aK0Short1Purity,aK0Short1PurityList,false);
  aCanvas->cd(4);
  DrawPurity(aK0Short1Purity,aK0Short1PurityList,true);
  //-----
  aCanvas->cd(5);
  DrawPurity(aAntiLambdaPurity,aAntiLambdaPurityList,false);
  aCanvas->cd(6);
  DrawPurity(aAntiLambdaPurity,aAntiLambdaPurityList,true);
  //-----
  aCanvas->cd(7);
  DrawPurity(aK0Short2Purity,aK0Short2PurityList,false);
  aCanvas->cd(8);
  DrawPurity(aK0Short2Purity,aK0Short2PurityList,true);

}















//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
