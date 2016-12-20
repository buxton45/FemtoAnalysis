///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamcKch2                                                        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "buildAllcLamcKch2.h"

#ifdef __ROOT__
ClassImp(buildAllcLamcKch2)
#endif

//________________________________________________________________________________________________________________
buildAllcLamcKch2::buildAllcLamcKch2(vector<TString> &aVectorOfFileNames, TString aDirNameLamKchP, TString aDirNameLamKchM, TString aDirNameALamKchP, TString aDirNameALamKchM):
  //General stuff----------------
  fDirNameLamKchP(aDirNameLamKchP),
  fDirNameLamKchM(aDirNameLamKchM),
  fDirNameALamKchP(aDirNameALamKchP),
  fDirNameALamKchM(aDirNameALamKchM),

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


}

//________________________________________________________________________________________________________________
TH1F* buildAllcLamcKch2::buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int aMinNormBin, int aMaxNormBin)
{
//  cout << "For hist "<< Num->GetName()<<", num pair = " << Num->Integral(1,Num->FindBin(.05)) << endl;
  double NumScale = Num->Integral(aMinNormBin,aMaxNormBin);
  double DenScale = Denom->Integral(aMinNormBin,aMaxNormBin);

  TH1F* CF = Num->Clone(name);
  CF->Sumw2();
  CF->Divide(Denom);
  CF->Scale(DenScale/NumScale);
  CF->SetTitle(title);

  return CF;
}

//________________________________________________________________________________________________________________
TH1F* buildAllcLamcKch2::CombineCFs(TString aReturnName, TString aReturnTitle, vector<TH1F*> &aCfCollection, vector<TH1F*> &aNumCollection, int aMinNormBin, int aMaxNormBin)
{
  double scale = 0.;
  int counter = 0;
  double temp = 0.;

  int SizeOfCfCollection = aCfCollection.size();
  int SizeOfNumCollection = aNumCollection.size();

  if(SizeOfCfCollection != SizeOfNumCollection) {cout << "ERROR: In CombineCFs, the CfCollection and NumCollection ARE NOT EQUAL IN SIZE!!!!" << endl;}

  TH1F* ReturnCf = aCfCollection[0]->Clone(aReturnName);
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
void buildAllcLamcKch2::SetVectorOfFileNames(vector<TString> &aVectorOfNames)
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
TObjArray* buildAllcLamcKch2::ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName)
{
  TFile f1(aFileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  TObjArray *ReturnArray = (TObjArray*)femtolist->FindObject(aDirectoryName);

  return ReturnArray;
}

//________________________________________________________________________________________________________________
void buildAllcLamcKch2::SetAnalysisDirectories()
{
  TString FileNameBp1, FileNameBp2, FileNameBm1, FileNameBm2, FileNameBm3;
  //The files should be ordered Bp1, Bp2, Bm1, Bm2, Bm3...but if they are not, the following for loop
  // will pick out the correct file names to associate with the directories

  for(int i=0; i<fVectorOfFileNames.size(); i++)
  {
    if(fVectorOfFileNames[i].Contains("Bp1")){FileNameBp1 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bp2")){FileNameBp2 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm1")){FileNameBm1 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm2")){FileNameBm2 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm3")){FileNameBm3 = fVectorOfFileNames[i];}
    else{cout << "ERROR: In SetAnalysisDirectories....a file name was found to NOT contain Bp1,Bp2,Bm1,Bm2,or Bm3!!!!!" << endl;}
  }

  fDirLamKchPBp1 = ConnectAnalysisDirectory(FileNameBp1,fDirNameLamKchP);
  fDirLamKchMBp1 = ConnectAnalysisDirectory(FileNameBp1,fDirNameLamKchM);
  fDirALamKchPBp1 = ConnectAnalysisDirectory(FileNameBp1,fDirNameALamKchP);
  fDirALamKchMBp1 = ConnectAnalysisDirectory(FileNameBp1,fDirNameALamKchM);

  fDirLamKchPBp2 = ConnectAnalysisDirectory(FileNameBp2,fDirNameLamKchP);
  fDirLamKchMBp2 = ConnectAnalysisDirectory(FileNameBp2,fDirNameLamKchM);
  fDirALamKchPBp2 = ConnectAnalysisDirectory(FileNameBp2,fDirNameALamKchP);
  fDirALamKchMBp2 = ConnectAnalysisDirectory(FileNameBp2,fDirNameALamKchM);

  fDirLamKchPBm1 = ConnectAnalysisDirectory(FileNameBm1,fDirNameLamKchP);
  fDirLamKchMBm1 = ConnectAnalysisDirectory(FileNameBm1,fDirNameLamKchM);
  fDirALamKchPBm1 = ConnectAnalysisDirectory(FileNameBm1,fDirNameALamKchP);
  fDirALamKchMBm1 = ConnectAnalysisDirectory(FileNameBm1,fDirNameALamKchM);

  fDirLamKchPBm2 = ConnectAnalysisDirectory(FileNameBm2,fDirNameLamKchP);
  fDirLamKchMBm2 = ConnectAnalysisDirectory(FileNameBm2,fDirNameLamKchM);
  fDirALamKchPBm2 = ConnectAnalysisDirectory(FileNameBm2,fDirNameALamKchP);
  fDirALamKchMBm2 = ConnectAnalysisDirectory(FileNameBm2,fDirNameALamKchM);

  fDirLamKchPBm3 = ConnectAnalysisDirectory(FileNameBm3,fDirNameLamKchP);
  fDirLamKchMBm3 = ConnectAnalysisDirectory(FileNameBm3,fDirNameLamKchM);
  fDirALamKchPBm3 = ConnectAnalysisDirectory(FileNameBm3,fDirNameALamKchP);
  fDirALamKchMBm3 = ConnectAnalysisDirectory(FileNameBm3,fDirNameALamKchM);
}

//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch2::GetAnalysisDirectory(TString aFile, TString aDirectoryName)
{
  if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBp1;}
  if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBp1;}
  if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBp1;}
  if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBp1;}

  if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBp2;}
  if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBp2;}
  if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBp2;}
  if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBp2;}

  if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBm1;}
  if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBm1;}
  if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBm1;}
  if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBm1;}

  if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBm2;}
  if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBm2;}
  if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBm2;}
  if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBm2;}

  if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBm3;}
  if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBm3;}
  if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBm3;}
  if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBm3;}

}


//________________________________________________________________________________________________________________
TH1F* buildAllcLamcKch2::GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName)
{
  TH1F *ReturnHisto = (TH1F*)aAnalysisDirectory->FindObject(aHistoName);
  TH1F *ReturnHistoClone = (TH1F*)ReturnHisto->Clone(aCloneHistoName);
  ReturnHistoClone->SetDirectory(0);
  ReturnHistoClone->Sumw2();

  return ReturnHistoClone;
}


//________________________________________________________________________________________________________________
vector<TH1F*> buildAllcLamcKch2::LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName)
{
  vector<TH1F*> ReturnCollection;

  cout << "A" << endl;
  if(aDirectoryName.EqualTo(fDirNameLamKchP))
  {
    cout << "B" << endl;
    ReturnCollection.push_back(GetHistoClone(fDirLamKchPBp1,aHistoName,aHistoName+"_Bp1"));
    cout << "C" << endl;
    ReturnCollection.push_back(GetHistoClone(fDirLamKchPBp2,aHistoName,aHistoName+"_Bp2"));
    cout << "D" << endl;
    ReturnCollection.push_back(GetHistoClone(fDirLamKchPBm1,aHistoName,aHistoName+"_Bm1"));
    cout << "E" << endl;
    ReturnCollection.push_back(GetHistoClone(fDirLamKchPBm2,aHistoName,aHistoName+"_Bm2"));
    cout << "F" << endl;
    ReturnCollection.push_back(GetHistoClone(fDirLamKchPBm3,aHistoName,aHistoName+"_Bm3"));
    cout << "G" << endl;
  }

  else if(aDirectoryName.EqualTo(fDirNameLamKchM))
  {
    ReturnCollection.push_back(GetHistoClone(fDirLamKchMBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection.push_back(GetHistoClone(fDirLamKchMBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection.push_back(GetHistoClone(fDirLamKchMBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection.push_back(GetHistoClone(fDirLamKchMBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection.push_back(GetHistoClone(fDirLamKchMBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamKchP))
  {
    ReturnCollection.push_back(GetHistoClone(fDirALamKchPBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection.push_back(GetHistoClone(fDirALamKchPBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection.push_back(GetHistoClone(fDirALamKchPBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection.push_back(GetHistoClone(fDirALamKchPBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection.push_back(GetHistoClone(fDirALamKchPBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamKchM))
  {
    ReturnCollection.push_back(GetHistoClone(fDirALamKchMBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection.push_back(GetHistoClone(fDirALamKchMBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection.push_back(GetHistoClone(fDirALamKchMBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection.push_back(GetHistoClone(fDirALamKchMBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection.push_back(GetHistoClone(fDirALamKchMBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else{cout << "ERROR IN LoadCollectionOfHistograms!!!!!!!!!!!!!!!!" << endl;}
  cout << "H" << endl;


  return ReturnCollection;

}

//________________________________________________________________________________________________________________
vector<TH1F*> buildAllcLamcKch2::BuildCollectionOfCfs(vector<TH1F*> &aNumCollection, vector<TH1F*> &aDenCollection, int aMinNormBin, int aMaxNormBin)
{
  vector<TH1F*> ReturnCollection;

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
void buildAllcLamcKch2::BuildCFCollections()
{
  TString NumName_LamKchP = "NumLamKchPKStarCF";
  TString DenName_LamKchP = "DenLamKchPKStarCF";

  TString NumName_LamKchM = "NumLamKchMKStarCF";
  TString DenName_LamKchM = "DenLamKchMKStarCF";

  TString NumName_ALamKchP = "NumALamKchPKStarCF";
  TString DenName_ALamKchP = "DenALamKchPKStarCF";

  TString NumName_ALamKchM = "NumALamKchMKStarCF";
  TString DenName_ALamKchM = "DenALamKchMKStarCF";

  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamKchP.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamKchP.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamKchP.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  NumName_LamKchP+=CentralityTag;
  DenName_LamKchP+=CentralityTag;
  NumName_LamKchM+=CentralityTag;
  DenName_LamKchM+=CentralityTag;
  NumName_ALamKchP+=CentralityTag;
  DenName_ALamKchP+=CentralityTag;
  NumName_ALamKchM+=CentralityTag;
  DenName_ALamKchM+=CentralityTag;
  //----------------------------------------------------------------------
  cout << "0" << endl;
  fNumCollection_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,NumName_LamKchP);
  cout << "1" << endl;
  fDenCollection_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,DenName_LamKchP);
  cout << "2" << endl;
  fCfCollection_LamKchP = BuildCollectionOfCfs(fNumCollection_LamKchP,fDenCollection_LamKchP,fMinNormBinCF,fMaxNormBinCF);
  cout << "3" << endl;
  fCf_LamKchP_Tot = CombineCFs("fCf_LamKchP_Tot","Lam-K+ (Tot)",fCfCollection_LamKchP,fNumCollection_LamKchP,fMinNormBinCF,fMaxNormBinCF);
  cout << "4" << endl;

  fNumCollection_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,NumName_LamKchM);
  fDenCollection_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,DenName_LamKchM);
  fCfCollection_LamKchM = BuildCollectionOfCfs(fNumCollection_LamKchM,fDenCollection_LamKchM,fMinNormBinCF,fMaxNormBinCF);
  fCf_LamKchM_Tot = CombineCFs("fCf_LamKchM_Tot","Lam-K- (Tot)",fCfCollection_LamKchM,fNumCollection_LamKchM,fMinNormBinCF,fMaxNormBinCF);

  fNumCollection_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,NumName_ALamKchP);
  fDenCollection_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,DenName_ALamKchP);
  fCfCollection_ALamKchP = BuildCollectionOfCfs(fNumCollection_ALamKchP,fDenCollection_ALamKchP,fMinNormBinCF,fMaxNormBinCF);
  fCf_ALamKchP_Tot = CombineCFs("fCf_ALamKchP_Tot","ALam-K+ (Tot)",fCfCollection_ALamKchP,fNumCollection_ALamKchP,fMinNormBinCF,fMaxNormBinCF);

  fNumCollection_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,NumName_ALamKchM);
  fDenCollection_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,DenName_ALamKchM);
  fCfCollection_ALamKchM = BuildCollectionOfCfs(fNumCollection_ALamKchM,fDenCollection_ALamKchM,fMinNormBinCF,fMaxNormBinCF);
  fCf_ALamKchM_Tot = CombineCFs("fCf_ALamKchM_Tot","ALam-K- (Tot)",fCfCollection_ALamKchM,fNumCollection_ALamKchM,fMinNormBinCF,fMaxNormBinCF);

  //--------------BpTot-----------------------------------------------------
  vector<TH1F*> fNumCollection_LamKchP_BpTot;
    fNumCollection_LamKchP_BpTot.push_back(fNumCollection_LamKchP[0]);
    fNumCollection_LamKchP_BpTot.push_back(fNumCollection_LamKchP[1]);
  vector<TH1F*> fCfCollection_LamKchP_BpTot;
    fCfCollection_LamKchP_BpTot.push_back(fCfCollection_LamKchP[0]);
    fCfCollection_LamKchP_BpTot.push_back(fCfCollection_LamKchP[1]);
  fCf_LamKchP_BpTot = CombineCFs("fCf_LamKchP_BpTot","Lam-K+ (B+ Tot)",fCfCollection_LamKchP_BpTot,fNumCollection_LamKchP_BpTot,fMinNormBinCF,fMaxNormBinCF);

  vector<TH1F*> fNumCollection_LamKchM_BpTot;
    fNumCollection_LamKchM_BpTot.push_back(fNumCollection_LamKchM[0]);
    fNumCollection_LamKchM_BpTot.push_back(fNumCollection_LamKchM[1]);
  vector<TH1F*> fCfCollection_LamKchM_BpTot;
    fCfCollection_LamKchM_BpTot.push_back(fCfCollection_LamKchM[0]);
    fCfCollection_LamKchM_BpTot.push_back(fCfCollection_LamKchM[1]);
  fCf_LamKchM_BpTot = CombineCFs("fCf_LamKchM_BpTot","Lam-K- (B+ Tot)",fCfCollection_LamKchM_BpTot,fNumCollection_LamKchM_BpTot,fMinNormBinCF,fMaxNormBinCF);

  vector<TH1F*> fNumCollection_ALamKchP_BpTot;
    fNumCollection_ALamKchP_BpTot.push_back(fNumCollection_ALamKchP[0]);
    fNumCollection_ALamKchP_BpTot.push_back(fNumCollection_ALamKchP[1]);
  vector<TH1F*> fCfCollection_ALamKchP_BpTot;
    fCfCollection_ALamKchP_BpTot.push_back(fCfCollection_ALamKchP[0]);
    fCfCollection_ALamKchP_BpTot.push_back(fCfCollection_ALamKchP[1]);
  fCf_ALamKchP_BpTot = CombineCFs("fCf_ALamKchP_BpTot","ALam-K+ (B+ Tot)",fCfCollection_ALamKchP_BpTot,fNumCollection_ALamKchP_BpTot,fMinNormBinCF,fMaxNormBinCF);

  vector<TH1F*> fNumCollection_ALamKchM_BpTot;
    fNumCollection_ALamKchM_BpTot.push_back(fNumCollection_ALamKchM[0]);
    fNumCollection_ALamKchM_BpTot.push_back(fNumCollection_ALamKchM[1]);
  vector<TH1F*> fCfCollection_ALamKchM_BpTot;
    fCfCollection_ALamKchM_BpTot.push_back(fCfCollection_ALamKchM[0]);
    fCfCollection_ALamKchM_BpTot.push_back(fCfCollection_ALamKchM[1]);
  fCf_ALamKchM_BpTot = CombineCFs("fCf_ALamKchM_BpTot","ALam-K- (B+ Tot)",fCfCollection_ALamKchM_BpTot,fNumCollection_ALamKchM_BpTot,fMinNormBinCF,fMaxNormBinCF);


  //--------------BmTot-----------------------------------------------------
  vector<TH1F*> fNumCollection_LamKchP_BmTot;
    fNumCollection_LamKchP_BmTot.push_back(fNumCollection_LamKchP[2]);
    fNumCollection_LamKchP_BmTot.push_back(fNumCollection_LamKchP[3]);
    fNumCollection_LamKchP_BmTot.push_back(fNumCollection_LamKchP[4]);
  vector<TH1F*> fCfCollection_LamKchP_BmTot;
    fCfCollection_LamKchP_BmTot.push_back(fCfCollection_LamKchP[2]);
    fCfCollection_LamKchP_BmTot.push_back(fCfCollection_LamKchP[3]);
    fCfCollection_LamKchP_BmTot.push_back(fCfCollection_LamKchP[4]);
  fCf_LamKchP_BmTot = CombineCFs("fCf_LamKchP_BmTot","Lam-K+ (B- Tot)",fCfCollection_LamKchP_BmTot,fNumCollection_LamKchP_BmTot,fMinNormBinCF,fMaxNormBinCF);

  vector<TH1F*> fNumCollection_LamKchM_BmTot;
    fNumCollection_LamKchM_BmTot.push_back(fNumCollection_LamKchM[2]);
    fNumCollection_LamKchM_BmTot.push_back(fNumCollection_LamKchM[3]);
    fNumCollection_LamKchM_BmTot.push_back(fNumCollection_LamKchM[4]);
  vector<TH1F*> fCfCollection_LamKchM_BmTot;
    fCfCollection_LamKchM_BmTot.push_back(fCfCollection_LamKchM[2]);
    fCfCollection_LamKchM_BmTot.push_back(fCfCollection_LamKchM[3]);
    fCfCollection_LamKchM_BmTot.push_back(fCfCollection_LamKchM[4]);
  fCf_LamKchM_BmTot = CombineCFs("fCf_LamKchM_BmTot","Lam-K- (B- Tot)",fCfCollection_LamKchM_BmTot,fNumCollection_LamKchM_BmTot,fMinNormBinCF,fMaxNormBinCF);

  vector<TH1F*> fNumCollection_ALamKchP_BmTot;
    fNumCollection_ALamKchP_BmTot.push_back(fNumCollection_ALamKchP[2]);
    fNumCollection_ALamKchP_BmTot.push_back(fNumCollection_ALamKchP[3]);
    fNumCollection_ALamKchP_BmTot.push_back(fNumCollection_ALamKchP[4]);
  vector<TH1F*> fCfCollection_ALamKchP_BmTot;
    fCfCollection_ALamKchP_BmTot.push_back(fCfCollection_ALamKchP[2]);
    fCfCollection_ALamKchP_BmTot.push_back(fCfCollection_ALamKchP[3]);
    fCfCollection_ALamKchP_BmTot.push_back(fCfCollection_ALamKchP[4]);
  fCf_ALamKchP_BmTot = CombineCFs("fCf_ALamKchP_BmTot","ALam-K+ (B- Tot)",fCfCollection_ALamKchP_BmTot,fNumCollection_ALamKchP_BmTot,fMinNormBinCF,fMaxNormBinCF);

  vector<TH1F*> fNumCollection_ALamKchM_BmTot;
    fNumCollection_ALamKchM_BmTot.push_back(fNumCollection_ALamKchM[2]);
    fNumCollection_ALamKchM_BmTot.push_back(fNumCollection_ALamKchM[3]);
    fNumCollection_ALamKchM_BmTot.push_back(fNumCollection_ALamKchM[4]);
  vector<TH1F*> fCfCollection_ALamKchM_BmTot;
    fCfCollection_ALamKchM_BmTot.push_back(fCfCollection_ALamKchM[2]);
    fCfCollection_ALamKchM_BmTot.push_back(fCfCollection_ALamKchM[3]);
    fCfCollection_ALamKchM_BmTot.push_back(fCfCollection_ALamKchM[4]);
  fCf_ALamKchM_BmTot = CombineCFs("fCf_ALamKchM_BmTot","ALam-K- (B- Tot)",fCfCollection_ALamKchM_BmTot,fNumCollection_ALamKchM_BmTot,fMinNormBinCF,fMaxNormBinCF);

}

//________________________________________________________________________________________________________________
void buildAllcLamcKch2::DrawFinalCFs(TCanvas *aCanvas)
{
  aCanvas->cd();
  gStyle->SetOptStat(0);

  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aCfLamKchP = fCf_LamKchP_Tot->Clone();
  TH1F *aCfLamKchM = fCf_LamKchM_Tot->Clone();
  TH1F *aCfALamKchP = fCf_ALamKchP_Tot->Clone();
  TH1F *aCfALamKchM = fCf_ALamKchM_Tot->Clone();
  //------------------------------------------------------
  TAxis *xax1 = aCfLamKchP->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
  TAxis *yax1 = aCfLamKchP->GetYaxis();
    yax1->SetRangeUser(0.8,1.2);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();

  TAxis *xax2 = aCfALamKchM->GetXaxis();
    xax2->SetTitle("k* (GeV/c)");
    xax2->SetTitleSize(0.05);
    xax2->SetTitleOffset(1.0);
    //xax2->CenterTitle();
  TAxis *yax2 = aCfALamKchM->GetYaxis();
    yax2->SetRangeUser(0.8,1.2);
    yax2->SetTitle("C(k*)");
    yax2->SetTitleSize(0.05);
    yax2->SetTitleOffset(1.0);
    yax2->CenterTitle();

  TAxis *xax3 = aCfLamKchM->GetXaxis();
    xax3->SetTitle("k* (GeV/c)");
    xax3->SetTitleSize(0.05);
    xax3->SetTitleOffset(1.0);
    //xax3->CenterTitle();
  TAxis *yax3 = aCfLamKchM->GetYaxis();
    yax3->SetRangeUser(0.8,1.2);
    yax3->SetTitle("C(k*)");
    yax3->SetTitleSize(0.05);
    yax3->SetTitleOffset(1.0);
    yax3->CenterTitle();

  TAxis *xax4 = aCfALamKchP->GetXaxis();
    xax4->SetTitle("k* (GeV/c)");
    xax4->SetTitleSize(0.05);
    xax4->SetTitleOffset(1.0);
    //xax4->CenterTitle();
  TAxis *yax4 = aCfALamKchP->GetYaxis();
    yax4->SetRangeUser(0.8,1.2);
    yax4->SetTitle("C(k*)");
    yax4->SetTitleSize(0.05);
    yax4->SetTitleOffset(1.0);
    yax4->CenterTitle();
  //------------------------------------------------------
  aCfLamKchP->SetMarkerStyle(20);
  aCfLamKchP->SetMarkerSize(1);
  aCfLamKchP->SetMarkerColor(1);
  aCfLamKchP->SetLineColor(1);
  aCfLamKchP->SetTitle("#Lambda - K+");

  aCfLamKchM->SetMarkerStyle(20);
  aCfLamKchM->SetMarkerSize(1);
  aCfLamKchM->SetMarkerColor(1);
  aCfLamKchM->SetLineColor(1);
  aCfLamKchM->SetTitle("#Lambda - K-");

  aCfALamKchP->SetMarkerStyle(20);
  aCfALamKchP->SetMarkerSize(1);
  aCfALamKchP->SetMarkerColor(1);
  aCfALamKchP->SetLineColor(1);
  aCfALamKchP->SetTitle("#bar{#Lambda} - K+");

  aCfALamKchM->SetMarkerStyle(20);
  aCfALamKchM->SetMarkerSize(1);
  aCfALamKchM->SetMarkerColor(1);
  aCfALamKchM->SetLineColor(1);
  aCfALamKchM->SetTitle("#bar{#Lambda} - K-");

  //------------------------------------------------------
  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);

  aCanvas->Divide(2,2);

  aCanvas->cd(1);
  aCfLamKchP->Draw();
  line->Draw();

  aCanvas->cd(2);
  aCfALamKchM->Draw();
  line->Draw();

  aCanvas->cd(3);
  aCfLamKchM->Draw();
  line->Draw();

  aCanvas->cd(4);
  aCfALamKchP->Draw();
  line->Draw();

}

//________________________________________________________________________________________________________________
void buildAllcLamcKch2::BuildAvgSepCollections()
{
  TString NumName_AvgSepTrackPos_LamKchP = "NumTrackPosAvgSepCF_LamKchP";
  TString DenName_AvgSepTrackPos_LamKchP = "DenTrackPosAvgSepCF_LamKchP";
  TString NumName_AvgSepTrackNeg_LamKchP = "NumTrackNegAvgSepCF_LamKchP";
  TString DenName_AvgSepTrackNeg_LamKchP = "DenTrackNegAvgSepCF_LamKchP";

  TString NumName_AvgSepTrackPos_LamKchM = "NumTrackPosAvgSepCF_LamKchM";
  TString DenName_AvgSepTrackPos_LamKchM = "DenTrackPosAvgSepCF_LamKchM";
  TString NumName_AvgSepTrackNeg_LamKchM = "NumTrackNegAvgSepCF_LamKchM";
  TString DenName_AvgSepTrackNeg_LamKchM = "DenTrackNegAvgSepCF_LamKchM";


  TString NumName_AvgSepTrackPos_ALamKchP = "NumTrackPosAvgSepCF_ALamKchP";
  TString DenName_AvgSepTrackPos_ALamKchP = "DenTrackPosAvgSepCF_ALamKchP";
  TString NumName_AvgSepTrackNeg_ALamKchP = "NumTrackNegAvgSepCF_ALamKchP";
  TString DenName_AvgSepTrackNeg_ALamKchP = "DenTrackNegAvgSepCF_ALamKchP";

  TString NumName_AvgSepTrackPos_ALamKchM = "NumTrackPosAvgSepCF_ALamKchM";
  TString DenName_AvgSepTrackPos_ALamKchM = "DenTrackPosAvgSepCF_ALamKchM";
  TString NumName_AvgSepTrackNeg_ALamKchM = "NumTrackNegAvgSepCF_ALamKchM";
  TString DenName_AvgSepTrackNeg_ALamKchM = "DenTrackNegAvgSepCF_ALamKchM";


  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamKchP.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamKchP.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamKchP.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  NumName_AvgSepTrackPos_LamKchP += CentralityTag;
  DenName_AvgSepTrackPos_LamKchP += CentralityTag;
  NumName_AvgSepTrackNeg_LamKchP += CentralityTag;
  DenName_AvgSepTrackNeg_LamKchP += CentralityTag;

  NumName_AvgSepTrackPos_LamKchM += CentralityTag;
  DenName_AvgSepTrackPos_LamKchM += CentralityTag;
  NumName_AvgSepTrackNeg_LamKchM += CentralityTag;
  DenName_AvgSepTrackNeg_LamKchM += CentralityTag;


  NumName_AvgSepTrackPos_ALamKchP += CentralityTag;
  DenName_AvgSepTrackPos_ALamKchP += CentralityTag;
  NumName_AvgSepTrackNeg_ALamKchP += CentralityTag;
  DenName_AvgSepTrackNeg_ALamKchP += CentralityTag;

  NumName_AvgSepTrackPos_ALamKchM += CentralityTag;
  DenName_AvgSepTrackPos_ALamKchM += CentralityTag;
  NumName_AvgSepTrackNeg_ALamKchM += CentralityTag;
  DenName_AvgSepTrackNeg_ALamKchM += CentralityTag;

  //----------------------------------------------------------------------

  //--LamKchP
  fAvgSepNumCollection_TrackPos_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,NumName_AvgSepTrackPos_LamKchP);
  fAvgSepDenCollection_TrackPos_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,DenName_AvgSepTrackPos_LamKchP);
  fAvgSepCfCollection_TrackPos_LamKchP = BuildCollectionOfCfs(fAvgSepNumCollection_TrackPos_LamKchP,fAvgSepDenCollection_TrackPos_LamKchP,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackPos_LamKchP_Tot = CombineCFs("fAvgSepCf_TrackPos_LamKchP_Tot", "Track+ (Lam-K+ (Tot))", fAvgSepCfCollection_TrackPos_LamKchP, fAvgSepNumCollection_TrackPos_LamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_TrackNeg_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,NumName_AvgSepTrackNeg_LamKchP);
  fAvgSepDenCollection_TrackNeg_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,DenName_AvgSepTrackNeg_LamKchP);
  fAvgSepCfCollection_TrackNeg_LamKchP = BuildCollectionOfCfs(fAvgSepNumCollection_TrackNeg_LamKchP,fAvgSepDenCollection_TrackNeg_LamKchP,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackNeg_LamKchP_Tot = CombineCFs("fAvgSepCf_TrackNeg_LamKchP_Tot", "Track- (Lam-K+ (Tot))", fAvgSepCfCollection_TrackNeg_LamKchP, fAvgSepNumCollection_TrackNeg_LamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);

  //--LamKchM
  fAvgSepNumCollection_TrackPos_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,NumName_AvgSepTrackPos_LamKchM);
  fAvgSepDenCollection_TrackPos_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,DenName_AvgSepTrackPos_LamKchM);
  fAvgSepCfCollection_TrackPos_LamKchM = BuildCollectionOfCfs(fAvgSepNumCollection_TrackPos_LamKchM,fAvgSepDenCollection_TrackPos_LamKchM,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackPos_LamKchM_Tot = CombineCFs("fAvgSepCf_TrackPos_LamKchM_Tot", "Track+ (Lam-K- (Tot))", fAvgSepCfCollection_TrackPos_LamKchM, fAvgSepNumCollection_TrackPos_LamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_TrackNeg_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,NumName_AvgSepTrackNeg_LamKchM);
  fAvgSepDenCollection_TrackNeg_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,DenName_AvgSepTrackNeg_LamKchM);
  fAvgSepCfCollection_TrackNeg_LamKchM = BuildCollectionOfCfs(fAvgSepNumCollection_TrackNeg_LamKchM,fAvgSepDenCollection_TrackNeg_LamKchM,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackNeg_LamKchM_Tot = CombineCFs("fAvgSepCf_TrackNeg_LamKchM_Tot", "Track- (Lam-K- (Tot))", fAvgSepCfCollection_TrackNeg_LamKchM, fAvgSepNumCollection_TrackNeg_LamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);


  //--ALam-KchP
  fAvgSepNumCollection_TrackPos_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,NumName_AvgSepTrackPos_ALamKchP);
  fAvgSepDenCollection_TrackPos_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,DenName_AvgSepTrackPos_ALamKchP);
  fAvgSepCfCollection_TrackPos_ALamKchP = BuildCollectionOfCfs(fAvgSepNumCollection_TrackPos_ALamKchP, fAvgSepDenCollection_TrackPos_ALamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackPos_ALamKchP_Tot = CombineCFs("fAvgSepCf_TrackPos_ALamKchP_Tot", "Track+ (ALam-K+ (Tot))", fAvgSepCfCollection_TrackPos_ALamKchP, fAvgSepNumCollection_TrackPos_ALamKchP, fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_TrackNeg_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,NumName_AvgSepTrackNeg_ALamKchP);
  fAvgSepDenCollection_TrackNeg_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,DenName_AvgSepTrackNeg_ALamKchP);
  fAvgSepCfCollection_TrackNeg_ALamKchP = BuildCollectionOfCfs(fAvgSepNumCollection_TrackNeg_ALamKchP, fAvgSepDenCollection_TrackNeg_ALamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackNeg_ALamKchP_Tot = CombineCFs("fAvgSepCf_TrackNeg_ALamKchP_Tot", "Track- (ALam-K+ (Tot))", fAvgSepCfCollection_TrackNeg_ALamKchP, fAvgSepNumCollection_TrackNeg_ALamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);

  //--ALam-KchM
  fAvgSepNumCollection_TrackPos_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,NumName_AvgSepTrackPos_ALamKchM);
  fAvgSepDenCollection_TrackPos_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,DenName_AvgSepTrackPos_ALamKchM);
  fAvgSepCfCollection_TrackPos_ALamKchM = BuildCollectionOfCfs(fAvgSepNumCollection_TrackPos_ALamKchM, fAvgSepDenCollection_TrackPos_ALamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackPos_ALamKchM_Tot = CombineCFs("fAvgSepCf_TrackPos_ALamKchM_Tot", "Track+ (ALam-K- (Tot))", fAvgSepCfCollection_TrackPos_ALamKchM, fAvgSepNumCollection_TrackPos_ALamKchM, fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_TrackNeg_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,NumName_AvgSepTrackNeg_ALamKchM);
  fAvgSepDenCollection_TrackNeg_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,DenName_AvgSepTrackNeg_ALamKchM);
  fAvgSepCfCollection_TrackNeg_ALamKchM = BuildCollectionOfCfs(fAvgSepNumCollection_TrackNeg_ALamKchM, fAvgSepDenCollection_TrackNeg_ALamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackNeg_ALamKchM_Tot = CombineCFs("fAvgSepCf_TrackNeg_ALamKchM_Tot", "Track- (ALam-K- (Tot))", fAvgSepCfCollection_TrackNeg_ALamKchM, fAvgSepNumCollection_TrackNeg_ALamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);


}

//________________________________________________________________________________________________________________
void buildAllcLamcKch2::DrawFinalAvgSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aAvgSepCf_TrackPos_LamKchP = fAvgSepCf_TrackPos_LamKchP_Tot->Clone();
  TH1F *aAvgSepCf_TrackNeg_LamKchP = fAvgSepCf_TrackNeg_LamKchP_Tot->Clone();

  TH1F *aAvgSepCf_TrackPos_LamKchM = fAvgSepCf_TrackPos_LamKchM_Tot->Clone();
  TH1F *aAvgSepCf_TrackNeg_LamKchM = fAvgSepCf_TrackNeg_LamKchM_Tot->Clone();
  //
  TH1F *aAvgSepCf_TrackPos_ALamKchP = fAvgSepCf_TrackPos_ALamKchP_Tot->Clone();
  TH1F *aAvgSepCf_TrackNeg_ALamKchP = fAvgSepCf_TrackNeg_ALamKchP_Tot->Clone();

  TH1F *aAvgSepCf_TrackPos_ALamKchM = fAvgSepCf_TrackPos_ALamKchM_Tot->Clone();
  TH1F *aAvgSepCf_TrackNeg_ALamKchM = fAvgSepCf_TrackNeg_ALamKchM_Tot->Clone();


  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  //------------------------------------------------------------------
  //-------LamKchP------------
  aCanvasLamKchP->cd();
  aCanvasLamKchP->Divide(1,2);

  aCanvasLamKchP->cd(1);
  aAvgSepCf_TrackPos_LamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_TrackPos_LamKchP->SetTitle("p(#Lambda) - K+");
  aAvgSepCf_TrackPos_LamKchP->Draw();
  line->Draw();

  aCanvasLamKchP->cd(2);
  aAvgSepCf_TrackNeg_LamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_TrackNeg_LamKchP->SetTitle("#pi^{-}(#Lambda) - K+");
  aAvgSepCf_TrackNeg_LamKchP->Draw();
  line->Draw();

  //-------LamKchM------------
  aCanvasLamKchM->cd();
  aCanvasLamKchM->Divide(1,2);

  aCanvasLamKchM->cd(1);
  aAvgSepCf_TrackPos_LamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_TrackPos_LamKchM->SetTitle("p(#Lambda) - K-");
  aAvgSepCf_TrackPos_LamKchM->Draw();
  line->Draw();

  aCanvasLamKchM->cd(2);
  aAvgSepCf_TrackNeg_LamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_TrackNeg_LamKchM->SetTitle("#pi^{-}(#Lambda) - K-");
  aAvgSepCf_TrackNeg_LamKchM->Draw();
  line->Draw();

  //-------ALamKchP------------
  aCanvasALamKchP->cd();
  aCanvasALamKchP->Divide(1,2);

  aCanvasALamKchP->cd(1);
  aAvgSepCf_TrackPos_ALamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_TrackPos_ALamKchP->SetTitle("#pi^{+}(#bar{#Lambda}) - K+");
  aAvgSepCf_TrackPos_ALamKchP->Draw();
  line->Draw();

  aCanvasALamKchP->cd(2);
  aAvgSepCf_TrackNeg_ALamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_TrackNeg_ALamKchP->SetTitle("#bar{p}(#bar{#Lambda}) - K+");
  aAvgSepCf_TrackNeg_ALamKchP->Draw();
  line->Draw();

  //-------ALamKchM------------
  aCanvasALamKchM->cd();
  aCanvasALamKchM->Divide(1,2);

  aCanvasALamKchM->cd(1);
  aAvgSepCf_TrackPos_ALamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_TrackPos_ALamKchM->SetTitle("#pi^{+}(#bar{#Lambda}) - K-");
  aAvgSepCf_TrackPos_ALamKchM->Draw();
  line->Draw();

  aCanvasALamKchM->cd(2);
  aAvgSepCf_TrackNeg_ALamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_TrackNeg_ALamKchM->SetTitle("#bar{p}(#bar{#Lambda}) - K-");
  aAvgSepCf_TrackNeg_ALamKchM->Draw();
  line->Draw();


}













//________________________________________________________________________________________________________________
void buildAllcLamcKch2::SetPurityRegimes(TH1F* aLambdaPurity)
{
  fLamBgFitLow[0] = 1.09;
  fLamBgFitLow[1] = 1.102;

  fLamBgFitHigh[0] = 1.130;
  fLamBgFitHigh[1] = aLambdaPurity->GetBinLowEdge(aLambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)

  fLamROI[0] = LambdaMass-0.0038;
  fLamROI[1] = LambdaMass+0.0038;

}




//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch2::CalculatePurity(char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2])
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
  TObjArray* temp = new TObjArray();
  temp->Add(fitBgd2);
  temp->Add(vInfo);
  temp->Add(vROI);
  temp->Add(vBgFitLow);
  temp->Add(vBgFitHigh);
  return temp;

}

//________________________________________________________________________________________________________________
TH1F* buildAllcLamcKch2::CombineCollectionOfHistograms(TString aReturnHistoName, vector<TH1F*> &aCollectionOfHistograms)
{
  TH1F* ReturnHisto = aCollectionOfHistograms[0]->Clone(aReturnHistoName);
  ReturnHisto->Sumw2();

  for(int i=1; i<aCollectionOfHistograms.size(); i++)
  {
    ReturnHisto->Add(aCollectionOfHistograms[i]);
  }

  return ReturnHisto;

}


//________________________________________________________________________________________________________________
void buildAllcLamcKch2::BuildPurityCollections()
{
  TString Name_LambdaPurity = "LambdaPurity";
  TString Name_AntiLambdaPurity = "AntiLambdaPurity";
  //----------------------------------------------------------
  fLambdaPurityHistogramCollection_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,Name_LambdaPurity);
  fLambdaPurityHistogramCollection_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,Name_LambdaPurity);

  fAntiLambdaPurityHistogramCollection_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,Name_AntiLambdaPurity);
  fAntiLambdaPurityHistogramCollection_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,Name_AntiLambdaPurity);
  //----------------------------------------------------------
  SetPurityRegimes(fLambdaPurityHistogramCollection_LamKchP[0]);
  //----------------------------------------------------------
  for(int i=0; i<fLambdaPurityHistogramCollection_LamKchP.size(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("LambdaPurity_LamKchP",fLambdaPurityHistogramCollection_LamKchP[i],fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fLambdaPurityListCollection_LamKchP.push_back(TempList);
  }

  for(int i=0; i<fLambdaPurityHistogramCollection_LamKchM.size(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("LambdaPurity_LamKchM",fLambdaPurityHistogramCollection_LamKchM[i],fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fLambdaPurityListCollection_LamKchM.push_back(TempList);
  }


  for(int i=0; i<fAntiLambdaPurityHistogramCollection_ALamKchP.size(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("AntiLambdaPurity_ALamKchP",fAntiLambdaPurityHistogramCollection_ALamKchP[i],fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fAntiLambdaPurityListCollection_ALamKchP.push_back(TempList);
  }

  for(int i=0; i<fAntiLambdaPurityHistogramCollection_ALamKchM.size(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("AntiLambdaPurity_ALamKchM",fAntiLambdaPurityHistogramCollection_ALamKchM[i],fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fAntiLambdaPurityListCollection_ALamKchM.push_back(TempList);
  }

  //----------------------------------------------------------
  fLambdaPurityTot_LamKchP = CombineCollectionOfHistograms("fLambdaPurityTot_LamKchP",fLambdaPurityHistogramCollection_LamKchP);
  fLambdaPurityTot_LamKchM = CombineCollectionOfHistograms("fLambdaPurityTot_LamKchM",fLambdaPurityHistogramCollection_LamKchM);
  fAntiLambdaPurityTot_ALamKchP = CombineCollectionOfHistograms("fAntiLambdaPurityTot_ALamKchP",fAntiLambdaPurityHistogramCollection_ALamKchP);
  fAntiLambdaPurityTot_ALamKchM = CombineCollectionOfHistograms("fAntiLambdaPurityTot_ALamKchM",fAntiLambdaPurityHistogramCollection_ALamKchM);
  //----------------------------------------------------------
  fLambdaPurityListTot_LamKchP = CalculatePurity("LambdaPurity_LamKchP",fLambdaPurityTot_LamKchP,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  fLambdaPurityListTot_LamKchM = CalculatePurity("LambdaPurity_LamKchM",fLambdaPurityTot_LamKchM,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  fAntiLambdaPurityListTot_ALamKchP = CalculatePurity("AntiLambdaPurity_ALamKchP",fAntiLambdaPurityTot_ALamKchP,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  fAntiLambdaPurityListTot_ALamKchM = CalculatePurity("AntiLambdaPurity_ALamKchM",fAntiLambdaPurityTot_ALamKchM,fLamBgFitLow,fLamBgFitHigh,fLamROI);

}



//________________________________________________________________________________________________________________
void buildAllcLamcKch2::DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg)
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

  if(!ZoomBg)
  {
    double HistoMaxValue = aPurityHisto->GetMaximum();
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
    aPurityHisto->GetXaxis()->SetRange(aPurityHisto->FindBin(vBgFitLow(0)),aPurityHisto->FindBin(vBgFitLow(1)));
      double maxLow = aPurityHisto->GetMaximum();
      double minLow = aPurityHisto->GetMinimum();
    aPurityHisto->GetXaxis()->SetRange(aPurityHisto->FindBin(vBgFitHigh(0)),aPurityHisto->FindBin(vBgFitHigh(1))-1);
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
    double purity = vInfo(3);
    char title[20] = aPurityHisto->GetName();
    sprintf(buffer, "%s = %.2f\%",title, 100.*purity);
    myText->AddText(buffer);
    myText->Draw();
  }
}


//________________________________________________________________________________________________________________
void buildAllcLamcKch2::DrawFinalPurity(TCanvas *aCanvas)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aLambdaPurity_LamKchP = fLambdaPurityTot_LamKchP->Clone();
  TH1F *aLambdaPurity_LamKchM = fLambdaPurityTot_LamKchM->Clone();

  TH1F *aAntiLambdaPurity_ALamKchP = fAntiLambdaPurityTot_ALamKchP->Clone();
  TH1F *aAntiLambdaPurity_ALamKchM = fAntiLambdaPurityTot_ALamKchM->Clone();
  //----------------
  TObjArray *aLambdaPurityList_LamKchP = fLambdaPurityListTot_LamKchP->Clone();
  TObjArray *aLambdaPurityList_LamKchM = fLambdaPurityListTot_LamKchM->Clone();

  TObjArray *aAntiLambdaPurityList_ALamKchP = fAntiLambdaPurityListTot_ALamKchP->Clone();
  TObjArray *aAntiLambdaPurityList_ALamKchM = fAntiLambdaPurityListTot_ALamKchM->Clone();


  aCanvas->cd();
  aCanvas->Divide(2,4);
  //-----
  aCanvas->cd(1);
  DrawPurity(aLambdaPurity_LamKchP,aLambdaPurityList_LamKchP,false);
  aCanvas->cd(2);
  DrawPurity(aLambdaPurity_LamKchP,aLambdaPurityList_LamKchP,true);
  //-----
  aCanvas->cd(3);
  DrawPurity(aLambdaPurity_LamKchM,aLambdaPurityList_LamKchM,false);
  aCanvas->cd(4);
  DrawPurity(aLambdaPurity_LamKchM,aLambdaPurityList_LamKchM,true);
  //-----
  aCanvas->cd(5);
  DrawPurity(aAntiLambdaPurity_ALamKchP,aAntiLambdaPurityList_ALamKchP,false);
  aCanvas->cd(6);
  DrawPurity(aAntiLambdaPurity_ALamKchP,aAntiLambdaPurityList_ALamKchP,true);
  //-----
  aCanvas->cd(7);
  DrawPurity(aAntiLambdaPurity_ALamKchM,aAntiLambdaPurityList_ALamKchM,false);
  aCanvas->cd(8);
  DrawPurity(aAntiLambdaPurity_ALamKchM,aAntiLambdaPurityList_ALamKchM,true);

}















//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
