///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamcKch3                                                        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "buildAllcLamcKch3.h"

#ifdef __ROOT__
ClassImp(buildAllcLamcKch3)
#endif

//________________________________________________________________________________________________________________
buildAllcLamcKch3::buildAllcLamcKch3(vector<TString> &aVectorOfFileNames, TString aDirNameLamKchP, TString aDirNameLamKchM, TString aDirNameALamKchP, TString aDirNameALamKchM):
  fDebug(kFALSE),
  fOutputPurityFitInfo(kFALSE),

  //General stuff----------------
  fDirNameLamKchP(aDirNameLamKchP),
  fDirNameLamKchM(aDirNameLamKchM),
  fDirNameALamKchP(aDirNameALamKchP),
  fDirNameALamKchM(aDirNameALamKchM),

  //KStar CF------------------
  fMinNormCF(0.32),
  fMaxNormCF(0.4),
  fMinNormBinCF(60),
  fMaxNormBinCF(75),

  //Average Separation CF------------------
  fMinNormAvgSepCF(14.99),
  fMaxNormAvgSepCF(19.99),
  fMinNormBinAvgSepCF(150),
  fMaxNormBinAvgSepCF(200),

  //Separation CFs------------------
  fMinNormBinSepCF(150),
  fMaxNormBinSepCF(200)


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
buildAllcLamcKch3::~buildAllcLamcKch3()
{
  cout << "Object is being deleted" << endl;
}

//________________________________________________________________________________________________________________
TH1F* buildAllcLamcKch3::buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int aMinNormBin, int aMaxNormBin)
{
//  cout << "For hist "<< Num->GetName()<<", num pair = " << Num->Integral(1,Num->FindBin(.05)) << endl;
  double NumScale = Num->Integral(aMinNormBin,aMaxNormBin);
  double DenScale = Denom->Integral(aMinNormBin,aMaxNormBin);

  TH1F* CF = (TH1F*)Num->Clone(name);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!CF->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << name << ", so calling it now" << endl;
    CF->Sumw2();
  }

  CF->Divide(Denom);
  CF->Scale(DenScale/NumScale);
  CF->SetTitle(title);

  return CF;
}

//________________________________________________________________________________________________________________
TH1F* buildAllcLamcKch3::CombineCFs(TString aReturnName, TString aReturnTitle, TObjArray* aCfCollection, TObjArray* aNumCollection, int aMinNormBin, int aMaxNormBin)
{
  double scale = 0.;
  int counter = 0;
  double temp = 0.;

  int SizeOfCfCollection = aCfCollection->GetEntries();
  int SizeOfNumCollection = aNumCollection->GetEntries();

  if(SizeOfCfCollection != SizeOfNumCollection) {cout << "ERROR: In CombineCFs, the CfCollection and NumCollection ARE NOT EQUAL IN SIZE!!!!" << endl;}

  TH1F* ReturnCf = (TH1F*)(aCfCollection->At(0))->Clone(aReturnName);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnCf->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aReturnName << ", so calling it now" << endl;
    ReturnCf->Sumw2();
  }

  ReturnCf->SetTitle(aReturnTitle);

  TH1F* Num1 = (TH1F*)(aNumCollection->At(0));
  temp = Num1->Integral(aMinNormBin,aMaxNormBin);

  scale+=temp;
  counter++;

  ReturnCf->Scale(temp);

  if(fDebug)
  {
    cout << "Return Name: " << ReturnCf->GetName() << endl;
    cout << "  Including: " << ((TH1F*)aCfCollection->At(0))->GetName() << endl;
    cout << "\t" << Num1->GetName() << "  NumScale: " << Num1->Integral(aMinNormBin,aMaxNormBin) << endl;
  }

  for(int i=1; i<SizeOfCfCollection; i++)
  {
    temp = ((TH1F*)aNumCollection->At(i))->Integral(aMinNormBin,aMaxNormBin);

    ReturnCf->Add((TH1F*)aCfCollection->At(i),temp);
    scale += temp;
    counter ++;

    if(fDebug)
    {
      cout << "  Including: " << ((TH1F*)aCfCollection->At(i))->GetName() << endl;
      cout << "\t" << ((TH1F*)aNumCollection->At(i))->GetName() << " NumScale: " << ((TH1F*)aNumCollection->At(i))->Integral(aMinNormBin,aMaxNormBin) << endl;
    }
  }
  ReturnCf->Scale(1./scale);

  if(fDebug)
  {
    cout << "  Final overall scale = " << scale << endl;
    cout << "  Number of histograms combined = " << counter << endl << endl;
  }

  return ReturnCf;

}



//________________________________________________________________________________________________________________
void buildAllcLamcKch3::SetVectorOfFileNames(vector<TString> &aVectorOfNames)
{
  int InputVectorSize = aVectorOfNames.size();
  if(InputVectorSize != 5) {cout << "MISSING FILE(S)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl << endl <<  "INPUT VECTOR HAS SIZE: " << InputVectorSize << endl;}

  for(int i=0; i<InputVectorSize; i++)
  {
    fVectorOfFileNames.push_back(aVectorOfNames[i]);
  }

  //Double check that all names were loaded correctly
  int OutputVectorSize = fVectorOfFileNames.size();
  cout << "fVectorOfFileNames now has size " << OutputVectorSize << " and contains..." << endl;
  for(int i=0; i<OutputVectorSize; i++)
  {
    cout << "\t" << fVectorOfFileNames[i] << endl;
  }
  cout << endl;
}


//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch3::ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName)
{
  TFile f1(aFileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  TObjArray *ReturnArray = (TObjArray*)femtolist->FindObject(aDirectoryName);

  return ReturnArray;
}

//________________________________________________________________________________________________________________
void buildAllcLamcKch3::SetAnalysisDirectories()
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
TObjArray* buildAllcLamcKch3::GetAnalysisDirectory(TString aFile, TString aDirectoryName)
{
  if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBp1;}
  else if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBp1;}
  else if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBp1;}
  else if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBp1;}

  else if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBp2;}
  else if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBp2;}
  else if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBp2;}
  else if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBp2;}

  else if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBm1;}
  else if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBm1;}
  else if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBm1;}
  else if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBm1;}

  else if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBm2;}
  else if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBm2;}
  else if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBm2;}
  else if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBm2;}

  else if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("LamKchP")){return fDirLamKchPBm3;}
  else if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("LamKchM")){return fDirLamKchMBm3;}
  else if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("ALamKchP")){return fDirALamKchPBm3;}
  else if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("ALamKchM")){return fDirALamKchMBm3;}

  else
  {
    cout << "ERROR in GetAnalysisDirectory:  No directory to return!!!!!" << endl;
    return 0;
  }

}


//________________________________________________________________________________________________________________
TH1F* buildAllcLamcKch3::GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName)
{
  TH1F *ReturnHisto = (TH1F*)aAnalysisDirectory->FindObject(aHistoName);
  TH1F *ReturnHistoClone = (TH1F*)ReturnHisto->Clone(aCloneHistoName);
  ReturnHistoClone->SetDirectory(0);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHistoClone->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aCloneHistoName << ", so calling it now" << endl;
    ReturnHistoClone->Sumw2();
  }

  return (TH1F*)ReturnHistoClone;
}


//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch3::LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName)
{
  TObjArray* ReturnCollection = new TObjArray();

  if(aDirectoryName.EqualTo(fDirNameLamKchP))
  {
    ReturnCollection->Add(GetHistoClone(fDirLamKchPBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(GetHistoClone(fDirLamKchPBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(GetHistoClone(fDirLamKchPBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(GetHistoClone(fDirLamKchPBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(GetHistoClone(fDirLamKchPBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameLamKchM))
  {
    ReturnCollection->Add(GetHistoClone(fDirLamKchMBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(GetHistoClone(fDirLamKchMBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(GetHistoClone(fDirLamKchMBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(GetHistoClone(fDirLamKchMBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(GetHistoClone(fDirLamKchMBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamKchP))
  {
    ReturnCollection->Add(GetHistoClone(fDirALamKchPBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(GetHistoClone(fDirALamKchPBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(GetHistoClone(fDirALamKchPBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(GetHistoClone(fDirALamKchPBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(GetHistoClone(fDirALamKchPBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamKchM))
  {
    ReturnCollection->Add(GetHistoClone(fDirALamKchMBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(GetHistoClone(fDirALamKchMBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(GetHistoClone(fDirALamKchMBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(GetHistoClone(fDirALamKchMBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(GetHistoClone(fDirALamKchMBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else{cout << "ERROR IN LoadCollectionOfHistograms!!!!!!!!!!!!!!!!" << endl;}

  return ReturnCollection;

}

//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch3::BuildCollectionOfCfs(TString aContainedHistosBaseName, TObjArray* aNumCollection, TObjArray* aDenCollection, int aMinNormBin, int aMaxNormBin)
{
  TObjArray* ReturnCollection = new TObjArray();

  int SizeOfNumCollection = aNumCollection->GetEntries();
  int SizeOfDenCollection = aDenCollection->GetEntries();

  if( (SizeOfNumCollection != 5) && (SizeOfDenCollection != 5) ) {cout << "In BuildCollectionOfCfs, the input collection sizes DO NOT EQUAL 5!!!!!" << endl;}
  if( SizeOfNumCollection != SizeOfDenCollection ) {cout << "In BuildCollectionOfCfs, the NumCollection and DenCollection DO NOT HAVE EQUAL SIZES!!!!!!!" << endl;}

  for(int i=0; i<SizeOfNumCollection; i++)
  {
    //------Add a file tag--------
    TString aNumName = ((TH1F*)aNumCollection->At(i))->GetName();
    TString aHistoName = aContainedHistosBaseName;

    if (aNumName.Contains("Bp1")) {aHistoName+="_Bp1";}
    else if (aNumName.Contains("Bp2")) {aHistoName+="_Bp2";}
    else if (aNumName.Contains("Bm1")) {aHistoName+="_Bm1";}
    else if (aNumName.Contains("Bm2")) {aHistoName+="_Bm2";}
    else if (aNumName.Contains("Bm3")) {aHistoName+="_Bm3";}

    ReturnCollection->Add(buildCF(aHistoName,aHistoName,((TH1F*)aNumCollection->At(i)),((TH1F*)aDenCollection->At(i)),aMinNormBin,aMaxNormBin));

  }
  return ReturnCollection;
}

//________________________________________________________________________________________________________________
void buildAllcLamcKch3::BuildCFCollections()
{
  cout << "_________________________Beginning BuildCFCollections()_________________________" << endl;

  TString NumName_LamKchP = "NumLamKchPKStarCF";
  TString DenName_LamKchP = "DenLamKchPKStarCF";

  TString NumName_LamKchM = "NumLamKchMKStarCF";
  TString DenName_LamKchM = "DenLamKchMKStarCF";

  TString NumName_ALamKchP = "NumALamKchPKStarCF";
  TString DenName_ALamKchP = "DenALamKchPKStarCF";

  TString NumName_ALamKchM = "NumALamKchMKStarCF";
  TString DenName_ALamKchM = "DenALamKchMKStarCF";

  cout << "fMinNormBinCF (default constructor) = " << fMinNormBinCF << endl;
  cout << "fMaxNormBinCF (default constructor) = " << fMaxNormBinCF << endl;

  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamKchP.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamKchP.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamKchP.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  if(!CentralityTag.IsNull())  //for some reason, the centrality dependent CFs have different naming that min bias!!!  This needs fixed
  {
    NumName_LamKchP = "NumKStarCF_LamKchP";
    DenName_LamKchP = "DenKStarCF_LamKchP";

    NumName_LamKchM = "NumKStarCF_LamKchM";
    DenName_LamKchM = "DenKStarCF_LamKchM";

    NumName_ALamKchP = "NumKStarCF_ALamKchP";
    DenName_ALamKchP = "DenKStarCF_ALamKchP";

    NumName_ALamKchM = "NumKStarCF_ALamKchM";
    DenName_ALamKchM = "DenKStarCF_ALamKchM";

  }

  NumName_LamKchP+=CentralityTag;
  DenName_LamKchP+=CentralityTag;
  NumName_LamKchM+=CentralityTag;
  DenName_LamKchM+=CentralityTag;
  NumName_ALamKchP+=CentralityTag;
  DenName_ALamKchP+=CentralityTag;
  NumName_ALamKchM+=CentralityTag;
  DenName_ALamKchM+=CentralityTag;
  //----------------------------------------------------------------------

  fNumCollection_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,NumName_LamKchP);
  fDenCollection_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,DenName_LamKchP);

  fNumCollection_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,NumName_LamKchM);
  fDenCollection_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,DenName_LamKchM);

  fNumCollection_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,NumName_ALamKchP);
  fDenCollection_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,DenName_ALamKchP);

  fNumCollection_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,NumName_ALamKchM);
  fDenCollection_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,DenName_ALamKchM);

  //-----Set the normalization bins from the normalization values (ex set fMinNormBinCF from fMinNormCF)
  fMinNormBinCF = ((TH1F*)fNumCollection_LamKchP->At(0))->FindBin(fMinNormCF);
  fMaxNormBinCF = ((TH1F*)fNumCollection_LamKchP->At(0))->FindBin(fMaxNormCF);
  cout << "fMinNormBinCF (set) = " << fMinNormBinCF << endl;
  cout << "fMaxNormBinCF (set) = " << fMaxNormBinCF << endl << endl;

  fCfCollection_LamKchP = BuildCollectionOfCfs("fCf_LamKchP",fNumCollection_LamKchP,fDenCollection_LamKchP,fMinNormBinCF,fMaxNormBinCF);
  fCf_LamKchP_Tot = CombineCFs("fCf_LamKchP_Tot","Lam-K+ (Tot)",fCfCollection_LamKchP,fNumCollection_LamKchP,fMinNormBinCF,fMaxNormBinCF);
  //fDenCollection_LamKchP->Delete();

  fCfCollection_LamKchM = BuildCollectionOfCfs("fCf_LamKchM",fNumCollection_LamKchM,fDenCollection_LamKchM,fMinNormBinCF,fMaxNormBinCF);
  fCf_LamKchM_Tot = CombineCFs("fCf_LamKchM_Tot","Lam-K- (Tot)",fCfCollection_LamKchM,fNumCollection_LamKchM,fMinNormBinCF,fMaxNormBinCF);
  //fDenCollection_LamKchM->Delete();

  fCfCollection_ALamKchP = BuildCollectionOfCfs("fCf_ALamKchP",fNumCollection_ALamKchP,fDenCollection_ALamKchP,fMinNormBinCF,fMaxNormBinCF);
  fCf_ALamKchP_Tot = CombineCFs("fCf_ALamKchP_Tot","ALam-K+ (Tot)",fCfCollection_ALamKchP,fNumCollection_ALamKchP,fMinNormBinCF,fMaxNormBinCF);
  //fDenCollection_ALamKchP->Delete();

  fCfCollection_ALamKchM = BuildCollectionOfCfs("fCf_ALamKchM",fNumCollection_ALamKchM,fDenCollection_ALamKchM,fMinNormBinCF,fMaxNormBinCF);
  fCf_ALamKchM_Tot = CombineCFs("fCf_ALamKchM_Tot","ALam-K- (Tot)",fCfCollection_ALamKchM,fNumCollection_ALamKchM,fMinNormBinCF,fMaxNormBinCF);
  //fDenCollection_ALamKchM->Delete();
/*
  //--------------BpTot-----------------------------------------------------
  TObjArray* fNumCollection_LamKchP_BpTot = new TObjArray();
    fNumCollection_LamKchP_BpTot->Add(fNumCollection_LamKchP->At(0));
    fNumCollection_LamKchP_BpTot->Add(fNumCollection_LamKchP->At(1));
  TObjArray* fCfCollection_LamKchP_BpTot = new TObjArray();
    fCfCollection_LamKchP_BpTot->Add(fCfCollection_LamKchP->At(0));
    fCfCollection_LamKchP_BpTot->Add(fCfCollection_LamKchP->At(1));
  fCf_LamKchP_BpTot = CombineCFs("fCf_LamKchP_BpTot","Lam-K+ (B+ Tot)",fCfCollection_LamKchP_BpTot,fNumCollection_LamKchP_BpTot,fMinNormBinCF,fMaxNormBinCF);

  TObjArray* fNumCollection_LamKchM_BpTot = new TObjArray();
    fNumCollection_LamKchM_BpTot->Add(fNumCollection_LamKchM->At(0));
    fNumCollection_LamKchM_BpTot->Add(fNumCollection_LamKchM->At(1));
  TObjArray* fCfCollection_LamKchM_BpTot = new TObjArray();
    fCfCollection_LamKchM_BpTot->Add(fCfCollection_LamKchM->At(0));
    fCfCollection_LamKchM_BpTot->Add(fCfCollection_LamKchM->At(1));
  fCf_LamKchM_BpTot = CombineCFs("fCf_LamKchM_BpTot","Lam-K- (B+ Tot)",fCfCollection_LamKchM_BpTot,fNumCollection_LamKchM_BpTot,fMinNormBinCF,fMaxNormBinCF);

  TObjArray* fNumCollection_ALamKchP_BpTot = new TObjArray();
    fNumCollection_ALamKchP_BpTot->Add(fNumCollection_ALamKchP->At(0));
    fNumCollection_ALamKchP_BpTot->Add(fNumCollection_ALamKchP->At(1));
  TObjArray* fCfCollection_ALamKchP_BpTot = new TObjArray();
    fCfCollection_ALamKchP_BpTot->Add(fCfCollection_ALamKchP->At(0));
    fCfCollection_ALamKchP_BpTot->Add(fCfCollection_ALamKchP->At(1));
  fCf_ALamKchP_BpTot = CombineCFs("fCf_ALamKchP_BpTot","ALam-K+ (B+ Tot)",fCfCollection_ALamKchP_BpTot,fNumCollection_ALamKchP_BpTot,fMinNormBinCF,fMaxNormBinCF);

  TObjArray* fNumCollection_ALamKchM_BpTot = new TObjArray();
    fNumCollection_ALamKchM_BpTot->Add(fNumCollection_ALamKchM->At(0));
    fNumCollection_ALamKchM_BpTot->Add(fNumCollection_ALamKchM->At(1));
  TObjArray* fCfCollection_ALamKchM_BpTot = new TObjArray();
    fCfCollection_ALamKchM_BpTot->Add(fCfCollection_ALamKchM->At(0));
    fCfCollection_ALamKchM_BpTot->Add(fCfCollection_ALamKchM->At(1));
  fCf_ALamKchM_BpTot = CombineCFs("fCf_ALamKchM_BpTot","ALam-K- (B+ Tot)",fCfCollection_ALamKchM_BpTot,fNumCollection_ALamKchM_BpTot,fMinNormBinCF,fMaxNormBinCF);


  //--------------BmTot-----------------------------------------------------
  TObjArray *fNumCollection_LamKchP_BmTot = new TObjArray();
    fNumCollection_LamKchP_BmTot->Add(fNumCollection_LamKchP->At(2));
    fNumCollection_LamKchP_BmTot->Add(fNumCollection_LamKchP->At(3));
    fNumCollection_LamKchP_BmTot->Add(fNumCollection_LamKchP->At(4));
  TObjArray* fCfCollection_LamKchP_BmTot = new TObjArray();
    fCfCollection_LamKchP_BmTot->Add(fCfCollection_LamKchP->At(2));
    fCfCollection_LamKchP_BmTot->Add(fCfCollection_LamKchP->At(3));
    fCfCollection_LamKchP_BmTot->Add(fCfCollection_LamKchP->At(4));
  fCf_LamKchP_BmTot = CombineCFs("fCf_LamKchP_BmTot","Lam-K+ (B- Tot)",fCfCollection_LamKchP_BmTot,fNumCollection_LamKchP_BmTot,fMinNormBinCF,fMaxNormBinCF);

  TObjArray *fNumCollection_LamKchM_BmTot = new TObjArray();
    fNumCollection_LamKchM_BmTot->Add(fNumCollection_LamKchM->At(2));
    fNumCollection_LamKchM_BmTot->Add(fNumCollection_LamKchM->At(3));
    fNumCollection_LamKchM_BmTot->Add(fNumCollection_LamKchM->At(4));
  TObjArray* fCfCollection_LamKchM_BmTot = new TObjArray();
    fCfCollection_LamKchM_BmTot->Add(fCfCollection_LamKchM->At(2));
    fCfCollection_LamKchM_BmTot->Add(fCfCollection_LamKchM->At(3));
    fCfCollection_LamKchM_BmTot->Add(fCfCollection_LamKchM->At(4));
  fCf_LamKchM_BmTot = CombineCFs("fCf_LamKchM_BmTot","Lam-K- (B- Tot)",fCfCollection_LamKchM_BmTot,fNumCollection_LamKchM_BmTot,fMinNormBinCF,fMaxNormBinCF);

  TObjArray* fNumCollection_ALamKchP_BmTot = new TObjArray();
    fNumCollection_ALamKchP_BmTot->Add(fNumCollection_ALamKchP->At(2));
    fNumCollection_ALamKchP_BmTot->Add(fNumCollection_ALamKchP->At(3));
    fNumCollection_ALamKchP_BmTot->Add(fNumCollection_ALamKchP->At(4));
  TObjArray* fCfCollection_ALamKchP_BmTot = new TObjArray();
    fCfCollection_ALamKchP_BmTot->Add(fCfCollection_ALamKchP->At(2));
    fCfCollection_ALamKchP_BmTot->Add(fCfCollection_ALamKchP->At(3));
    fCfCollection_ALamKchP_BmTot->Add(fCfCollection_ALamKchP->At(4));
  fCf_ALamKchP_BmTot = CombineCFs("fCf_ALamKchP_BmTot","ALam-K+ (B- Tot)",fCfCollection_ALamKchP_BmTot,fNumCollection_ALamKchP_BmTot,fMinNormBinCF,fMaxNormBinCF);

  TObjArray* fNumCollection_ALamKchM_BmTot = new TObjArray();
    fNumCollection_ALamKchM_BmTot->Add(fNumCollection_ALamKchM->At(2));
    fNumCollection_ALamKchM_BmTot->Add(fNumCollection_ALamKchM->At(3));
    fNumCollection_ALamKchM_BmTot->Add(fNumCollection_ALamKchM->At(4));
  TObjArray* fCfCollection_ALamKchM_BmTot = new TObjArray();
    fCfCollection_ALamKchM_BmTot->Add(fCfCollection_ALamKchM->At(2));
    fCfCollection_ALamKchM_BmTot->Add(fCfCollection_ALamKchM->At(3));
    fCfCollection_ALamKchM_BmTot->Add(fCfCollection_ALamKchM->At(4));
  fCf_ALamKchM_BmTot = CombineCFs("fCf_ALamKchM_BmTot","ALam-K- (B- Tot)",fCfCollection_ALamKchM_BmTot,fNumCollection_ALamKchM_BmTot,fMinNormBinCF,fMaxNormBinCF);
*/


  //------------Average KchP and KchM correlation functions--------------------------
    //--1 April 2015
  TObjArray* fNumCollection_AverageLamKchPM = new TObjArray();
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchP->At(0)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchP->At(1)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchP->At(2)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchP->At(3)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchP->At(4)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchM->At(0)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchM->At(1)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchM->At(2)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchM->At(3)->Clone());
    fNumCollection_AverageLamKchPM->Add(fNumCollection_LamKchM->At(4)->Clone());
  TObjArray* fCfCollection_AverageLamKchPM = new TObjArray();
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchP->At(0)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchP->At(1)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchP->At(2)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchP->At(3)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchP->At(4)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchM->At(0)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchM->At(1)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchM->At(2)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchM->At(3)->Clone());
    fCfCollection_AverageLamKchPM->Add(fCfCollection_LamKchM->At(4)->Clone());

  fCf_AverageLamKchPM_Tot = CombineCFs("fCf_AverageLamKchPM_Tot","Avg Lam-K+-",fCfCollection_AverageLamKchPM,fNumCollection_AverageLamKchPM,fMinNormBinCF,fMaxNormBinCF);

  fNumCollection_AverageLamKchPM->Delete();
  fCfCollection_AverageLamKchPM->Delete();

  TObjArray* fNumCollection_AverageALamKchPM = new TObjArray();
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchP->At(0)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchP->At(1)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchP->At(2)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchP->At(3)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchP->At(4)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchM->At(0)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchM->At(1)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchM->At(2)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchM->At(3)->Clone());
    fNumCollection_AverageALamKchPM->Add(fNumCollection_ALamKchM->At(4)->Clone());
  TObjArray* fCfCollection_AverageALamKchPM = new TObjArray();
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchP->At(0)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchP->At(1)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchP->At(2)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchP->At(3)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchP->At(4)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchM->At(0)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchM->At(1)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchM->At(2)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchM->At(3)->Clone());
    fCfCollection_AverageALamKchPM->Add(fCfCollection_ALamKchM->At(4)->Clone());
  fCf_AverageALamKchPM_Tot = CombineCFs("fCf_AverageALamKchPM_Tot","Avg ALam-K+-",fCfCollection_AverageALamKchPM,fNumCollection_AverageALamKchPM,fMinNormBinCF,fMaxNormBinCF);
  fNumCollection_AverageALamKchPM->Delete();
  fCfCollection_AverageALamKchPM->Delete();


  cout << "_________________________Done BuildCFCollections()_________________________" << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void buildAllcLamcKch3::DrawFinalCFs(TCanvas *aCanvas)
{
  aCanvas->cd();
  gStyle->SetOptStat(0);

  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aCfLamKchP = (TH1F*)fCf_LamKchP_Tot->Clone();
  TH1F *aCfLamKchM = (TH1F*)fCf_LamKchM_Tot->Clone();
  TH1F *aCfALamKchP = (TH1F*)fCf_ALamKchP_Tot->Clone();
  TH1F *aCfALamKchM = (TH1F*)fCf_ALamKchM_Tot->Clone();

  TAxis *xax1 = aCfLamKchP->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
  TAxis *yax1 = aCfLamKchP->GetYaxis();
    yax1->SetRangeUser(0.9,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();
/*
  TAxis *xax2 = aCfALamKchM->GetXaxis();
    xax2->SetTitle("k* (GeV/c)");
    xax2->SetTitleSize(0.05);
    xax2->SetTitleOffset(1.0);
    //xax2->CenterTitle();
  TAxis *yax2 = aCfALamKchM->GetYaxis();
    yax2->SetRangeUser(0.9,1.1);
    yax2->SetTitle("C(k*)");
    yax2->SetTitleSize(0.05);
    yax2->SetTitleOffset(1.0);
    yax2->CenterTitle();
*/
/*
  TAxis *xax3 = aCfLamKchM->GetXaxis();
    xax3->SetTitle("k* (GeV/c)");
    xax3->SetTitleSize(0.05);
    xax3->SetTitleOffset(1.0);
    //xax3->CenterTitle();
  TAxis *yax3 = aCfLamKchM->GetYaxis();
    yax3->SetRangeUser(0.9,1.1);
    yax3->SetTitle("C(k*)");
    yax3->SetTitleSize(0.05);
    yax3->SetTitleOffset(1.0);
    yax3->CenterTitle();
*/
  TAxis *xax4 = aCfALamKchP->GetXaxis();
    xax4->SetTitle("k* (GeV/c)");
    xax4->SetTitleSize(0.05);
    xax4->SetTitleOffset(1.0);
    //xax4->CenterTitle();
  TAxis *yax4 = aCfALamKchP->GetYaxis();
    yax4->SetRangeUser(0.9,1.1);
    yax4->SetTitle("C(k*)");
    yax4->SetTitleSize(0.05);
    yax4->SetTitleOffset(1.0);
    yax4->CenterTitle();
  //------------------------------------------------------
  aCfLamKchP->SetMarkerStyle(20);
  aCfLamKchP->SetMarkerSize(0.75);
  aCfLamKchP->SetMarkerColor(2);
  aCfLamKchP->SetLineColor(2);
  aCfLamKchP->SetTitle("#LambdaK+ & #LambdaK-");

  aCfLamKchM->SetMarkerStyle(20);
  aCfLamKchM->SetMarkerSize(0.75);
  aCfLamKchM->SetMarkerColor(4);
  aCfLamKchM->SetLineColor(4);
  //aCfLamKchM->SetTitle("#Lambda - K-");

  aCfALamKchP->SetMarkerStyle(20);
  aCfALamKchP->SetMarkerSize(0.75);
  aCfALamKchP->SetMarkerColor(4);
  aCfALamKchP->SetLineColor(4);
  aCfALamKchP->SetTitle("#bar{#Lambda}K+ & #bar{#Lambda}K-");

  aCfALamKchM->SetMarkerStyle(20);
  aCfALamKchM->SetMarkerSize(0.75);
  aCfALamKchM->SetMarkerColor(2);
  aCfALamKchM->SetLineColor(2);
  //aCfALamKchM->SetTitle("#bar{#Lambda} - K-");

  //------------------------------------------------------
  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);

  aCanvas->Divide(2,1);

  aCanvas->cd(1);
  aCfLamKchP->Draw();
  aCfLamKchM->Draw("same");
  line->Draw();
  TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
  leg1->SetFillColor(0);
  leg1->AddEntry(aCfLamKchP, "#LambdaK+","lp");
  leg1->AddEntry(aCfLamKchM, "#LambdaK-","lp");
  leg1->Draw();


  aCanvas->cd(2);
  aCfALamKchP->Draw();
  aCfALamKchM->Draw("same");
  line->Draw();
  TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
  leg2->SetFillColor(0);
  leg2->AddEntry(aCfALamKchM, "#bar{#Lambda}K-","lp");
  leg2->AddEntry(aCfALamKchP, "#bar{#Lambda}K+","lp");
  leg2->Draw();
}

//________________________________________________________________________________________________________________
void buildAllcLamcKch3::BuildAvgSepCollections()
{
  cout << "_________________________Beginning BuildAvgSepCollections()_________________________" << endl;

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
  cout << "fMinNormBinAvgSepCF (default constructor) = " << fMinNormBinAvgSepCF << endl;
  cout << "fMaxNormBinAvgSepCF (default constructor) = " << fMaxNormBinAvgSepCF << endl;

  //--LamKchP
  fAvgSepNumCollection_TrackPos_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,NumName_AvgSepTrackPos_LamKchP);
  fAvgSepDenCollection_TrackPos_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,DenName_AvgSepTrackPos_LamKchP);

  //-----Set the normalization bins from the normalization values (ex set fMinNormBinAvgSepCF from fMinNormAvgSepCF)
  fMinNormBinAvgSepCF = ((TH1F*)fAvgSepNumCollection_TrackPos_LamKchP->At(0))->FindBin(fMinNormAvgSepCF);
  fMaxNormBinAvgSepCF = ((TH1F*)fAvgSepNumCollection_TrackPos_LamKchP->At(0))->FindBin(fMaxNormAvgSepCF);
  cout << "fMinNormBinAvgSepCF (set) = " << fMinNormBinAvgSepCF << endl;
  cout << "fMaxNormBinAvgSepCF (set) = " << fMaxNormBinAvgSepCF << endl << endl;

  fAvgSepCfCollection_TrackPos_LamKchP = BuildCollectionOfCfs("fAvgSepCf_TrackPos_LamKchP",fAvgSepNumCollection_TrackPos_LamKchP, fAvgSepDenCollection_TrackPos_LamKchP,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackPos_LamKchP_Tot = CombineCFs("fAvgSepCf_TrackPos_LamKchP_Tot", "Track+ (Lam-K+ (Tot))", fAvgSepCfCollection_TrackPos_LamKchP, fAvgSepNumCollection_TrackPos_LamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_TrackNeg_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,NumName_AvgSepTrackNeg_LamKchP);
  fAvgSepDenCollection_TrackNeg_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,DenName_AvgSepTrackNeg_LamKchP);
  fAvgSepCfCollection_TrackNeg_LamKchP = BuildCollectionOfCfs("fAvgSepCf_TrackNeg_LamKchP",fAvgSepNumCollection_TrackNeg_LamKchP, fAvgSepDenCollection_TrackNeg_LamKchP,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackNeg_LamKchP_Tot = CombineCFs("fAvgSepCf_TrackNeg_LamKchP_Tot", "Track- (Lam-K+ (Tot))", fAvgSepCfCollection_TrackNeg_LamKchP, fAvgSepNumCollection_TrackNeg_LamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);

  //--LamKchM
  fAvgSepNumCollection_TrackPos_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,NumName_AvgSepTrackPos_LamKchM);
  fAvgSepDenCollection_TrackPos_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,DenName_AvgSepTrackPos_LamKchM);
  fAvgSepCfCollection_TrackPos_LamKchM = BuildCollectionOfCfs("fAvgSepCf_TrackPos_LamKchM",fAvgSepNumCollection_TrackPos_LamKchM, fAvgSepDenCollection_TrackPos_LamKchM,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackPos_LamKchM_Tot = CombineCFs("fAvgSepCf_TrackPos_LamKchM_Tot", "Track+ (Lam-K- (Tot))", fAvgSepCfCollection_TrackPos_LamKchM, fAvgSepNumCollection_TrackPos_LamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_TrackNeg_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,NumName_AvgSepTrackNeg_LamKchM);
  fAvgSepDenCollection_TrackNeg_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,DenName_AvgSepTrackNeg_LamKchM);
  fAvgSepCfCollection_TrackNeg_LamKchM = BuildCollectionOfCfs("fAvgSepCf_TrackNeg_LamKchM",fAvgSepNumCollection_TrackNeg_LamKchM, fAvgSepDenCollection_TrackNeg_LamKchM,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackNeg_LamKchM_Tot = CombineCFs("fAvgSepCf_TrackNeg_LamKchM_Tot", "Track- (Lam-K- (Tot))", fAvgSepCfCollection_TrackNeg_LamKchM, fAvgSepNumCollection_TrackNeg_LamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);


  //--ALam-KchP
  fAvgSepNumCollection_TrackPos_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,NumName_AvgSepTrackPos_ALamKchP);
  fAvgSepDenCollection_TrackPos_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,DenName_AvgSepTrackPos_ALamKchP);
  fAvgSepCfCollection_TrackPos_ALamKchP = BuildCollectionOfCfs("fAvgSepCf_TrackPos_ALamKchP",fAvgSepNumCollection_TrackPos_ALamKchP, fAvgSepDenCollection_TrackPos_ALamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackPos_ALamKchP_Tot = CombineCFs("fAvgSepCf_TrackPos_ALamKchP_Tot", "Track+ (ALam-K+ (Tot))", fAvgSepCfCollection_TrackPos_ALamKchP, fAvgSepNumCollection_TrackPos_ALamKchP, fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_TrackNeg_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,NumName_AvgSepTrackNeg_ALamKchP);
  fAvgSepDenCollection_TrackNeg_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,DenName_AvgSepTrackNeg_ALamKchP);
  fAvgSepCfCollection_TrackNeg_ALamKchP = BuildCollectionOfCfs("fAvgSepCf_TrackNeg_ALamKchP",fAvgSepNumCollection_TrackNeg_ALamKchP, fAvgSepDenCollection_TrackNeg_ALamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackNeg_ALamKchP_Tot = CombineCFs("fAvgSepCf_TrackNeg_ALamKchP_Tot", "Track- (ALam-K+ (Tot))", fAvgSepCfCollection_TrackNeg_ALamKchP, fAvgSepNumCollection_TrackNeg_ALamKchP, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);

  //--ALam-KchM
  fAvgSepNumCollection_TrackPos_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,NumName_AvgSepTrackPos_ALamKchM);
  fAvgSepDenCollection_TrackPos_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,DenName_AvgSepTrackPos_ALamKchM);
  fAvgSepCfCollection_TrackPos_ALamKchM = BuildCollectionOfCfs("fAvgSepCf_TrackPos_ALamKchM",fAvgSepNumCollection_TrackPos_ALamKchM, fAvgSepDenCollection_TrackPos_ALamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackPos_ALamKchM_Tot = CombineCFs("fAvgSepCf_TrackPos_ALamKchM_Tot", "Track+ (ALam-K- (Tot))", fAvgSepCfCollection_TrackPos_ALamKchM, fAvgSepNumCollection_TrackPos_ALamKchM, fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_TrackNeg_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,NumName_AvgSepTrackNeg_ALamKchM);
  fAvgSepDenCollection_TrackNeg_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,DenName_AvgSepTrackNeg_ALamKchM);
  fAvgSepCfCollection_TrackNeg_ALamKchM = BuildCollectionOfCfs("fAvgSepCf_TrackNeg_ALamKchM",fAvgSepNumCollection_TrackNeg_ALamKchM, fAvgSepDenCollection_TrackNeg_ALamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);
  fAvgSepCf_TrackNeg_ALamKchM_Tot = CombineCFs("fAvgSepCf_TrackNeg_ALamKchM_Tot", "Track- (ALam-K- (Tot))", fAvgSepCfCollection_TrackNeg_ALamKchM, fAvgSepNumCollection_TrackNeg_ALamKchM, fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF);


  cout << "_________________________Done BuildAvgSepCollections()_________________________" << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void buildAllcLamcKch3::DrawFinalAvgSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aAvgSepCf_TrackPos_LamKchP = (TH1F*)fAvgSepCf_TrackPos_LamKchP_Tot->Clone();
  TH1F *aAvgSepCf_TrackNeg_LamKchP = (TH1F*)fAvgSepCf_TrackNeg_LamKchP_Tot->Clone();

  TH1F *aAvgSepCf_TrackPos_LamKchM = (TH1F*)fAvgSepCf_TrackPos_LamKchM_Tot->Clone();
  TH1F *aAvgSepCf_TrackNeg_LamKchM = (TH1F*)fAvgSepCf_TrackNeg_LamKchM_Tot->Clone();
  //
  TH1F *aAvgSepCf_TrackPos_ALamKchP = (TH1F*)fAvgSepCf_TrackPos_ALamKchP_Tot->Clone();
  TH1F *aAvgSepCf_TrackNeg_ALamKchP = (TH1F*)fAvgSepCf_TrackNeg_ALamKchP_Tot->Clone();

  TH1F *aAvgSepCf_TrackPos_ALamKchM = (TH1F*)fAvgSepCf_TrackPos_ALamKchM_Tot->Clone();
  TH1F *aAvgSepCf_TrackNeg_ALamKchM = (TH1F*)fAvgSepCf_TrackNeg_ALamKchM_Tot->Clone();


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
void buildAllcLamcKch3::SetPurityRegimes(TH1F* aLambdaPurity)
{
  fLamBgFitLow[0] = 1.09;
  fLamBgFitLow[1] = 1.102;

  fLamBgFitHigh[0] = 1.130;
  fLamBgFitHigh[1] = aLambdaPurity->GetBinLowEdge(aLambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)

  fLamROI[0] = LambdaMass-0.0038;
  fLamROI[1] = LambdaMass+0.0038;

}




//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch3::CalculatePurity(const char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2])
{
  char buffer[50];
  sprintf(buffer, "fitBgd_%s",aReturnFitName);

  reject = kTRUE;
  ffBgFitLow[0] = aBgFitLow[0];
  ffBgFitLow[1] = aBgFitLow[1];
  ffBgFitHigh[0] = aBgFitHigh[0];
  ffBgFitHigh[1] = aBgFitHigh[1];
  TF1 *fitBgd = new TF1("fitBgd",PurityBgFitFunction,aPurityHisto->GetBinLowEdge(1),aPurityHisto->GetBinLowEdge(aPurityHisto->GetNbinsX()+1),5);
  if(fOutputPurityFitInfo){aPurityHisto->Fit("fitBgd","0");}  //option 0 = Do not plot the result of the fit
  else{aPurityHisto->Fit("fitBgd","0q");}  //option q = quiet mode = minimum printing

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
TH1F* buildAllcLamcKch3::CombineCollectionOfHistograms(TString aReturnHistoName, TObjArray* aCollectionOfHistograms)
{
  TH1F* ReturnHisto = (TH1F*)(aCollectionOfHistograms->At(0))->Clone(aReturnHistoName);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aReturnHistoName << ", so calling it now" << endl;
    ReturnHisto->Sumw2();
  }

  for(int i=1; i<aCollectionOfHistograms->GetEntries(); i++)
  {
    ReturnHisto->Add(((TH1F*)aCollectionOfHistograms->At(i)));
  }

  return ReturnHisto;

}


//________________________________________________________________________________________________________________
void buildAllcLamcKch3::BuildPurityCollections()
{
  cout << "_________________________Beginning BuildPurityCollections()_________________________" << endl;

  TString Name_LambdaPurity = "LambdaPurity";
  TString Name_AntiLambdaPurity = "AntiLambdaPurity";
  //----------------------------------------------------------
  fLambdaPurityHistogramCollection_LamKchP = LoadCollectionOfHistograms(fDirNameLamKchP,Name_LambdaPurity);
  fLambdaPurityHistogramCollection_LamKchM = LoadCollectionOfHistograms(fDirNameLamKchM,Name_LambdaPurity);

  fAntiLambdaPurityHistogramCollection_ALamKchP = LoadCollectionOfHistograms(fDirNameALamKchP,Name_AntiLambdaPurity);
  fAntiLambdaPurityHistogramCollection_ALamKchM = LoadCollectionOfHistograms(fDirNameALamKchM,Name_AntiLambdaPurity);
  //----------------------------------------------------------
  SetPurityRegimes((TH1F*)fLambdaPurityHistogramCollection_LamKchP->At(0));
  //----------------------------------------------------------
  cout << "********** Lambda Purity (LamKchP) for each individual file (Bp1, Bp2, ...) **********" << endl;
  for(int i=0; i<fLambdaPurityHistogramCollection_LamKchP->GetEntries(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("LambdaPurity_LamKchP",(TH1F*)fLambdaPurityHistogramCollection_LamKchP->At(i),fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fLambdaPurityListCollection_LamKchP.push_back(TempList);
  }

  cout << "********** Lambda Purity (LamKchM) for each individual file (Bp1, Bp2, ...) **********" << endl;
  for(int i=0; i<fLambdaPurityHistogramCollection_LamKchM->GetEntries(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("LambdaPurity_LamKchM",(TH1F*)fLambdaPurityHistogramCollection_LamKchM->At(i),fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fLambdaPurityListCollection_LamKchM.push_back(TempList);
  }

  cout << "********** AntiLambda Purity (ALamKchP) for each individual file (Bp1, Bp2, ...) **********" << endl;
  for(int i=0; i<fAntiLambdaPurityHistogramCollection_ALamKchP->GetEntries(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("AntiLambdaPurity_ALamKchP",(TH1F*)fAntiLambdaPurityHistogramCollection_ALamKchP->At(i),fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fAntiLambdaPurityListCollection_ALamKchP.push_back(TempList);
  }

  cout << "********** AntiLambda Purity (ALamKchM) for each individual file (Bp1, Bp2, ...) **********" << endl;
  for(int i=0; i<fAntiLambdaPurityHistogramCollection_ALamKchM->GetEntries(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("AntiLambdaPurity_ALamKchM",(TH1F*)fAntiLambdaPurityHistogramCollection_ALamKchM->At(i),fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fAntiLambdaPurityListCollection_ALamKchM.push_back(TempList);
  }

  //----------------------------------------------------------
  fLambdaPurityTot_LamKchP = CombineCollectionOfHistograms("fLambdaPurityTot_LamKchP",fLambdaPurityHistogramCollection_LamKchP);
  fLambdaPurityTot_LamKchM = CombineCollectionOfHistograms("fLambdaPurityTot_LamKchM",fLambdaPurityHistogramCollection_LamKchM);
  fAntiLambdaPurityTot_ALamKchP = CombineCollectionOfHistograms("fAntiLambdaPurityTot_ALamKchP",fAntiLambdaPurityHistogramCollection_ALamKchP);
  fAntiLambdaPurityTot_ALamKchM = CombineCollectionOfHistograms("fAntiLambdaPurityTot_ALamKchM",fAntiLambdaPurityHistogramCollection_ALamKchM);
  //----------------------------------------------------------
  cout << "****************************** TOTAL Lambda Purity (LamKchP) ******************************" << endl;
  fLambdaPurityListTot_LamKchP = CalculatePurity("LambdaPurity_LamKchP",fLambdaPurityTot_LamKchP,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  cout << "****************************** TOTAL Lambda Purity (LamKchM) ******************************" << endl;
  fLambdaPurityListTot_LamKchM = CalculatePurity("LambdaPurity_LamKchM",fLambdaPurityTot_LamKchM,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  cout << "****************************** TOTAL AntiLambda Purity (ALamKchP) ******************************" << endl;
  fAntiLambdaPurityListTot_ALamKchP = CalculatePurity("AntiLambdaPurity_ALamKchP",fAntiLambdaPurityTot_ALamKchP,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  cout << "****************************** TOTAL AntiLambda Purity (ALamKchM) ******************************" << endl;
  fAntiLambdaPurityListTot_ALamKchM = CalculatePurity("AntiLambdaPurity_ALamKchM",fAntiLambdaPurityTot_ALamKchM,fLamBgFitLow,fLamBgFitHigh,fLamROI);


  cout << "_________________________Done BuildPurityCollections()_________________________" << endl << endl << endl;
}



//________________________________________________________________________________________________________________
void buildAllcLamcKch3::DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg)
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
void buildAllcLamcKch3::DrawFinalPurity(TCanvas *aCanvas)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aLambdaPurity_LamKchP = (TH1F*)fLambdaPurityTot_LamKchP->Clone();
  TH1F *aLambdaPurity_LamKchM = (TH1F*)fLambdaPurityTot_LamKchM->Clone();

  TH1F *aAntiLambdaPurity_ALamKchP = (TH1F*)fAntiLambdaPurityTot_ALamKchP->Clone();
  TH1F *aAntiLambdaPurity_ALamKchM = (TH1F*)fAntiLambdaPurityTot_ALamKchM->Clone();
  //----------------
  TObjArray *aLambdaPurityList_LamKchP = (TObjArray*)fLambdaPurityListTot_LamKchP->Clone();
  TObjArray *aLambdaPurityList_LamKchM = (TObjArray*)fLambdaPurityListTot_LamKchM->Clone();

  TObjArray *aAntiLambdaPurityList_ALamKchP = (TObjArray*)fAntiLambdaPurityListTot_ALamKchP->Clone();
  TObjArray *aAntiLambdaPurityList_ALamKchM = (TObjArray*)fAntiLambdaPurityListTot_ALamKchM->Clone();


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



//------27 April 2015
//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch3::GetCfCollection(TString aType, TString aDirectoryName)
{
  if(aType.EqualTo("Num") && aDirectoryName.EqualTo("LamKchP")){return fNumCollection_LamKchP;}
  else if(aType.EqualTo("Den") && aDirectoryName.EqualTo("LamKchP")){return fDenCollection_LamKchP;}
  else if(aType.EqualTo("Cf") && aDirectoryName.EqualTo("LamKchP")){return fCfCollection_LamKchP;}

  else if(aType.EqualTo("Num") && aDirectoryName.EqualTo("LamKchM")){return fNumCollection_LamKchM;}
  else if(aType.EqualTo("Den") && aDirectoryName.EqualTo("LamKchM")){return fDenCollection_LamKchM;}
  else if(aType.EqualTo("Cf") && aDirectoryName.EqualTo("LamKchM")){return fCfCollection_LamKchM;}

  else if(aType.EqualTo("Num") && aDirectoryName.EqualTo("ALamKchP")){return fNumCollection_ALamKchP;}
  else if(aType.EqualTo("Den") && aDirectoryName.EqualTo("ALamKchP")){return fDenCollection_ALamKchP;}
  else if(aType.EqualTo("Cf") && aDirectoryName.EqualTo("ALamKchP")){return fCfCollection_ALamKchP;}

  else if(aType.EqualTo("Num") && aDirectoryName.EqualTo("ALamKchM")){return fNumCollection_ALamKchM;}
  else if(aType.EqualTo("Den") && aDirectoryName.EqualTo("ALamKchM")){return fDenCollection_ALamKchM;}
  else if(aType.EqualTo("Cf") && aDirectoryName.EqualTo("ALamKchM")){return fCfCollection_ALamKchM;}

  else
  {
    cout << "ERROR in GetCfCollection:  No collection to return!!!!!" << endl;
    return 0;
  }

}


//-----29 April 2015
//________________________________________________________________________________________________________________
void buildAllcLamcKch3::SaveAll(TFile* aFile)
{
  //TFile *aReturnFile = new TFile(aName, aOption);

  //-----KStar Cfs
  fNumCollection_LamKchP->Write("fNumCollection_LamKchP",TObject::kSingleKey);
  fNumCollection_LamKchM->Write("fNumCollection_LamKchM",TObject::kSingleKey);
  fNumCollection_ALamKchP->Write("fNumCollection_ALamKchP",TObject::kSingleKey);
  fNumCollection_ALamKchM->Write("fNumCollection_ALamKchM",TObject::kSingleKey);

  fDenCollection_LamKchP->Write("fDenCollection_LamKchP",TObject::kSingleKey);
  fDenCollection_LamKchM->Write("fDenCollection_LamKchM",TObject::kSingleKey);
  fDenCollection_ALamKchP->Write("fDenCollection_ALamKchP",TObject::kSingleKey);
  fDenCollection_ALamKchM->Write("fDenCollection_ALamKchM",TObject::kSingleKey);

  fCfCollection_LamKchP->Write("fCfCollection_LamKchP",TObject::kSingleKey);
  fCfCollection_LamKchM->Write("fCfCollection_LamKchM",TObject::kSingleKey);
  fCfCollection_ALamKchP->Write("fCfCollection_ALamKchP",TObject::kSingleKey);
  fCfCollection_ALamKchM->Write("fCfCollection_ALamKchM",TObject::kSingleKey);

  fCf_LamKchP_Tot->Write("fCf_LamKchP_Tot");
  fCf_LamKchM_Tot->Write("fCf_LamKchM_Tot");
  fCf_ALamKchP_Tot->Write("fCf_ALamKchP_Tot");
  fCf_ALamKchM_Tot->Write("fCf_ALamKchM_Tot");

  //-----Average Separation Cfs
    //-----TrackPos
  fAvgSepNumCollection_TrackPos_LamKchP->Write("fAvgSepNumCollection_TrackPos_LamKchP",TObject::kSingleKey);
  fAvgSepNumCollection_TrackPos_LamKchM->Write("fAvgSepNumCollection_TrackPos_LamKchM",TObject::kSingleKey);
  fAvgSepNumCollection_TrackPos_ALamKchP->Write("fAvgSepNumCollection_TrackPos_ALamKchP",TObject::kSingleKey);
  fAvgSepNumCollection_TrackPos_ALamKchM->Write("fAvgSepNumCollection_TrackPos_ALamKchM",TObject::kSingleKey);

  fAvgSepDenCollection_TrackPos_LamKchP->Write("fAvgSepDenCollection_TrackPos_LamKchP",TObject::kSingleKey);
  fAvgSepDenCollection_TrackPos_LamKchM->Write("fAvgSepDenCollection_TrackPos_LamKchM",TObject::kSingleKey);
  fAvgSepDenCollection_TrackPos_ALamKchP->Write("fAvgSepDenCollection_TrackPos_ALamKchP",TObject::kSingleKey);
  fAvgSepDenCollection_TrackPos_ALamKchM->Write("fAvgSepDenCollection_TrackPos_ALamKchM",TObject::kSingleKey);

  fAvgSepCfCollection_TrackPos_LamKchP->Write("fAvgSepCfCollection_TrackPos_LamKchP",TObject::kSingleKey);
  fAvgSepCfCollection_TrackPos_LamKchM->Write("fAvgSepCfCollection_TrackPos_LamKchM",TObject::kSingleKey);
  fAvgSepCfCollection_TrackPos_ALamKchP->Write("fAvgSepCfCollection_TrackPos_ALamKchP",TObject::kSingleKey);
  fAvgSepCfCollection_TrackPos_ALamKchM->Write("fAvgSepCfCollection_TrackPos_ALamKchM",TObject::kSingleKey);

  fAvgSepCf_TrackPos_LamKchP_Tot->Write("fAvgSepCf_TrackPos_LamKchP_Tot");
  fAvgSepCf_TrackPos_LamKchM_Tot->Write("fAvgSepCf_TrackPos_LamKchM_Tot");
  fAvgSepCf_TrackPos_ALamKchP_Tot->Write("fAvgSepCf_TrackPos_ALamKchP_Tot");
  fAvgSepCf_TrackPos_ALamKchM_Tot->Write("fAvgSepCf_TrackPos_ALamKchM_Tot");

    //-----TrackNeg
  fAvgSepNumCollection_TrackNeg_LamKchP->Write("fAvgSepNumCollection_TrackNeg_LamKchP",TObject::kSingleKey);
  fAvgSepNumCollection_TrackNeg_LamKchM->Write("fAvgSepNumCollection_TrackNeg_LamKchM",TObject::kSingleKey);
  fAvgSepNumCollection_TrackNeg_ALamKchP->Write("fAvgSepNumCollection_TrackNeg_ALamKchP",TObject::kSingleKey);
  fAvgSepNumCollection_TrackNeg_ALamKchM->Write("fAvgSepNumCollection_TrackNeg_ALamKchM",TObject::kSingleKey);

  fAvgSepDenCollection_TrackNeg_LamKchP->Write("fAvgSepDenCollection_TrackNeg_LamKchP",TObject::kSingleKey);
  fAvgSepDenCollection_TrackNeg_LamKchM->Write("fAvgSepDenCollection_TrackNeg_LamKchM",TObject::kSingleKey);
  fAvgSepDenCollection_TrackNeg_ALamKchP->Write("fAvgSepDenCollection_TrackNeg_ALamKchP",TObject::kSingleKey);
  fAvgSepDenCollection_TrackNeg_ALamKchM->Write("fAvgSepDenCollection_TrackNeg_ALamKchM",TObject::kSingleKey);

  fAvgSepCfCollection_TrackNeg_LamKchP->Write("fAvgSepCfCollection_TrackNeg_LamKchP",TObject::kSingleKey);
  fAvgSepCfCollection_TrackNeg_LamKchM->Write("fAvgSepCfCollection_TrackNeg_LamKchM",TObject::kSingleKey);
  fAvgSepCfCollection_TrackNeg_ALamKchP->Write("fAvgSepCfCollection_TrackNeg_ALamKchP",TObject::kSingleKey);
  fAvgSepCfCollection_TrackNeg_ALamKchM->Write("fAvgSepCfCollection_TrackNeg_ALamKchM",TObject::kSingleKey);

  fAvgSepCf_TrackNeg_LamKchP_Tot->Write("fAvgSepCf_TrackNeg_LamKchP_Tot");
  fAvgSepCf_TrackNeg_LamKchM_Tot->Write("fAvgSepCf_TrackNeg_LamKchM_Tot");
  fAvgSepCf_TrackNeg_ALamKchP_Tot->Write("fAvgSepCf_TrackNeg_ALamKchP_Tot");
  fAvgSepCf_TrackNeg_ALamKchM_Tot->Write("fAvgSepCf_TrackNeg_ALamKchM_Tot");
}



//-----7 May 2015

//________________________________________________________________________________________________________________
TH2F* buildAllcLamcKch3::Get2DHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName)
{
  TH2F *ReturnHisto = (TH2F*)aAnalysisDirectory->FindObject(aHistoName);
  TH2F *ReturnHistoClone = (TH2F*)ReturnHisto->Clone(aCloneHistoName);
  ReturnHistoClone->SetDirectory(0);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHistoClone->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aCloneHistoName << ", so calling it now" << endl;
    ReturnHistoClone->Sumw2();
  }

  return (TH2F*)ReturnHistoClone;
}


//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch3::LoadCollectionOf2DHistograms(TString aDirectoryName, TString aHistoName)
{
  TObjArray* ReturnCollection = new TObjArray();

  if(aDirectoryName.EqualTo(fDirNameLamKchP))
  {
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchPBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchPBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchPBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchPBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchPBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameLamKchM))
  {
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchMBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchMBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchMBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchMBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamKchMBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamKchP))
  {
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchPBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchPBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchPBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchPBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchPBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamKchM))
  {
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchMBp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchMBp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchMBm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchMBm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamKchMBm3,aHistoName,aHistoName+"_Bm3"));
  }

  else{cout << "ERROR IN LoadCollectionOf2DHistograms!!!!!!!!!!!!!!!!" << endl;}

  return ReturnCollection;

}


//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch3::BuildSepCfs(int aRebinFactor, TString aContainedHistosBaseName, int aNumberOfXbins, TObjArray* a2DNumCollection, TObjArray* a2DDenCollection, int aMinNormBin, int aMaxNormBin)
{
  cout << "***** Working on collection of SepCfs containing: " << aContainedHistosBaseName << " *****" << endl;

  TObjArray* ReturnCollection = new TObjArray();

  int SizeOfNumCollection = a2DNumCollection->GetEntries();
  int SizeOfDenCollection = a2DDenCollection->GetEntries();

  //-----Rescaling normalization bins
  aMinNormBin /= aRebinFactor;
  aMaxNormBin /= aRebinFactor;

  if( (SizeOfNumCollection != 5) && (SizeOfDenCollection != 5) ) {cout << "In BuildSepCfs, the input collection sizes DO NOT EQUAL 5!!!!!" << endl;}
  if( SizeOfNumCollection != SizeOfDenCollection ) {cout << "In BuildSepCfs, the NumCollection and DenCollection DO NOT HAVE EQUAL SIZES!!!!!!!" << endl;}

  for(int aBin=1; aBin<=aNumberOfXbins; aBin++)
  {
    TObjArray* CfCollection = new TObjArray();
    TObjArray* NumCollection = new TObjArray();

    for(int i=0; i<SizeOfNumCollection; i++)
    {

      //------Add a file tag--------
      TString aNumName = ((TH2F*)a2DNumCollection->At(i))->GetName();
      TString aHistoName = aContainedHistosBaseName;
        aHistoName += "_bin";
        aHistoName += aBin;

      if (aNumName.Contains("Bp1")) {aHistoName+="_Bp1";}
      else if (aNumName.Contains("Bp2")) {aHistoName+="_Bp2";}
      else if (aNumName.Contains("Bm1")) {aHistoName+="_Bm1";}
      else if (aNumName.Contains("Bm2")) {aHistoName+="_Bm2";}
      else if (aNumName.Contains("Bm3")) {aHistoName+="_Bm3";}

      TH1D* Num = ((TH2F*)a2DNumCollection->At(i))->ProjectionY("",aBin,aBin);
        TH1F* NumClone = Num->Clone();  //Projections are always TH1D for some reason.  My functions work with TH1F, not sure how they will react to TH1D, but worth checking
				        //At some point I should go back and do dynamic_cast where necessary
      TH1D* Den = ((TH2F*)a2DDenCollection->At(i))->ProjectionY("",aBin,aBin);
        TH1F* DenClone = Den->Clone();

      //------Rebinning
      NumClone->Rebin(aRebinFactor);
      DenClone->Rebin(aRebinFactor);
      //-----

      CfCollection->Add(buildCF(aHistoName,aHistoName,NumClone,DenClone,aMinNormBin,aMaxNormBin));
      NumCollection->Add(NumClone);

    }
    TString CfName = aContainedHistosBaseName+"_bin";
    CfName+=aBin;
    TH1F* CfForSpecificBin = CombineCFs(CfName,CfName,CfCollection,NumCollection,aMinNormBin, aMaxNormBin);

    ReturnCollection->Add(CfForSpecificBin);

  }

  cout << "Final size of SepCfCollection: " << ReturnCollection->GetEntries() << endl;
  return ReturnCollection;

}


//________________________________________________________________________________________________________________
void buildAllcLamcKch3::BuildSepCollections(int aRebinFactor)
{
  cout << "_________________________Beginning BuildSepCollections()_________________________" << endl;

  //-----LamKchP
  TString NumName_SepTrackPos_LamKchP = "NumTrackPosSepCFs_LamKchP";
  TString DenName_SepTrackPos_LamKchP = "DenTrackPosSepCFs_LamKchP";
  TString NumName_SepTrackNeg_LamKchP = "NumTrackNegSepCFs_LamKchP";
  TString DenName_SepTrackNeg_LamKchP = "DenTrackNegSepCFs_LamKchP";

  //-----LamKchM
  TString NumName_SepTrackPos_LamKchM = "NumTrackPosSepCFs_LamKchM";
  TString DenName_SepTrackPos_LamKchM = "DenTrackPosSepCFs_LamKchM";
  TString NumName_SepTrackNeg_LamKchM = "NumTrackNegSepCFs_LamKchM";
  TString DenName_SepTrackNeg_LamKchM = "DenTrackNegSepCFs_LamKchM";

  //-----ALamKchP
  TString NumName_SepTrackPos_ALamKchP = "NumTrackPosSepCFs_ALamKchP";
  TString DenName_SepTrackPos_ALamKchP = "DenTrackPosSepCFs_ALamKchP";
  TString NumName_SepTrackNeg_ALamKchP = "NumTrackNegSepCFs_ALamKchP";
  TString DenName_SepTrackNeg_ALamKchP = "DenTrackNegSepCFs_ALamKchP";

  //-----ALamKchM
  TString NumName_SepTrackPos_ALamKchM = "NumTrackPosSepCFs_ALamKchM";
  TString DenName_SepTrackPos_ALamKchM = "DenTrackPosSepCFs_ALamKchM";
  TString NumName_SepTrackNeg_ALamKchM = "NumTrackNegSepCFs_ALamKchM";
  TString DenName_SepTrackNeg_ALamKchM = "DenTrackNegSepCFs_ALamKchM";


  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamKchP.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamKchP.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamKchP.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  NumName_SepTrackPos_LamKchP += CentralityTag;
  DenName_SepTrackPos_LamKchP += CentralityTag;
  NumName_SepTrackNeg_LamKchP += CentralityTag;
  DenName_SepTrackNeg_LamKchP += CentralityTag;

  NumName_SepTrackPos_LamKchM += CentralityTag;
  DenName_SepTrackPos_LamKchM += CentralityTag;
  NumName_SepTrackNeg_LamKchM += CentralityTag;
  DenName_SepTrackNeg_LamKchM += CentralityTag;


  NumName_SepTrackPos_ALamKchP += CentralityTag;
  DenName_SepTrackPos_ALamKchP += CentralityTag;
  NumName_SepTrackNeg_ALamKchP += CentralityTag;
  DenName_SepTrackNeg_ALamKchP += CentralityTag;

  NumName_SepTrackPos_ALamKchM += CentralityTag;
  DenName_SepTrackPos_ALamKchM += CentralityTag;
  NumName_SepTrackNeg_ALamKchM += CentralityTag;
  DenName_SepTrackNeg_ALamKchM += CentralityTag;

  int NumXbins = 8;

  //----------------------------------------------------------------------

  //-----For now, simply set fMinNormBinSepCF = fMinNormBinAvgSepCF && fMaxNormBinSepCF = fMaxNormBinAvgSepCF
  cout << "fMinNormBinSepCF (default constructor) = " << fMinNormBinSepCF << endl;
  cout << "fMaxNormBinSepCF (default constructor) = " << fMaxNormBinSepCF << endl;
  fMinNormBinSepCF = fMinNormBinAvgSepCF;
  fMaxNormBinSepCF = fMaxNormBinAvgSepCF;
  cout << "fMinNormBinSepCF (set) = " << fMinNormBinSepCF << endl;
  cout << "fMaxNormBinSepCF (set) = " << fMaxNormBinSepCF << endl << endl;

  //--LamKchP
  f2DSepNumCollection_TrackPos_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,NumName_SepTrackPos_LamKchP);
  f2DSepDenCollection_TrackPos_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,DenName_SepTrackPos_LamKchP);
  f1DSepCfCollection_TrackPos_LamKchP = BuildSepCfs(aRebinFactor,"f1DSepCf_TrackPos_LamKchP",NumXbins,f2DSepNumCollection_TrackPos_LamKchP,f2DSepDenCollection_TrackPos_LamKchP,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_TrackNeg_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,NumName_SepTrackNeg_LamKchP);
  f2DSepDenCollection_TrackNeg_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,DenName_SepTrackNeg_LamKchP);
  f1DSepCfCollection_TrackNeg_LamKchP = BuildSepCfs(aRebinFactor,"f1DSepCf_TrackNeg_LamKchP",NumXbins,f2DSepNumCollection_TrackNeg_LamKchP,f2DSepDenCollection_TrackNeg_LamKchP,fMinNormBinSepCF,fMaxNormBinSepCF);

  //--LamKchM
  f2DSepNumCollection_TrackPos_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,NumName_SepTrackPos_LamKchM);
  f2DSepDenCollection_TrackPos_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,DenName_SepTrackPos_LamKchM);
  f1DSepCfCollection_TrackPos_LamKchM = BuildSepCfs(aRebinFactor,"f1DSepCf_TrackPos_LamKchM",NumXbins,f2DSepNumCollection_TrackPos_LamKchM,f2DSepDenCollection_TrackPos_LamKchM,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_TrackNeg_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,NumName_SepTrackNeg_LamKchM);
  f2DSepDenCollection_TrackNeg_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,DenName_SepTrackNeg_LamKchM);
  f1DSepCfCollection_TrackNeg_LamKchM = BuildSepCfs(aRebinFactor,"f1DSepCf_TrackNeg_LamKchM",NumXbins,f2DSepNumCollection_TrackNeg_LamKchM,f2DSepDenCollection_TrackNeg_LamKchM,fMinNormBinSepCF,fMaxNormBinSepCF);

  //--ALam-KchP
  f2DSepNumCollection_TrackPos_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,NumName_SepTrackPos_ALamKchP);
  f2DSepDenCollection_TrackPos_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,DenName_SepTrackPos_ALamKchP);
  f1DSepCfCollection_TrackPos_ALamKchP = BuildSepCfs(aRebinFactor,"f1DSepCf_TrackPos_ALamKchP",NumXbins,f2DSepNumCollection_TrackPos_ALamKchP,f2DSepDenCollection_TrackPos_ALamKchP,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_TrackNeg_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,NumName_SepTrackNeg_ALamKchP);
  f2DSepDenCollection_TrackNeg_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,DenName_SepTrackNeg_ALamKchP);
  f1DSepCfCollection_TrackNeg_ALamKchP = BuildSepCfs(aRebinFactor,"f1DSepCf_TrackNeg_ALamKchP",NumXbins,f2DSepNumCollection_TrackNeg_ALamKchP,f2DSepDenCollection_TrackNeg_ALamKchP,fMinNormBinSepCF,fMaxNormBinSepCF);

  //--ALam-KchM
  f2DSepNumCollection_TrackPos_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,NumName_SepTrackPos_ALamKchM);
  f2DSepDenCollection_TrackPos_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,DenName_SepTrackPos_ALamKchM);
  f1DSepCfCollection_TrackPos_ALamKchM = BuildSepCfs(aRebinFactor,"f1DSepCf_TrackPos_ALamKchM",NumXbins,f2DSepNumCollection_TrackPos_ALamKchM,f2DSepDenCollection_TrackPos_ALamKchM,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_TrackNeg_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,NumName_SepTrackNeg_ALamKchM);
  f2DSepDenCollection_TrackNeg_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,DenName_SepTrackNeg_ALamKchM);
  f1DSepCfCollection_TrackNeg_ALamKchM = BuildSepCfs(aRebinFactor,"f1DSepCf_TrackNeg_ALamKchM",NumXbins,f2DSepNumCollection_TrackNeg_ALamKchM,f2DSepDenCollection_TrackNeg_ALamKchM,fMinNormBinSepCF,fMaxNormBinSepCF);


  cout << "_________________________Done BuildSepCollections()_________________________" << endl;
}


//________________________________________________________________________________________________________________
void buildAllcLamcKch3::DrawFinalSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TObjArray *a1DSepCfCollection_TrackPos_LamKchP = (TObjArray*)f1DSepCfCollection_TrackPos_LamKchP->Clone();
  TObjArray *a1DSepCfCollection_TrackNeg_LamKchP = (TObjArray*)f1DSepCfCollection_TrackNeg_LamKchP->Clone();

  TObjArray *a1DSepCfCollection_TrackPos_LamKchM = (TObjArray*)f1DSepCfCollection_TrackPos_LamKchM->Clone();
  TObjArray *a1DSepCfCollection_TrackNeg_LamKchM = (TObjArray*)f1DSepCfCollection_TrackNeg_LamKchM->Clone();

  TObjArray *a1DSepCfCollection_TrackPos_ALamKchP = (TObjArray*)f1DSepCfCollection_TrackPos_ALamKchP->Clone();
  TObjArray *a1DSepCfCollection_TrackNeg_ALamKchP = (TObjArray*)f1DSepCfCollection_TrackNeg_ALamKchP->Clone();

  TObjArray *a1DSepCfCollection_TrackPos_ALamKchM = (TObjArray*)f1DSepCfCollection_TrackPos_ALamKchM->Clone();
  TObjArray *a1DSepCfCollection_TrackNeg_ALamKchM = (TObjArray*)f1DSepCfCollection_TrackNeg_ALamKchM->Clone();


  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  double YRangeMin = 0.8;
  double YRangeMax = 2.0;

  int NumXbins = 8;

  //------------------------------------------------------------------
  //-------LamKchP------------
  aCanvasLamKchP->cd();
  aCanvasLamKchP->Divide(8,2,0,0);

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1;
    aCanvasLamKchP->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_TrackPos_LamKchP->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("p(#Lambda) - K+");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1+8;
    aCanvasLamKchP->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_TrackNeg_LamKchP->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#pi^{-}(#Lambda) - K+");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  //-------LamKchM------------
  aCanvasLamKchM->cd();
  aCanvasLamKchM->Divide(8,2,0,0);

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1;
    aCanvasLamKchM->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_TrackPos_LamKchM->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("p(#Lambda) - K-");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1+8;
    aCanvasLamKchM->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_TrackNeg_LamKchM->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#pi^{-}(#Lambda) - K-");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  //-------ALamKchP------------
  aCanvasALamKchP->cd();
  aCanvasALamKchP->Divide(8,2,0,0);

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1;
    aCanvasALamKchP->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_TrackPos_ALamKchP->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#pi^{+}(#bar{#Lambda}) - K+");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1+8;
    aCanvasALamKchP->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_TrackNeg_ALamKchP->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#bar{p}(#bar{#Lambda}) - K+");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  //-------ALamKchM------------
  aCanvasALamKchM->cd();
  aCanvasALamKchM->Divide(8,2,0,0);

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1;
    aCanvasALamKchM->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_TrackPos_ALamKchM->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#pi^{+}(#bar{#Lambda}) - K-");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1+8;
    aCanvasALamKchM->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_TrackNeg_ALamKchM->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#bar{p}(#bar{#Lambda}) - K-");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

}


//________________________________________________________________________________________________________________
void buildAllcLamcKch3::DrawAvgOfFinalSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TObjArray *a1DSepCfCollection_TrackPos_LamKchP = (TObjArray*)f1DSepCfCollection_TrackPos_LamKchP->Clone();
  TObjArray *a1DSepCfCollection_TrackNeg_LamKchP = (TObjArray*)f1DSepCfCollection_TrackNeg_LamKchP->Clone();

  TObjArray *a1DSepCfCollection_TrackPos_LamKchM = (TObjArray*)f1DSepCfCollection_TrackPos_LamKchM->Clone();
  TObjArray *a1DSepCfCollection_TrackNeg_LamKchM = (TObjArray*)f1DSepCfCollection_TrackNeg_LamKchM->Clone();

  TObjArray *a1DSepCfCollection_TrackPos_ALamKchP = (TObjArray*)f1DSepCfCollection_TrackPos_ALamKchP->Clone();
  TObjArray *a1DSepCfCollection_TrackNeg_ALamKchP = (TObjArray*)f1DSepCfCollection_TrackNeg_ALamKchP->Clone();

  TObjArray *a1DSepCfCollection_TrackPos_ALamKchM = (TObjArray*)f1DSepCfCollection_TrackPos_ALamKchM->Clone();
  TObjArray *a1DSepCfCollection_TrackNeg_ALamKchM = (TObjArray*)f1DSepCfCollection_TrackNeg_ALamKchM->Clone();

  //---------------------------------------------------------------------------------------------------------
  TH1F* aAvgSepCf_TrackPos_LamKchP = (TH1F*)a1DSepCfCollection_TrackPos_LamKchP->At(0);
    aAvgSepCf_TrackPos_LamKchP->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_TrackNeg_LamKchP = (TH1F*)a1DSepCfCollection_TrackNeg_LamKchP->At(0);
    aAvgSepCf_TrackNeg_LamKchP->SetBit(TH1::kIsAverage);

  TH1F* aAvgSepCf_TrackPos_LamKchM = (TH1F*)a1DSepCfCollection_TrackPos_LamKchM->At(0);
    aAvgSepCf_TrackPos_LamKchM->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_TrackNeg_LamKchM = (TH1F*)a1DSepCfCollection_TrackNeg_LamKchM->At(0);
    aAvgSepCf_TrackNeg_LamKchM->SetBit(TH1::kIsAverage);

  TH1F* aAvgSepCf_TrackPos_ALamKchP = (TH1F*)a1DSepCfCollection_TrackPos_ALamKchP->At(0);
    aAvgSepCf_TrackPos_ALamKchP->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_TrackNeg_ALamKchP = (TH1F*)a1DSepCfCollection_TrackNeg_ALamKchP->At(0);
    aAvgSepCf_TrackNeg_ALamKchP->SetBit(TH1::kIsAverage);

  TH1F* aAvgSepCf_TrackPos_ALamKchM = (TH1F*)a1DSepCfCollection_TrackPos_ALamKchM->At(0);
    aAvgSepCf_TrackPos_ALamKchM->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_TrackNeg_ALamKchM = (TH1F*)a1DSepCfCollection_TrackNeg_ALamKchM->At(0);
    aAvgSepCf_TrackNeg_ALamKchM->SetBit(TH1::kIsAverage);


  //---------------------------------------------------------------------------------------------------------
  int NumXbins = 8;
  for(int i=1; i<NumXbins; i++)
  {
    TH1F* tempSepCf_TrackPos_LamKchP = ((TH1F*)a1DSepCfCollection_TrackPos_LamKchP->At(i))->Clone();
      tempSepCf_TrackPos_LamKchP->SetBit(TH1::kIsAverage);
      aAvgSepCf_TrackPos_LamKchP->Add(tempSepCf_TrackPos_LamKchP);
    TH1F* tempSepCf_TrackNeg_LamKchP = ((TH1F*)a1DSepCfCollection_TrackNeg_LamKchP->At(i))->Clone();
      tempSepCf_TrackNeg_LamKchP->SetBit(TH1::kIsAverage);
      aAvgSepCf_TrackNeg_LamKchP->Add(tempSepCf_TrackNeg_LamKchP);

    TH1F* tempSepCf_TrackPos_LamKchM = ((TH1F*)a1DSepCfCollection_TrackPos_LamKchM->At(i))->Clone();
      tempSepCf_TrackPos_LamKchM->SetBit(TH1::kIsAverage);
      aAvgSepCf_TrackPos_LamKchM->Add(tempSepCf_TrackPos_LamKchM);
    TH1F* tempSepCf_TrackNeg_LamKchM = ((TH1F*)a1DSepCfCollection_TrackNeg_LamKchM->At(i))->Clone();
      tempSepCf_TrackNeg_LamKchM->SetBit(TH1::kIsAverage);
      aAvgSepCf_TrackNeg_LamKchM->Add(tempSepCf_TrackNeg_LamKchM);

    TH1F* tempSepCf_TrackPos_ALamKchP = ((TH1F*)a1DSepCfCollection_TrackPos_ALamKchP->At(i))->Clone();
      tempSepCf_TrackPos_ALamKchP->SetBit(TH1::kIsAverage);
      aAvgSepCf_TrackPos_ALamKchP->Add(tempSepCf_TrackPos_ALamKchP);
    TH1F* tempSepCf_TrackNeg_ALamKchP = ((TH1F*)a1DSepCfCollection_TrackNeg_ALamKchP->At(i))->Clone();
      tempSepCf_TrackNeg_ALamKchP->SetBit(TH1::kIsAverage);
      aAvgSepCf_TrackNeg_ALamKchP->Add(tempSepCf_TrackNeg_ALamKchP);

    TH1F* tempSepCf_TrackPos_ALamKchM = ((TH1F*)a1DSepCfCollection_TrackPos_ALamKchM->At(i))->Clone();
      tempSepCf_TrackPos_ALamKchM->SetBit(TH1::kIsAverage);
      aAvgSepCf_TrackPos_ALamKchM->Add(tempSepCf_TrackPos_ALamKchM);
    TH1F* tempSepCf_TrackNeg_ALamKchM = ((TH1F*)a1DSepCfCollection_TrackNeg_ALamKchM->At(i))->Clone();
      tempSepCf_TrackNeg_ALamKchM->SetBit(TH1::kIsAverage);
      aAvgSepCf_TrackNeg_ALamKchM->Add(tempSepCf_TrackNeg_ALamKchM);

  }


  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  double YRangeMin = 0.8;
  double YRangeMax = 1.2;

  //------------------------------------------------------------------
  //-------LamKchP------------
  aCanvasLamKchP->cd();
  aCanvasLamKchP->Divide(1,2);

  aCanvasLamKchP->cd(1);
  aAvgSepCf_TrackPos_LamKchP->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackPos_LamKchP->SetTitle("p(#Lambda) - K+");
  aAvgSepCf_TrackPos_LamKchP->Draw();
  line->Draw();

  aCanvasLamKchP->cd(2);
  aAvgSepCf_TrackNeg_LamKchP->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackNeg_LamKchP->SetTitle("#pi^{-}(#Lambda) - K+");
  aAvgSepCf_TrackNeg_LamKchP->Draw();
  line->Draw();

  //-------LamKchM------------
  aCanvasLamKchM->cd();
  aCanvasLamKchM->Divide(1,2);

  aCanvasLamKchM->cd(1);
  aAvgSepCf_TrackPos_LamKchM->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackPos_LamKchM->SetTitle("p(#Lambda) - K-");
  aAvgSepCf_TrackPos_LamKchM->Draw();
  line->Draw();

  aCanvasLamKchM->cd(2);
  aAvgSepCf_TrackNeg_LamKchM->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackNeg_LamKchM->SetTitle("#pi^{-}(#Lambda) - K-");
  aAvgSepCf_TrackNeg_LamKchM->Draw();
  line->Draw();

  //-------ALamKchP------------
  aCanvasALamKchP->cd();
  aCanvasALamKchP->Divide(1,2);

  aCanvasALamKchP->cd(1);
  aAvgSepCf_TrackPos_ALamKchP->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackPos_ALamKchP->SetTitle("#pi^{+}(#bar{#Lambda}) - K+");
  aAvgSepCf_TrackPos_ALamKchP->Draw();
  line->Draw();

  aCanvasALamKchP->cd(2);
  aAvgSepCf_TrackNeg_ALamKchP->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackNeg_ALamKchP->SetTitle("#bar{p}(#bar{#Lambda}) - K+");
  aAvgSepCf_TrackNeg_ALamKchP->Draw();
  line->Draw();

  //-------ALamKchM------------
  aCanvasALamKchM->cd();
  aCanvasALamKchM->Divide(1,2);

  aCanvasALamKchM->cd(1);
  aAvgSepCf_TrackPos_ALamKchM->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackPos_ALamKchM->SetTitle("#pi^{+}(#bar{#Lambda}) - K-");
  aAvgSepCf_TrackPos_ALamKchM->Draw();
  line->Draw();

  aCanvasALamKchM->cd(2);
  aAvgSepCf_TrackNeg_ALamKchM->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackNeg_ALamKchM->SetTitle("#bar{p}(#bar{#Lambda}) - K-");
  aAvgSepCf_TrackNeg_ALamKchM->Draw();
  line->Draw();


}



//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
