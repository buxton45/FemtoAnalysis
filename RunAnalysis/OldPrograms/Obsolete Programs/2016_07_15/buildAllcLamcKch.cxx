///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamcKch                                                        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "buildAllcLamcKch.h"

#ifdef __ROOT__
ClassImp(buildAllcLamcKch)
#endif

//________________________________________________________________________________________________________________
buildAllcLamcKch::buildAllcLamcKch(vector<TString> &aVectorOfFileNames, TString aDirNameLamKchP, TString aDirNameLamKchM, TString aDirNameALamKchP, TString aDirNameALamKchM):
  buildAll(aVectorOfFileNames),

  //General stuff----------------
  fDirNameLamKchP(aDirNameLamKchP),
  fDirNameLamKchM(aDirNameLamKchM),
  fDirNameALamKchP(aDirNameALamKchP),
  fDirNameALamKchM(aDirNameALamKchM)

{
  SetAnalysisDirectories();
}


//________________________________________________________________________________________________________________
buildAllcLamcKch::~buildAllcLamcKch()
{
  cout << "Object is being deleted" << endl;
}


//________________________________________________________________________________________________________________
void buildAllcLamcKch::SetAnalysisDirectories()
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
TObjArray* buildAllcLamcKch::GetAnalysisDirectory(TString aFile, TString aDirectoryName)
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
TObjArray* buildAllcLamcKch::LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName)
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
void buildAllcLamcKch::BuildCFCollections()
{
  cout << "_________________________Beginning BuildCFCollections()_________________________" << endl;

  TString NumName_LamKchP = "NumKStarCf_LamKchP";
  TString DenName_LamKchP = "DenKStarCf_LamKchP";

  TString NumName_LamKchM = "NumKStarCf_LamKchM";
  TString DenName_LamKchM = "DenKStarCf_LamKchM";

  TString NumName_ALamKchP = "NumKStarCf_ALamKchP";
  TString DenName_ALamKchP = "DenKStarCf_ALamKchP";

  TString NumName_ALamKchM = "NumKStarCf_ALamKchM";
  TString DenName_ALamKchM = "DenKStarCf_ALamKchM";

  cout << "fMinNormBinCF (default constructor) = " << fMinNormBinCF << endl;
  cout << "fMaxNormBinCF (default constructor) = " << fMaxNormBinCF << endl;
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
void buildAllcLamcKch::DrawFinalCFs(TCanvas *aCanvas)
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
void buildAllcLamcKch::BuildAvgSepCollections()
{
  cout << "_________________________Beginning BuildAvgSepCollections()_________________________" << endl;

  TString NumName_AvgSepTrackPos_LamKchP = "NumTrackPosAvgSepCf_LamKchP";
  TString DenName_AvgSepTrackPos_LamKchP = "DenTrackPosAvgSepCf_LamKchP";
  TString NumName_AvgSepTrackNeg_LamKchP = "NumTrackNegAvgSepCf_LamKchP";
  TString DenName_AvgSepTrackNeg_LamKchP = "DenTrackNegAvgSepCf_LamKchP";

  TString NumName_AvgSepTrackPos_LamKchM = "NumTrackPosAvgSepCf_LamKchM";
  TString DenName_AvgSepTrackPos_LamKchM = "DenTrackPosAvgSepCf_LamKchM";
  TString NumName_AvgSepTrackNeg_LamKchM = "NumTrackNegAvgSepCf_LamKchM";
  TString DenName_AvgSepTrackNeg_LamKchM = "DenTrackNegAvgSepCf_LamKchM";


  TString NumName_AvgSepTrackPos_ALamKchP = "NumTrackPosAvgSepCf_ALamKchP";
  TString DenName_AvgSepTrackPos_ALamKchP = "DenTrackPosAvgSepCf_ALamKchP";
  TString NumName_AvgSepTrackNeg_ALamKchP = "NumTrackNegAvgSepCf_ALamKchP";
  TString DenName_AvgSepTrackNeg_ALamKchP = "DenTrackNegAvgSepCf_ALamKchP";

  TString NumName_AvgSepTrackPos_ALamKchM = "NumTrackPosAvgSepCf_ALamKchM";
  TString DenName_AvgSepTrackPos_ALamKchM = "DenTrackPosAvgSepCf_ALamKchM";
  TString NumName_AvgSepTrackNeg_ALamKchM = "NumTrackNegAvgSepCf_ALamKchM";
  TString DenName_AvgSepTrackNeg_ALamKchM = "DenTrackNegAvgSepCf_ALamKchM";

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
void buildAllcLamcKch::DrawFinalAvgSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM)
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
void buildAllcLamcKch::BuildPurityCollections()
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
void buildAllcLamcKch::DrawFinalPurity(TCanvas *aCanvas)
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
TObjArray* buildAllcLamcKch::GetCfCollection(TString aType, TString aDirectoryName)
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
void buildAllcLamcKch::SaveAll(TFile* aFile)
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
TObjArray* buildAllcLamcKch::LoadCollectionOf2DHistograms(TString aDirectoryName, TString aHistoName)
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
void buildAllcLamcKch::BuildSepCollections(int aRebinFactor)
{
  cout << "_________________________Beginning BuildSepCollections()_________________________" << endl;

  //-----LamKchP
  TString NumName_SepTrackPos_LamKchP = "NumTrackPosSepCfs_LamKchP";
  TString DenName_SepTrackPos_LamKchP = "DenTrackPosSepCfs_LamKchP";
  TString NumName_SepTrackNeg_LamKchP = "NumTrackNegSepCfs_LamKchP";
  TString DenName_SepTrackNeg_LamKchP = "DenTrackNegSepCfs_LamKchP";

  //-----LamKchM
  TString NumName_SepTrackPos_LamKchM = "NumTrackPosSepCfs_LamKchM";
  TString DenName_SepTrackPos_LamKchM = "DenTrackPosSepCfs_LamKchM";
  TString NumName_SepTrackNeg_LamKchM = "NumTrackNegSepCfs_LamKchM";
  TString DenName_SepTrackNeg_LamKchM = "DenTrackNegSepCfs_LamKchM";

  //-----ALamKchP
  TString NumName_SepTrackPos_ALamKchP = "NumTrackPosSepCfs_ALamKchP";
  TString DenName_SepTrackPos_ALamKchP = "DenTrackPosSepCfs_ALamKchP";
  TString NumName_SepTrackNeg_ALamKchP = "NumTrackNegSepCfs_ALamKchP";
  TString DenName_SepTrackNeg_ALamKchP = "DenTrackNegSepCfs_ALamKchP";

  //-----ALamKchM
  TString NumName_SepTrackPos_ALamKchM = "NumTrackPosSepCfs_ALamKchM";
  TString DenName_SepTrackPos_ALamKchM = "DenTrackPosSepCfs_ALamKchM";
  TString NumName_SepTrackNeg_ALamKchM = "NumTrackNegSepCfs_ALamKchM";
  TString DenName_SepTrackNeg_ALamKchM = "DenTrackNegSepCfs_ALamKchM";


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


  cout << "_________________________Done BuildSepCollections()_________________________" << endl << endl << endl;
}


//________________________________________________________________________________________________________________
void buildAllcLamcKch::DrawFinalSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM)
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
void buildAllcLamcKch::DrawAvgOfFinalSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM)
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
void buildAllcLamcKch::BuildCowCollections(int aRebinFactor)
{
  cout << "_________________________Beginning BuildCowCollections()_________________________" << endl;

  //-----LamKchP
  TString NumName_CowTrackPos_LamKchP = "NumTrackPosAvgSepCfCowboysAndSailors_LamKchP";
  TString DenName_CowTrackPos_LamKchP = "DenTrackPosAvgSepCfCowboysAndSailors_LamKchP";
  TString NumName_CowTrackNeg_LamKchP = "NumTrackNegAvgSepCfCowboysAndSailors_LamKchP";
  TString DenName_CowTrackNeg_LamKchP = "DenTrackNegAvgSepCfCowboysAndSailors_LamKchP";

  //-----LamKchM
  TString NumName_CowTrackPos_LamKchM = "NumTrackPosAvgSepCfCowboysAndSailors_LamKchM";
  TString DenName_CowTrackPos_LamKchM = "DenTrackPosAvgSepCfCowboysAndSailors_LamKchM";
  TString NumName_CowTrackNeg_LamKchM = "NumTrackNegAvgSepCfCowboysAndSailors_LamKchM";
  TString DenName_CowTrackNeg_LamKchM = "DenTrackNegAvgSepCfCowboysAndSailors_LamKchM";

  //-----ALamKchP
  TString NumName_CowTrackPos_ALamKchP = "NumTrackPosAvgSepCfCowboysAndSailors_ALamKchP";
  TString DenName_CowTrackPos_ALamKchP = "DenTrackPosAvgSepCfCowboysAndSailors_ALamKchP";
  TString NumName_CowTrackNeg_ALamKchP = "NumTrackNegAvgSepCfCowboysAndSailors_ALamKchP";
  TString DenName_CowTrackNeg_ALamKchP = "DenTrackNegAvgSepCfCowboysAndSailors_ALamKchP";

  //-----ALamKchM
  TString NumName_CowTrackPos_ALamKchM = "NumTrackPosAvgSepCfCowboysAndSailors_ALamKchM";
  TString DenName_CowTrackPos_ALamKchM = "DenTrackPosAvgSepCfCowboysAndSailors_ALamKchM";
  TString NumName_CowTrackNeg_ALamKchM = "NumTrackNegAvgSepCfCowboysAndSailors_ALamKchM";
  TString DenName_CowTrackNeg_ALamKchM = "DenTrackNegAvgSepCfCowboysAndSailors_ALamKchM";



  //----------------------------------------------------------------------


  //--LamKchP
  f2DCowNumCollection_TrackPos_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,NumName_CowTrackPos_LamKchP);
  f2DCowDenCollection_TrackPos_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,DenName_CowTrackPos_LamKchP);
  f1DCowCfCollection_TrackPos_LamKchP = BuildCowCfs(aRebinFactor,"f1DCowCf_TrackPos_LamKchP",f2DCowNumCollection_TrackPos_LamKchP,f2DCowDenCollection_TrackPos_LamKchP,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_TrackNeg_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,NumName_CowTrackNeg_LamKchP);
  f2DCowDenCollection_TrackNeg_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,DenName_CowTrackNeg_LamKchP);
  f1DCowCfCollection_TrackNeg_LamKchP = BuildCowCfs(aRebinFactor,"f1DCowCf_TrackNeg_LamKchP",f2DCowNumCollection_TrackNeg_LamKchP,f2DCowDenCollection_TrackNeg_LamKchP,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  //--LamKchM
  f2DCowNumCollection_TrackPos_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,NumName_CowTrackPos_LamKchM);
  f2DCowDenCollection_TrackPos_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,DenName_CowTrackPos_LamKchM);
  f1DCowCfCollection_TrackPos_LamKchM = BuildCowCfs(aRebinFactor,"f1DCowCf_TrackPos_LamKchM",f2DCowNumCollection_TrackPos_LamKchM,f2DCowDenCollection_TrackPos_LamKchM,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_TrackNeg_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,NumName_CowTrackNeg_LamKchM);
  f2DCowDenCollection_TrackNeg_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,DenName_CowTrackNeg_LamKchM);
  f1DCowCfCollection_TrackNeg_LamKchM = BuildCowCfs(aRebinFactor,"f1DCowCf_TrackNeg_LamKchM",f2DCowNumCollection_TrackNeg_LamKchM,f2DCowDenCollection_TrackNeg_LamKchM,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  //--ALam-KchP
  f2DCowNumCollection_TrackPos_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,NumName_CowTrackPos_ALamKchP);
  f2DCowDenCollection_TrackPos_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,DenName_CowTrackPos_ALamKchP);
  f1DCowCfCollection_TrackPos_ALamKchP = BuildCowCfs(aRebinFactor,"f1DCowCf_TrackPos_ALamKchP",f2DCowNumCollection_TrackPos_ALamKchP,f2DCowDenCollection_TrackPos_ALamKchP,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_TrackNeg_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,NumName_CowTrackNeg_ALamKchP);
  f2DCowDenCollection_TrackNeg_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,DenName_CowTrackNeg_ALamKchP);
  f1DCowCfCollection_TrackNeg_ALamKchP = BuildCowCfs(aRebinFactor,"f1DCowCf_TrackNeg_ALamKchP",f2DCowNumCollection_TrackNeg_ALamKchP,f2DCowDenCollection_TrackNeg_ALamKchP,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  //--ALam-KchM
  f2DCowNumCollection_TrackPos_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,NumName_CowTrackPos_ALamKchM);
  f2DCowDenCollection_TrackPos_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,DenName_CowTrackPos_ALamKchM);
  f1DCowCfCollection_TrackPos_ALamKchM = BuildCowCfs(aRebinFactor,"f1DCowCf_TrackPos_ALamKchM",f2DCowNumCollection_TrackPos_ALamKchM,f2DCowDenCollection_TrackPos_ALamKchM,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_TrackNeg_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,NumName_CowTrackNeg_ALamKchM);
  f2DCowDenCollection_TrackNeg_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,DenName_CowTrackNeg_ALamKchM);
  f1DCowCfCollection_TrackNeg_ALamKchM = BuildCowCfs(aRebinFactor,"f1DCowCf_TrackNeg_ALamKchM",f2DCowNumCollection_TrackNeg_ALamKchM,f2DCowDenCollection_TrackNeg_ALamKchM,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);


  cout << "_________________________Done BuildCowCollections()_________________________" << endl << endl << endl;
}



//________________________________________________________________________________________________________________
void buildAllcLamcKch::DrawFinalCowCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TObjArray *a1DCowCfCollection_TrackPos_LamKchP = (TObjArray*)f1DCowCfCollection_TrackPos_LamKchP->Clone();
  TObjArray *a1DCowCfCollection_TrackNeg_LamKchP = (TObjArray*)f1DCowCfCollection_TrackNeg_LamKchP->Clone();

  TObjArray *a1DCowCfCollection_TrackPos_LamKchM = (TObjArray*)f1DCowCfCollection_TrackPos_LamKchM->Clone();
  TObjArray *a1DCowCfCollection_TrackNeg_LamKchM = (TObjArray*)f1DCowCfCollection_TrackNeg_LamKchM->Clone();

  TObjArray *a1DCowCfCollection_TrackPos_ALamKchP = (TObjArray*)f1DCowCfCollection_TrackPos_ALamKchP->Clone();
  TObjArray *a1DCowCfCollection_TrackNeg_ALamKchP = (TObjArray*)f1DCowCfCollection_TrackNeg_ALamKchP->Clone();

  TObjArray *a1DCowCfCollection_TrackPos_ALamKchM = (TObjArray*)f1DCowCfCollection_TrackPos_ALamKchM->Clone();
  TObjArray *a1DCowCfCollection_TrackNeg_ALamKchM = (TObjArray*)f1DCowCfCollection_TrackNeg_ALamKchM->Clone();


  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  double YRangeMin = -0.5;
  double YRangeMax = 5.0;

  double aMarkerSize = 0.75;
  double aMarkerStyle = 20;

  //------------------------------------------------------------------
  //-------LamKchP------------
  aCanvasLamKchP->cd();
  aCanvasLamKchP->Divide(1,2);

  aCanvasLamKchP->cd(1);
  TH1D* aAvgSepCf_TrackPos_LamKchP1 = (TH1D*)a1DCowCfCollection_TrackPos_LamKchP->At(0);
    aAvgSepCf_TrackPos_LamKchP1->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackPos_LamKchP1->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackPos_LamKchP1->SetMarkerColor(1);
  TH1D* aAvgSepCf_TrackPos_LamKchP2 = (TH1D*)a1DCowCfCollection_TrackPos_LamKchP->At(1);
    aAvgSepCf_TrackPos_LamKchP2->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackPos_LamKchP2->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackPos_LamKchP2->SetMarkerColor(2);
  aAvgSepCf_TrackPos_LamKchP1->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackPos_LamKchP1->SetTitle("p(#Lambda) - K+");
  aAvgSepCf_TrackPos_LamKchP1->Draw();
  aAvgSepCf_TrackPos_LamKchP2->Draw("same");
  line->Draw();
  TLegend *legLamKchP1 = new TLegend(0.7,0.7,0.9,0.9);
    legLamKchP1->AddEntry(aAvgSepCf_TrackPos_LamKchP1, "P(K+) > P(p(#Lambda))", "p");
    legLamKchP1->AddEntry(aAvgSepCf_TrackPos_LamKchP2, "P(K+) < P(p(#Lambda))", "p");
    legLamKchP1->Draw();

  aCanvasLamKchP->cd(2);
  TH1D* aAvgSepCf_TrackNeg_LamKchP1 = (TH1D*)a1DCowCfCollection_TrackNeg_LamKchP->At(0);
    aAvgSepCf_TrackNeg_LamKchP1->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackNeg_LamKchP1->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackNeg_LamKchP1->SetMarkerColor(1);
  TH1D* aAvgSepCf_TrackNeg_LamKchP2 = (TH1D*)a1DCowCfCollection_TrackNeg_LamKchP->At(1);
    aAvgSepCf_TrackNeg_LamKchP2->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackNeg_LamKchP2->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackNeg_LamKchP2->SetMarkerColor(2);
  aAvgSepCf_TrackNeg_LamKchP1->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackNeg_LamKchP1->SetTitle("#pi^{-}(#Lambda) - K+");
  aAvgSepCf_TrackNeg_LamKchP1->Draw();
  aAvgSepCf_TrackNeg_LamKchP2->Draw("same");
  line->Draw();
  TLegend *legLamKchP2 = new TLegend(0.7,0.7,0.9,0.9);
    legLamKchP2->AddEntry(aAvgSepCf_TrackNeg_LamKchP1, "P(K+) > P(#pi^{-}(#Lambda))", "p");
    legLamKchP2->AddEntry(aAvgSepCf_TrackNeg_LamKchP2, "P(K+) < P(#pi^{-}(#Lambda))", "p");
    legLamKchP2->Draw();

  //-------LamKchM------------
  aCanvasLamKchM->cd();
  aCanvasLamKchM->Divide(1,2);

  aCanvasLamKchM->cd(1);
  TH1D* aAvgSepCf_TrackPos_LamKchM1 = (TH1D*)a1DCowCfCollection_TrackPos_LamKchM->At(0);
    aAvgSepCf_TrackPos_LamKchM1->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackPos_LamKchM1->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackPos_LamKchM1->SetMarkerColor(1);
  TH1D* aAvgSepCf_TrackPos_LamKchM2 = (TH1D*)a1DCowCfCollection_TrackPos_LamKchM->At(1);
    aAvgSepCf_TrackPos_LamKchM2->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackPos_LamKchM2->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackPos_LamKchM2->SetMarkerColor(2);
  aAvgSepCf_TrackPos_LamKchM1->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackPos_LamKchM1->SetTitle("p(#Lambda) - K-");
  aAvgSepCf_TrackPos_LamKchM1->Draw();
  aAvgSepCf_TrackPos_LamKchM2->Draw("same");
  line->Draw();
  TLegend *legLamKchM1 = new TLegend(0.7,0.7,0.9,0.9);
    legLamKchM1->AddEntry(aAvgSepCf_TrackPos_LamKchM1, "P(K-) > P(p(#Lambda))", "p");
    legLamKchM1->AddEntry(aAvgSepCf_TrackPos_LamKchM2, "P(K-) < P(p(#Lambda))", "p");
    legLamKchM1->Draw();

  aCanvasLamKchM->cd(2);
  TH1D* aAvgSepCf_TrackNeg_LamKchM1 = (TH1D*)a1DCowCfCollection_TrackNeg_LamKchM->At(0);
    aAvgSepCf_TrackNeg_LamKchM1->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackNeg_LamKchM1->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackNeg_LamKchM1->SetMarkerColor(1);
  TH1D* aAvgSepCf_TrackNeg_LamKchM2 = (TH1D*)a1DCowCfCollection_TrackNeg_LamKchM->At(1);
    aAvgSepCf_TrackNeg_LamKchM2->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackNeg_LamKchM2->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackNeg_LamKchM2->SetMarkerColor(2);
  aAvgSepCf_TrackNeg_LamKchM1->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackNeg_LamKchM1->SetTitle("#pi^{-}(#Lambda) - K-");
  aAvgSepCf_TrackNeg_LamKchM1->Draw();
  aAvgSepCf_TrackNeg_LamKchM2->Draw("same");
  line->Draw();
  TLegend *legLamKchM2 = new TLegend(0.7,0.7,0.9,0.9);
    legLamKchM2->AddEntry(aAvgSepCf_TrackNeg_LamKchM1, "P(K-) > P(#pi^{-}(#Lambda))", "p");
    legLamKchM2->AddEntry(aAvgSepCf_TrackNeg_LamKchM2, "P(K-) < P(#pi^{-}(#Lambda))", "p");
    legLamKchM2->Draw();

  //-------ALamKchP------------
  aCanvasALamKchP->cd();
  aCanvasALamKchP->Divide(1,2);

  aCanvasALamKchP->cd(1);
  TH1D* aAvgSepCf_TrackPos_ALamKchP1 = (TH1D*)a1DCowCfCollection_TrackPos_ALamKchP->At(0);
    aAvgSepCf_TrackPos_ALamKchP1->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackPos_ALamKchP1->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackPos_ALamKchP1->SetMarkerColor(1);
  TH1D* aAvgSepCf_TrackPos_ALamKchP2 = (TH1D*)a1DCowCfCollection_TrackPos_ALamKchP->At(1);
    aAvgSepCf_TrackPos_ALamKchP2->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackPos_ALamKchP2->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackPos_ALamKchP2->SetMarkerColor(2);
  aAvgSepCf_TrackPos_ALamKchP1->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackPos_ALamKchP1->SetTitle("#pi^{+}(#bar{#Lambda}) - K+");
  aAvgSepCf_TrackPos_ALamKchP1->Draw();
  aAvgSepCf_TrackPos_ALamKchP2->Draw("same");
  line->Draw();
  TLegend *legALamKchP1 = new TLegend(0.7,0.7,0.9,0.9);
    legALamKchP1->AddEntry(aAvgSepCf_TrackPos_ALamKchP1, "P(K+) > P(#pi^{+}(#bar{#Lambda}))", "p");
    legALamKchP1->AddEntry(aAvgSepCf_TrackPos_ALamKchP2, "P(K+) < P(#pi^{+}(#bar{#Lambda}))", "p");
    legALamKchP1->Draw();

  aCanvasALamKchP->cd(2);
  TH1D* aAvgSepCf_TrackNeg_ALamKchP1 = (TH1D*)a1DCowCfCollection_TrackNeg_ALamKchP->At(0);
    aAvgSepCf_TrackNeg_ALamKchP1->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackNeg_ALamKchP1->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackNeg_ALamKchP1->SetMarkerColor(1);
  TH1D* aAvgSepCf_TrackNeg_ALamKchP2 = (TH1D*)a1DCowCfCollection_TrackNeg_ALamKchP->At(1);
    aAvgSepCf_TrackNeg_ALamKchP2->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackNeg_ALamKchP2->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackNeg_ALamKchP2->SetMarkerColor(2);
  aAvgSepCf_TrackNeg_ALamKchP1->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackNeg_ALamKchP1->SetTitle("#bar{p}(#bar{#Lambda}) - K+");
  aAvgSepCf_TrackNeg_ALamKchP1->Draw();
  aAvgSepCf_TrackNeg_ALamKchP2->Draw("same");
  line->Draw();
  TLegend *legALamKchP2 = new TLegend(0.7,0.7,0.9,0.9);
    legALamKchP2->AddEntry(aAvgSepCf_TrackNeg_ALamKchP1, "P(K+) > P(#bar{p}(#bar{#Lambda}))", "p");
    legALamKchP2->AddEntry(aAvgSepCf_TrackNeg_ALamKchP2, "P(K+) < P(#bar{p}(#bar{#Lambda}))", "p");
    legALamKchP2->Draw();

  //-------ALamKchM------------
  aCanvasALamKchM->cd();
  aCanvasALamKchM->Divide(1,2);

  aCanvasALamKchM->cd(1);
  TH1D* aAvgSepCf_TrackPos_ALamKchM1 = (TH1D*)a1DCowCfCollection_TrackPos_ALamKchM->At(0);
    aAvgSepCf_TrackPos_ALamKchM1->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackPos_ALamKchM1->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackPos_ALamKchM1->SetMarkerColor(1);
  TH1D* aAvgSepCf_TrackPos_ALamKchM2 = (TH1D*)a1DCowCfCollection_TrackPos_ALamKchM->At(1);
    aAvgSepCf_TrackPos_ALamKchM2->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackPos_ALamKchM2->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackPos_ALamKchM2->SetMarkerColor(2);
  aAvgSepCf_TrackPos_ALamKchM1->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackPos_ALamKchM1->SetTitle("#pi^{+}(#bar{#Lambda}) - K-");
  aAvgSepCf_TrackPos_ALamKchM1->Draw();
  aAvgSepCf_TrackPos_ALamKchM2->Draw("same");
  line->Draw();
  TLegend *legALamKchM1 = new TLegend(0.7,0.7,0.9,0.9);
    legALamKchM1->AddEntry(aAvgSepCf_TrackPos_ALamKchM1, "P(K-) > P(#pi^{+}(#bar{#Lambda}))", "p");
    legALamKchM1->AddEntry(aAvgSepCf_TrackPos_ALamKchM2, "P(K-) < P(#pi^{+}(#bar{#Lambda}))", "p");
    legALamKchM1->Draw();

  aCanvasALamKchM->cd(2);
  TH1D* aAvgSepCf_TrackNeg_ALamKchM1 = (TH1D*)a1DCowCfCollection_TrackNeg_ALamKchM->At(0);
    aAvgSepCf_TrackNeg_ALamKchM1->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackNeg_ALamKchM1->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackNeg_ALamKchM1->SetMarkerColor(1);
  TH1D* aAvgSepCf_TrackNeg_ALamKchM2 = (TH1D*)a1DCowCfCollection_TrackNeg_ALamKchM->At(1);
    aAvgSepCf_TrackNeg_ALamKchM2->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_TrackNeg_ALamKchM2->SetMarkerSize(aMarkerSize);
    aAvgSepCf_TrackNeg_ALamKchM2->SetMarkerColor(2);
  aAvgSepCf_TrackNeg_ALamKchM1->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_TrackNeg_ALamKchM1->SetTitle("#bar{p}(#bar{#Lambda}) - K-");
  aAvgSepCf_TrackNeg_ALamKchM1->Draw();
  aAvgSepCf_TrackNeg_ALamKchM2->Draw("same");
  line->Draw();
  TLegend *legALamKchM2 = new TLegend(0.7,0.7,0.9,0.9);
    legALamKchM2->AddEntry(aAvgSepCf_TrackNeg_ALamKchM1, "P(K-) > P(#bar{p}(#bar{#Lambda}))", "p");
    legALamKchM2->AddEntry(aAvgSepCf_TrackNeg_ALamKchM2, "P(K-) < P(#bar{p}(#bar{#Lambda}))", "p");
    legALamKchM2->Draw();

}


//________________________________________________________________________________________________________________
double buildAllcLamcKch::GetMCKchPurity(AnalysisType aAnalysisType, bool aBeforePairCut)
{
  TObjArray* tDirBp1 = new TObjArray();
  TObjArray* tDirBp2 = new TObjArray();
  TObjArray* tDirBm1 = new TObjArray();
  TObjArray* tDirBm2 = new TObjArray();
  TObjArray* tDirBm3 = new TObjArray();

  if(aAnalysisType == kLamKchP)
  {
    tDirBp1 = (TObjArray*)fDirLamKchPBp1->Clone();
    tDirBp2 = (TObjArray*)fDirLamKchPBp2->Clone();
    tDirBm1 = (TObjArray*)fDirLamKchPBm1->Clone();
    tDirBm2 = (TObjArray*)fDirLamKchPBm2->Clone();
    tDirBm3 = (TObjArray*)fDirLamKchPBm3->Clone();
  }

  else if(aAnalysisType == kALamKchP)
  {
    tDirBp1 = (TObjArray*)fDirALamKchPBp1->Clone();
    tDirBp2 = (TObjArray*)fDirALamKchPBp2->Clone();
    tDirBm1 = (TObjArray*)fDirALamKchPBm1->Clone();
    tDirBm2 = (TObjArray*)fDirALamKchPBm2->Clone();
    tDirBm3 = (TObjArray*)fDirALamKchPBm3->Clone();
  }

  else if(aAnalysisType == kLamKchM)
  {
    tDirBp1 = (TObjArray*)fDirLamKchMBp1->Clone();
    tDirBp2 = (TObjArray*)fDirLamKchMBp2->Clone();
    tDirBm1 = (TObjArray*)fDirLamKchMBm1->Clone();
    tDirBm2 = (TObjArray*)fDirLamKchMBm2->Clone();
    tDirBm3 = (TObjArray*)fDirLamKchMBm3->Clone();
  }

  else if(aAnalysisType == kALamKchM)
  {
    tDirBp1 = (TObjArray*)fDirALamKchMBp1->Clone();
    tDirBp2 = (TObjArray*)fDirALamKchMBp2->Clone();
    tDirBm1 = (TObjArray*)fDirALamKchMBm1->Clone();
    tDirBm2 = (TObjArray*)fDirALamKchMBm2->Clone();
    tDirBm3 = (TObjArray*)fDirALamKchMBm3->Clone();
  }

  //----------------------------------------------------------------
  TH1F* PId_Kch_BeforePairCut_Bp1 = new TH1F();
  TH1F* PId_Kch_AfterPairCut_Bp1 = new TH1F();

  TH1F* PId_Kch_BeforePairCut_Bp2 = new TH1F();
  TH1F* PId_Kch_AfterPairCut_Bp2 = new TH1F();

  TH1F* PId_Kch_BeforePairCut_Bm1 = new TH1F();
  TH1F* PId_Kch_AfterPairCut_Bm1 = new TH1F();

  TH1F* PId_Kch_BeforePairCut_Bm2 = new TH1F();
  TH1F* PId_Kch_AfterPairCut_Bm2 = new TH1F();

  TH1F* PId_Kch_BeforePairCut_Bm3 = new TH1F();
  TH1F* PId_Kch_AfterPairCut_Bm3 = new TH1F();

  if( (aAnalysisType == kLamKchP) || (aAnalysisType == kALamKchP) )
  {
    PId_Kch_BeforePairCut_Bp1 = GetHistoClone(tDirBp1, "PId_KchP_Pass", "PId_Kch_BeforePairCut_Bp1");
    PId_Kch_AfterPairCut_Bp1 = GetHistoClone(tDirBp1, "PIdSecondPass", "PId_Kch_AfterPairCut_Bp1");

    PId_Kch_BeforePairCut_Bp2 = GetHistoClone(tDirBp2, "PId_KchP_Pass", "PId_Kch_BeforePairCut_Bp2");
    PId_Kch_AfterPairCut_Bp2 = GetHistoClone(tDirBp2, "PIdSecondPass", "PId_Kch_AfterPairCut_Bp2");

    PId_Kch_BeforePairCut_Bm1 = GetHistoClone(tDirBm1, "PId_KchP_Pass", "PId_Kch_BeforePairCut_Bm1");
    PId_Kch_AfterPairCut_Bm1 = GetHistoClone(tDirBm1, "PIdSecondPass", "PId_Kch_AfterPairCut_Bm1");

    PId_Kch_BeforePairCut_Bm2 = GetHistoClone(tDirBm2, "PId_KchP_Pass", "PId_Kch_BeforePairCut_Bm2");
    PId_Kch_AfterPairCut_Bm2 = GetHistoClone(tDirBm2, "PIdSecondPass", "PId_Kch_AfterPairCut_Bm2");

    PId_Kch_BeforePairCut_Bm3 = GetHistoClone(tDirBm3, "PId_KchP_Pass", "PId_Kch_BeforePairCut_Bm3");
    PId_Kch_AfterPairCut_Bm3 = GetHistoClone(tDirBm3, "PIdSecondPass", "PId_Kch_AfterPairCut_Bm3");
  }

  else if( (aAnalysisType == kLamKchM) || (aAnalysisType == kALamKchM) )
  {
    PId_Kch_BeforePairCut_Bp1 = GetHistoClone(tDirBp1, "PId_KchM_Pass", "PId_Kch_BeforePairCut_Bp1");
    PId_Kch_AfterPairCut_Bp1 = GetHistoClone(tDirBp1, "PIdSecondPass", "PId_Kch_AfterPairCut_Bp1");

    PId_Kch_BeforePairCut_Bp2 = GetHistoClone(tDirBp2, "PId_KchM_Pass", "PId_Kch_BeforePairCut_Bp2");
    PId_Kch_AfterPairCut_Bp2 = GetHistoClone(tDirBp2, "PIdSecondPass", "PId_Kch_AfterPairCut_Bp2");

    PId_Kch_BeforePairCut_Bm1 = GetHistoClone(tDirBm1, "PId_KchM_Pass", "PId_Kch_BeforePairCut_Bm1");
    PId_Kch_AfterPairCut_Bm1 = GetHistoClone(tDirBm1, "PIdSecondPass", "PId_Kch_AfterPairCut_Bm1");

    PId_Kch_BeforePairCut_Bm2 = GetHistoClone(tDirBm2, "PId_KchM_Pass", "PId_Kch_BeforePairCut_Bm2");
    PId_Kch_AfterPairCut_Bm2 = GetHistoClone(tDirBm2, "PIdSecondPass", "PId_Kch_AfterPairCut_Bm2");

    PId_Kch_BeforePairCut_Bm3 = GetHistoClone(tDirBm3, "PId_KchM_Pass", "PId_Kch_BeforePairCut_Bm3");
    PId_Kch_AfterPairCut_Bm3 = GetHistoClone(tDirBm3, "PIdSecondPass", "PId_Kch_AfterPairCut_Bm3");
  }

  TH1D* PId_Kch_BeforePairCut = PId_Kch_BeforePairCut_Bp1->Clone("PId_Kch_BeforePairCut");
    PId_Kch_BeforePairCut->Add(PId_Kch_BeforePairCut_Bp2);
    PId_Kch_BeforePairCut->Add(PId_Kch_BeforePairCut_Bm1);
    PId_Kch_BeforePairCut->Add(PId_Kch_BeforePairCut_Bm2);
    PId_Kch_BeforePairCut->Add(PId_Kch_BeforePairCut_Bm3);

  TH1D* PId_Kch_AfterPairCut = PId_Kch_AfterPairCut_Bp1->Clone("PId_Kch_AfterPairCut");
    PId_Kch_AfterPairCut->Add(PId_Kch_AfterPairCut_Bp2);
    PId_Kch_AfterPairCut->Add(PId_Kch_AfterPairCut_Bm1);
    PId_Kch_AfterPairCut->Add(PId_Kch_AfterPairCut_Bm2);
    PId_Kch_AfterPairCut->Add(PId_Kch_AfterPairCut_Bm3);

  //----------------------------------------------------------------------------------
  tDirBp1->Delete();
  tDirBp2->Delete();
  tDirBm1->Delete();
  tDirBm2->Delete();
  tDirBm3->Delete();

  PId_Kch_BeforePairCut_Bp1->Delete();
  PId_Kch_BeforePairCut_Bp2->Delete();
  PId_Kch_BeforePairCut_Bm1->Delete();
  PId_Kch_BeforePairCut_Bm2->Delete();
  PId_Kch_BeforePairCut_Bm3->Delete();

  PId_Kch_AfterPairCut_Bp1->Delete();
  PId_Kch_AfterPairCut_Bp2->Delete();
  PId_Kch_AfterPairCut_Bm1->Delete();
  PId_Kch_AfterPairCut_Bm2->Delete();
  PId_Kch_AfterPairCut_Bm3->Delete();

  //----------------------------------------------------------------------------------
  double KchCounts_BeforePairCut = PId_Kch_BeforePairCut->GetBinContent(PId_Kch_BeforePairCut->FindBin(321));
  double TotalCounts_BeforePairCut = PId_Kch_BeforePairCut->GetEntries();
  double KchPurity_BeforePairCut = KchCounts_BeforePairCut/TotalCounts_BeforePairCut;

  double KchCounts_AfterPairCut = PId_Kch_AfterPairCut->GetBinContent(PId_Kch_AfterPairCut->FindBin(321));
  double TotalCounts_AfterPairCut = PId_Kch_AfterPairCut->GetEntries();
  double KchPurity_AfterPairCut = KchCounts_AfterPairCut/TotalCounts_AfterPairCut;


  if(aBeforePairCut) {return KchPurity_BeforePairCut;}
  else {return KchPurity_AfterPairCut;}
}


//________________________________________________________________________________________________________________
TObjArray* buildAllcLamcKch::DecomposeKStar2DCfs(TString aAnalysisTag, TObjArray *a2DNumCollection, TObjArray* a2DDenCollection)
{

  TObjArray *ReturnCollection = new TObjArray();

  assert(a2DNumCollection->GetEntries() == a2DDenCollection->GetEntries());

  TObjArray *tNumKStarCfPosCollection = new TObjArray();
  TObjArray *tDenKStarCfPosCollection = new TObjArray();
  TObjArray *tKStarCfPosCollection = new TObjArray();

  TObjArray *tNumKStarCfNegCollection = new TObjArray();
  TObjArray *tDenKStarCfNegCollection = new TObjArray();
  TObjArray *tKStarCfNegCollection = new TObjArray();

  for(unsigned int i=0; i<a2DNumCollection->GetEntries(); i++)
  {
    TString NumNamePos = "NumKStarCfPos";
    TString DenNamePos = "DenKStarCfPos";

    TString NumNameNeg = "NumKStarCfNeg";
    TString DenNameNeg = "DenKStarCfNeg";

    NumNamePos += i;
    DenNamePos += i;
    
    NumNameNeg += i;
    DenNameNeg += i;


    TH1D *tempNumNeg = ((TH2F*)a2DNumCollection->At(i))->ProjectionX(NumNamePos,1,1);
    TH1D *tempDenNeg = ((TH2F*)a2DDenCollection->At(i))->ProjectionX(DenNamePos,1,1);

    TH1D *tempNumPos = ((TH2F*)a2DNumCollection->At(i))->ProjectionX(NumNameNeg,2,2);
    TH1D *tempDenPos = ((TH2F*)a2DDenCollection->At(i))->ProjectionX(DenNameNeg,2,2);

    tNumKStarCfPosCollection->Add(tempNumPos);
    tDenKStarCfPosCollection->Add(tempDenPos);

    tNumKStarCfNegCollection->Add(tempNumNeg);
    tDenKStarCfNegCollection->Add(tempDenNeg);
  }

  for(unsigned int i=0; i<a2DNumCollection->GetEntries(); i++)
  {
    TString CfNamePos = "KStarCfPos";
    TString CfNameNeg = "KStarCfNeg";

    CfNamePos += i;
    CfNameNeg += i;

    TH1F *tempCfPos = buildCF(CfNamePos,CfNamePos,(TH1D*)tNumKStarCfPosCollection->At(i),(TH1D*)tDenKStarCfPosCollection->At(i),64,80);
    TH1F *tempCfNeg = buildCF(CfNameNeg,CfNameNeg,(TH1D*)tNumKStarCfNegCollection->At(i),(TH1D*)tDenKStarCfNegCollection->At(i),64,80);

    tKStarCfPosCollection->Add(tempCfPos);
    tKStarCfNegCollection->Add(tempCfNeg);
  }

  TString NameKStarCfPos = "KStarCfPos_";
  TString NameKStarCfNeg = "KStarCfNeg_";

  NameKStarCfPos += aAnalysisTag;
  NameKStarCfNeg += aAnalysisTag;

  TH1F *tKStarCfPos = CombineCFs(NameKStarCfPos,NameKStarCfPos,tKStarCfPosCollection,tNumKStarCfPosCollection,64,80);
  TH1F *tKStarCfNeg = CombineCFs(NameKStarCfNeg,NameKStarCfNeg,tKStarCfNegCollection,tNumKStarCfNegCollection,64,80);

  ReturnCollection->Add(tKStarCfPos);
  ReturnCollection->Add(tKStarCfNeg);


  return ReturnCollection;
}



//________________________________________________________________________________________________________________
void buildAllcLamcKch::BuildKStarCfsBinnedInKStarOut()
{
  cout << "_________________________Beginning BuildKStarCfsBinnedInKStarOut()_________________________" << endl;

  //-----LamKchP
  TString NumName_KStarCf2D_LamKchP = "NumKStarCf2D_LamKchP";
  TString DenName_KStarCf2D_LamKchP = "DenKStarCf2D_LamKchP";

  //-----LamKchM
  TString NumName_KStarCf2D_LamKchM = "NumKStarCf2D_LamKchM";
  TString DenName_KStarCf2D_LamKchM = "DenKStarCf2D_LamKchM";

  //-----ALamKchP
  TString NumName_KStarCf2D_ALamKchP = "NumKStarCf2D_ALamKchP";
  TString DenName_KStarCf2D_ALamKchP = "DenKStarCf2D_ALamKchP";

  //-----ALamKchM
  TString NumName_KStarCf2D_ALamKchM = "NumKStarCf2D_ALamKchM";
  TString DenName_KStarCf2D_ALamKchM = "DenKStarCf2D_ALamKchM";



  //----------------------------------------------------------------------

  TObjArray *tNumKStarCf2DCollection_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,NumName_KStarCf2D_LamKchP);
  TObjArray *tDenKStarCf2DCollection_LamKchP = LoadCollectionOf2DHistograms(fDirNameLamKchP,DenName_KStarCf2D_LamKchP);
  TObjArray *tKStarCfCollection_LamKchP = DecomposeKStar2DCfs("LamKchP",tNumKStarCf2DCollection_LamKchP,tDenKStarCf2DCollection_LamKchP);
    fKStarCfPosKStarOut_LamKchP = (TH1D*)tKStarCfCollection_LamKchP->At(0);
    fKStarCfNegKStarOut_LamKchP = (TH1D*)tKStarCfCollection_LamKchP->At(1);

    fKStarCfRatioPosNeg_LamKchP = (TH1D*)fKStarCfPosKStarOut_LamKchP->Clone("KStarCfRatioPosNeg_LamKchP");
    fKStarCfRatioPosNeg_LamKchP->Divide(fKStarCfNegKStarOut_LamKchP);

  TObjArray *tNumKStarCf2DCollection_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,NumName_KStarCf2D_LamKchM);
  TObjArray *tDenKStarCf2DCollection_LamKchM = LoadCollectionOf2DHistograms(fDirNameLamKchM,DenName_KStarCf2D_LamKchM);
  TObjArray *tKStarCfCollection_LamKchM = DecomposeKStar2DCfs("LamKchM",tNumKStarCf2DCollection_LamKchM,tDenKStarCf2DCollection_LamKchM);
    fKStarCfPosKStarOut_LamKchM = (TH1D*)tKStarCfCollection_LamKchM->At(0);
    fKStarCfNegKStarOut_LamKchM = (TH1D*)tKStarCfCollection_LamKchM->At(1);

    fKStarCfRatioPosNeg_LamKchM = (TH1D*)fKStarCfPosKStarOut_LamKchM->Clone("KStarCfRatioPosNeg_LamKchM");
    fKStarCfRatioPosNeg_LamKchM->Divide(fKStarCfNegKStarOut_LamKchM);

  TObjArray *tNumKStarCf2DCollection_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,NumName_KStarCf2D_ALamKchP);
  TObjArray *tDenKStarCf2DCollection_ALamKchP = LoadCollectionOf2DHistograms(fDirNameALamKchP,DenName_KStarCf2D_ALamKchP);
  TObjArray *tKStarCfCollection_ALamKchP = DecomposeKStar2DCfs("ALamKchP",tNumKStarCf2DCollection_ALamKchP,tDenKStarCf2DCollection_ALamKchP);
    fKStarCfPosKStarOut_ALamKchP = (TH1D*)tKStarCfCollection_ALamKchP->At(0);
    fKStarCfNegKStarOut_ALamKchP = (TH1D*)tKStarCfCollection_ALamKchP->At(1);

    fKStarCfRatioPosNeg_ALamKchP = (TH1D*)fKStarCfPosKStarOut_ALamKchP->Clone("KStarCfRatioPosNeg_ALamKchP");
    fKStarCfRatioPosNeg_ALamKchP->Divide(fKStarCfNegKStarOut_ALamKchP);

  TObjArray *tNumKStarCf2DCollection_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,NumName_KStarCf2D_ALamKchM);
  TObjArray *tDenKStarCf2DCollection_ALamKchM = LoadCollectionOf2DHistograms(fDirNameALamKchM,DenName_KStarCf2D_ALamKchM);
  TObjArray *tKStarCfCollection_ALamKchM = DecomposeKStar2DCfs("ALamKchM",tNumKStarCf2DCollection_ALamKchM,tDenKStarCf2DCollection_ALamKchM);
    fKStarCfPosKStarOut_ALamKchM = (TH1D*)tKStarCfCollection_ALamKchM->At(0);
    fKStarCfNegKStarOut_ALamKchM = (TH1D*)tKStarCfCollection_ALamKchM->At(1);

    fKStarCfRatioPosNeg_ALamKchM = (TH1D*)fKStarCfPosKStarOut_ALamKchM->Clone("KStarCfRatioPosNeg_ALamKchM");
    fKStarCfRatioPosNeg_ALamKchM->Divide(fKStarCfNegKStarOut_ALamKchM);

}


//________________________________________________________________________________________________________________
void buildAllcLamcKch::DrawKStarCfsBinnedInKStarOut(TCanvas *aCanvas)
{
  aCanvas->cd();
  aCanvas->Divide(2,2);

  aCanvas->cd(1);
  fKStarCfRatioPosNeg_LamKchP->SetMarkerStyle(20);
  fKStarCfRatioPosNeg_LamKchP->Draw();

  aCanvas->cd(2);
  fKStarCfRatioPosNeg_ALamKchM->SetMarkerStyle(20);
  fKStarCfRatioPosNeg_ALamKchM->Draw();

  aCanvas->cd(3);
  fKStarCfRatioPosNeg_LamKchM->SetMarkerStyle(20);
  fKStarCfRatioPosNeg_LamKchM->Draw();

  aCanvas->cd(4);
  fKStarCfRatioPosNeg_ALamKchP->SetMarkerStyle(20);
  fKStarCfRatioPosNeg_ALamKchP->Draw();

}





//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
