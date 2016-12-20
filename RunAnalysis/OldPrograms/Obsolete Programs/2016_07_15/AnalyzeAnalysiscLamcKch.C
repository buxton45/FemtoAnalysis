#include "buildAllcLamcKch.cxx"
#include <stdio.h>
#include <string.h>

char* AppendBaseDirectory(TString aBaseName, char* aAppend)
{
  char* ReturnStr = new char[80];
  strcpy(ReturnStr, gSystem->ExpandPathName((const char*)aBaseName));
  strcat(ReturnStr, aAppend);
  return ReturnStr;
}

void AnalyzeAnalysiscLamcKch()
{
  vector<TString> VectorOfFileNames;
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamcKch_AsRc_20150824/Results_cLamcKch_AsRc_20150824_Bp1.root");
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamcKch_AsRc_20150824/Results_cLamcKch_AsRc_20150824_Bp2.root");
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamcKch_AsRc_20150824/Results_cLamcKch_AsRc_20150824_Bm1.root");
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamcKch_AsRc_20150824/Results_cLamcKch_AsRc_20150824_Bm2.root");
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamcKch_AsRc_20150824/Results_cLamcKch_AsRc_20150824_Bm3.root");

  TString aSaveDirectory = "~/Analysis/K0Lam/Results_cLamcKch_AsRc_20150824/0010/";  //note:  This directory must already be created
  TString aSaveRootFileName = "Results_cLamcKch_AsRc_20150824_0010.root";
  TString aCentralityTag = "_0010";

  gSystem->mkdir(gSystem->ExpandPathName((const char*)aSaveDirectory), kTRUE);  //This will create the directory if not already created
										//If already created, this does nothing
				//I must call ExpandPathName because the name cannot contain any special shell characters (i.e. ~ or $)
				//and I must cast aSaveDirectory as const char* so ExpandPathName returns char*
				//instead of returning a Bool_t when used with a TString.
				//I want to keep aSaveDirectory as a TString so I can simply add strings to it (i.e. aSaveDirectory + "....")

  bool SavePDFs = false;
  bool SaveRootFile = false;

  //---------------------------------------------------------------------------------------------------------------------------

  buildAllcLamcKch *cLamcKchAnalysis = new buildAllcLamcKch(VectorOfFileNames,"LamKchP_0010","LamKchM_0010","ALamKchP_0010","ALamKchM_0010");
  cLamcKchAnalysis->BuildCFCollections();
  cLamcKchAnalysis->BuildAvgSepCollections();
  cLamcKchAnalysis->BuildPurityCollections();
  //cLamcKchAnalysis->BuildSepCollections(5);
  //cLamcKchAnalysis->BuildCowCollections(4);
  cLamcKchAnalysis->BuildKStarCfsBinnedInKStarOut();

  //-----KStarCfs
  TCanvas* canvasCfs = new TCanvas("canvasCfs","KStar Cfs",1400,500);
  cLamcKchAnalysis->DrawFinalCFs(canvasCfs);
  if(SavePDFs) 
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"Cfs"),kTRUE);
    canvasCfs->SaveAs(aSaveDirectory+"Cfs/cLamcKch_Cfs"+aCentralityTag+".pdf");
  }

  //-----AvgSepCfs
  TCanvas* canvasAvgSepLamKchP = new TCanvas("canvasAvgSepLamKchP","AvgSep LamKchP");
  TCanvas* canvasAvgSepLamKchM = new TCanvas("canvasAvgSepLamKchM","AvgSep LamKchM");
  TCanvas* canvasAvgSepALamKchP = new TCanvas("canvasAvgSepALamKchP","AvgSep ALamKchP");
  TCanvas* canvasAvgSepALamKchM = new TCanvas("canvasAvgSepALamKchM","AvgSep ALamKchM");
  cLamcKchAnalysis->DrawFinalAvgSepCFs(canvasAvgSepLamKchP,canvasAvgSepLamKchM,canvasAvgSepALamKchP,canvasAvgSepALamKchM);
  if(SavePDFs)
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"AvgSepCfs"),kTRUE);
    canvasAvgSepLamKchP->SaveAs(aSaveDirectory+"AvgSepCfs/LamKchP_AvgSepCfs"+aCentralityTag+".pdf");
    canvasAvgSepLamKchM->SaveAs(aSaveDirectory+"AvgSepCfs/LamKchM_AvgSepCfs"+aCentralityTag+".pdf");
    canvasAvgSepALamKchP->SaveAs(aSaveDirectory+"AvgSepCfs/ALamKchP_AvgSepCfs"+aCentralityTag+".pdf");
    canvasAvgSepALamKchM->SaveAs(aSaveDirectory+"AvgSepCfs/ALamKchM_AvgSepCfs"+aCentralityTag+".pdf");
  }

  //-----Purity
  TCanvas* canvasPurity = new TCanvas("canvasPurity","Purity");
  cLamcKchAnalysis->DrawFinalPurity(canvasPurity);
  if(SavePDFs) 
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"Purity"),kTRUE);
    canvasPurity->SaveAs(aSaveDirectory+"Purity/cLamcKch_Purity"+aCentralityTag+".pdf");
  }

/*
  //-----SepCfs
  TCanvas* canvasSepLamKchP = new TCanvas("canvasSepLamKchP","SepCfs LamKchP");
  TCanvas* canvasSepLamKchM = new TCanvas("canvasSepLamKchM","SepCfs LamKchM");
  TCanvas* canvasSepALamKchP = new TCanvas("canvasSepALamKchP","SepCfs ALamKchP");
  TCanvas* canvasSepALamKchM = new TCanvas("canvasSepALamKchM","SepCfs ALamKchM");
  cLamcKchAnalysis->DrawFinalSepCFs(canvasSepLamKchP,canvasSepLamKchM,canvasSepALamKchP,canvasSepALamKchM);
  if(SavePDFs)
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"Purity"),kTRUE);
    canvasSepLamKchP->SaveAs(aSaveDirectory+"SepCfs/LamKchP_SepCfs"+aCentralityTag+".pdf");
    canvasSepLamKchM->SaveAs(aSaveDirectory+"SepCfs/LamKchM_SepCfs"+aCentralityTag+".pdf");
    canvasSepALamKchP->SaveAs(aSaveDirectory+"SepCfs/ALamKchP_SepCfs"+aCentralityTag+".pdf");
    canvasSepALamKchM->SaveAs(aSaveDirectory+"SepCfs/ALamKchM_SepCfs"+aCentralityTag+".pdf");
  }

  //-----Avg of SepCfs
  TCanvas* canvasAvgOfSepsLamKchP = new TCanvas("canvasAvgOfSepsLamKchP","Avg of SepCfs LamKchP");
  TCanvas* canvasAvgOfSepsLamKchM = new TCanvas("canvasAvgOfSepsLamKchM","Avg of SepCfs LamKchM");
  TCanvas* canvasAvgOfSepsALamKchP = new TCanvas("canvasAvgOfSepsALamKchP","Avg of SepCfs ALamKchP");
  TCanvas* canvasAvgOfSepsALamKchM = new TCanvas("canvasAvgOfSepsALamKchM","Avg of SepCfs ALamKchM");
  cLamcKchAnalysis->DrawAvgOfFinalSepCFs(canvasAvgOfSepsLamKchP,canvasAvgOfSepsLamKchM,canvasAvgOfSepsALamKchP,canvasAvgOfSepsALamKchM);
  if(SavePDFs)
  {
    canvasAvgOfSepsLamKchP->SaveAs(aSaveDirectory+"SepCfs/LamKchP_AvgOfSepCfs"+aCentralityTag+".pdf");
    canvasAvgOfSepsLamKchM->SaveAs(aSaveDirectory+"SepCfs/LamKchM_AvgOfSepCfs"+aCentralityTag+".pdf");
    canvasAvgOfSepsALamKchP->SaveAs(aSaveDirectory+"SepCfs/ALamKchP_AvgOfSepCfs"+aCentralityTag+".pdf");
    canvasAvgOfSepsALamKchM->SaveAs(aSaveDirectory+"SepCfs/ALamKchM_AvgOfSepCfs"+aCentralityTag+".pdf");
  }


  //-----Cowboys and Sailors
  TCanvas* canvasCowLamKchP = new TCanvas("canvasCowLamKchP","AvgSepCow LamKchP");
  TCanvas* canvasCowLamKchM = new TCanvas("canvasCowLamKchM","AvgSepCow LamKchM");
  TCanvas* canvasCowALamKchP = new TCanvas("canvasCowALamKchP","AvgSepCow ALamKchP");
  TCanvas* canvasCowALamKchM = new TCanvas("canvasCowALamKchM","AvgSepCow ALamKchM");
  cLamcKchAnalysis->DrawFinalCowCFs(canvasCowLamKchP,canvasCowLamKchM,canvasCowALamKchP,canvasCowALamKchM);
  if(SavePDFs)
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"AvgSepCfsCowboysAndSailors"),kTRUE);
    canvasCowLamKchP->SaveAs(aSaveDirectory+"AvgSepCfsCowboysAndSailors/LamKchP_AvgSepCfsCowboysAndSailors"+aCentralityTag+".pdf");
    canvasCowLamKchM->SaveAs(aSaveDirectory+"AvgSepCfsCowboysAndSailors/LamKchM_AvgSepCfsCowboysAndSailors"+aCentralityTag+".pdf");
    canvasCowALamKchP->SaveAs(aSaveDirectory+"AvgSepCfsCowboysAndSailors/ALamKchP_AvgSepCfsCowboysAndSailors"+aCentralityTag+".pdf");
    canvasCowALamKchM->SaveAs(aSaveDirectory+"AvgSepCfsCowboysAndSailors/ALamKchM_AvgSepCfsCowboysAndSailors"+aCentralityTag+".pdf");
  }

  //-----MC Purity
  double LamKchPPurity_BeforePairCuts = cLamcKchAnalysis->GetMCKchPurity(buildAll::kLamKchP, true);
  double LamKchPPurity_AfterPairCuts = cLamcKchAnalysis->GetMCKchPurity(buildAll::kLamKchP, false);

  double ALamKchPPurity_BeforePairCuts = cLamcKchAnalysis->GetMCKchPurity(buildAll::kALamKchP, true);
  double ALamKchPPurity_AfterPairCuts = cLamcKchAnalysis->GetMCKchPurity(buildAll::kALamKchP, false);

  double LamKchMPurity_BeforePairCuts = cLamcKchAnalysis->GetMCKchPurity(buildAll::kLamKchM, true);
  double LamKchMPurity_AfterPairCuts = cLamcKchAnalysis->GetMCKchPurity(buildAll::kLamKchM, false);

  double ALamKchMPurity_BeforePairCuts = cLamcKchAnalysis->GetMCKchPurity(buildAll::kALamKchM, true);
  double ALamKchMPurity_AfterPairCuts = cLamcKchAnalysis->GetMCKchPurity(buildAll::kALamKchM, false);

  cout << "LamKchP analysis:  KchP purity BEFORE pair cuts = " <<  LamKchPPurity_BeforePairCuts << endl;
  cout << "LamKchP analysis:  KchP purity AFTER pair cuts = " <<  LamKchPPurity_AfterPairCuts << endl;

  cout << "ALamKchP analysis:  KchP purity BEFORE pair cuts = " <<  ALamKchPPurity_BeforePairCuts << endl;
  cout << "ALamKchP analysis:  KchP purity AFTER pair cuts = " <<  ALamKchPPurity_AfterPairCuts << endl;

  cout << "LamKchM analysis:  KchM purity BEFORE pair cuts = " <<  LamKchMPurity_BeforePairCuts << endl;
  cout << "LamKchM analysis:  KchM purity AFTER pair cuts = " <<  LamKchMPurity_AfterPairCuts << endl;

  cout << "ALamKchM analysis:  KchM purity BEFORE pair cuts = " <<  ALamKchMPurity_BeforePairCuts << endl;
  cout << "ALamKchM analysis:  KchM purity AFTER pair cuts = " <<  ALamKchMPurity_AfterPairCuts << endl;

*/


  TCanvas* canvasKStarOutBinned = new TCanvas("canvasKStarOutBinned","canvasKStarOutBinned");
  cLamcKchAnalysis->DrawKStarCfsBinnedInKStarOut(canvasKStarOutBinned);

  //-----ROOT file to save
  if(SaveRootFile)
  {
    TFile *myFile = new TFile(aSaveDirectory+aSaveRootFileName, "RECREATE");
    cLamcKchAnalysis->SaveAll(myFile);
    myFile->Close();
  }
}
