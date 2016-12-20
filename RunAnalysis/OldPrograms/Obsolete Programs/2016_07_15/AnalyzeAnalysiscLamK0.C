#include "buildAllcLamK0.cxx"
#include <stdio.h>
#include <string.h>

char* AppendBaseDirectory(TString aBaseName, char* aAppend)
{
  char* ReturnStr = new char[80];
  strcpy(ReturnStr, gSystem->ExpandPathName((const char*)aBaseName));
  strcat(ReturnStr, aAppend);
  return ReturnStr;
}

void AnalyzeAnalysiscLamK0()
{
  vector<TString> VectorOfFileNames;
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamK0_AsRc_20150824/Results_cLamK0_AsRc_20150824_Bp1.root");
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamK0_AsRc_20150824/Results_cLamK0_AsRc_20150824_Bp2.root");
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamK0_AsRc_20150824/Results_cLamK0_AsRc_20150824_Bm1.root");
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamK0_AsRc_20150824/Results_cLamK0_AsRc_20150824_Bm2.root");
    VectorOfFileNames.push_back("~/Analysis/K0Lam/Results_cLamK0_AsRc_20150824/Results_cLamK0_AsRc_20150824_Bm3.root");

  TString aSaveDirectory = "~/Analysis/K0Lam/Results_cLamK0_AsRc_20150824/3050/";  //note:  This directory must already be created
  TString aSaveRootFileName = "Results_cLamK0_AsRc_20150824_3050.root";
  TString aCentralityTag = "_3050";

  gSystem->mkdir(gSystem->ExpandPathName((const char*)aSaveDirectory), kTRUE);  //This will create the directory if not already created
										//If already created, this does nothing
				//I must call ExpandPathName because the name cannot contain any special shell characters (i.e. ~ or $)
				//and I must cast aSaveDirectory as const char* so ExpandPathName returns char*
				//instead of returning a Bool_t when used with a TString.
				//I want to keep aSaveDirectory as a TString so I can simply add strings to it (i.e. aSaveDirectory + "....")

  bool SavePDFs = true;
  bool SaveRootFile = true;

  //-------------------------------------------------------------------------------------------
  buildAllcLamK0 *cLamK0Analysis = new buildAllcLamK0(VectorOfFileNames,"LamK0_3050","ALamK0_3050");
  //cLamK0Analysis->SetDebug(kTRUE);
  cLamK0Analysis->BuildCFCollections();
  cLamK0Analysis->BuildAvgSepCollections();
  cLamK0Analysis->BuildPurityCollections();
  //cLamK0Analysis->BuildSepCollections(5);
  //cLamK0Analysis->BuildCowCollections(4);

  //-----KStarCfs
  TCanvas* canvasCfs = new TCanvas("canvasCfs","KStar Cfs");
  cLamK0Analysis->DrawFinalCFs(canvasCfs);
  if(SavePDFs) 
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"Cfs"),kTRUE);
    canvasCfs->SaveAs(aSaveDirectory+"Cfs/cLamK0_Cfs"+aCentralityTag+".pdf");
  }

  //-----AvgSepCfs
  TCanvas* canvasAvgSepLamK0 = new TCanvas("canvasAvgSepLamK0","AvgSep LamK0");
  TCanvas* canvasAvgSepALamK0 = new TCanvas("canvasAvgSepALamK0","AvgSep ALamK0");
  cLamK0Analysis->DrawFinalAvgSepCFs(canvasAvgSepLamK0,canvasAvgSepALamK0);
  if(SavePDFs)
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"AvgSepCfs"),kTRUE);
    canvasAvgSepLamK0->SaveAs(aSaveDirectory+"AvgSepCfs/LamK0_AvgSepCfs"+aCentralityTag+".pdf");
    canvasAvgSepALamK0->SaveAs(aSaveDirectory+"AvgSepCfs/ALamK0_AvgSepCfs"+aCentralityTag+".pdf");
  }

  //-----Purity
  TCanvas* canvasPurity = new TCanvas("canvasPurity","Purity");
  cLamK0Analysis->DrawFinalPurity(canvasPurity);
  if(SavePDFs) 
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"Purity"),kTRUE);
    canvasPurity->SaveAs(aSaveDirectory+"Purity/cLamK0_Purity"+aCentralityTag+".pdf");
  }

/*
  //-----SepCfs
  TCanvas* canvasSepLamK0LikeSigns = new TCanvas("canvasSepLamK0LikeSigns","SepCfs LamK0 Like Signs");
  TCanvas* canvasSepLamK0UnlikeSigns = new TCanvas("canvasSepLamK0UnlikeSigns","SepCfs LamK0 Unlike Signs");
  TCanvas* canvasSepALamK0LikeSigns = new TCanvas("canvasSepALamK0LikeSigns","SepCfs ALamK0 Like Signs");
  TCanvas* canvasSepALamK0UnlikeSigns = new TCanvas("canvasSepALamK0UnlikeSigns","SepCfs ALamK0 Unlike Signs");
  cLamK0Analysis->DrawFinalSepCFs(canvasSepLamK0LikeSigns,canvasSepLamK0UnlikeSigns,canvasSepALamK0LikeSigns,canvasSepALamK0UnlikeSigns);
  if(SavePDFs)
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"SepCfs"),kTRUE);
    canvasSepLamK0LikeSigns->SaveAs(aSaveDirectory+"SepCfs/LamK0_SepCfsLikeSigns"+aCentralityTag+".pdf");
    canvasSepLamK0UnlikeSigns->SaveAs(aSaveDirectory+"SepCfs/LamK0_SepCfsUnlikeSigns"+aCentralityTag+".pdf");
    canvasSepALamK0LikeSigns->SaveAs(aSaveDirectory+"SepCfs/ALamK0_SepCfsLikeSigns"+aCentralityTag+".pdf");
    canvasSepALamK0UnlikeSigns->SaveAs(aSaveDirectory+"SepCfs/ALamK0_SepCfsUnlikeSigns"+aCentralityTag+".pdf");
  }

  //-----Avg of SepCfs
  TCanvas* canvasAvgOfSepsLamK0LikeSigns = new TCanvas("canvasAvgOfSepsLamK0LikeSigns","Avg of SepCfs LamK0 Like Signs");
  TCanvas* canvasAvgOfSepsLamK0UnlikeSigns = new TCanvas("canvasAvgOfSepsLamK0UnlikeSigns","Avg of SepCfs LamK0 Unlike Signs");
  TCanvas* canvasAvgOfSepsALamK0LikeSigns = new TCanvas("canvasAvgOfSepsALamK0LikeSigns","Avg of SepCfs ALamK0 Like Signs");
  TCanvas* canvasAvgOfSepsALamK0UnlikeSigns = new TCanvas("canvasAvgOfSepsALamK0UnlikeSigns","Avg of SepCfs ALamK0 Unlike Signs");
  cLamK0Analysis->DrawAvgOfFinalSepCFs(canvasAvgOfSepsLamK0LikeSigns,canvasAvgOfSepsLamK0UnlikeSigns,canvasAvgOfSepsALamK0LikeSigns,canvasAvgOfSepsALamK0UnlikeSigns);
  if(SavePDFs)
  {
    canvasAvgOfSepsLamK0LikeSigns->SaveAs(aSaveDirectory+"SepCfs/LamK0_AvgOfSepCfsLikeSigns"+aCentralityTag+".pdf");
    canvasAvgOfSepsLamK0UnlikeSigns->SaveAs(aSaveDirectory+"SepCfs/LamK0_AvgOfSepCfsUnlikeSigns"+aCentralityTag+".pdf");
    canvasAvgOfSepsALamK0LikeSigns->SaveAs(aSaveDirectory+"SepCfs/ALamK0_AvgOfSepCfsLikeSigns"+aCentralityTag+".pdf");
    canvasAvgOfSepsALamK0UnlikeSigns->SaveAs(aSaveDirectory+"SepCfs/ALamK0_AvgOfSepCfsUnlikeSigns"+aCentralityTag+".pdf");
  }


  //-----Cowboys and Sailors
  TCanvas* canvasCowLamK0 = new TCanvas("canvasCowLamK0","AvgSepCow LamK0");
  TCanvas* canvasCowALamK0 = new TCanvas("canvasCowALamK0","AvgSepCow ALamK0");
  cLamK0Analysis->DrawFinalCowCFs(canvasCowLamK0,canvasCowALamK0);
  if(SavePDFs)
  {
    gSystem->mkdir(AppendBaseDirectory(aSaveDirectory,"AvgSepCfsCowboysAndSailors"),kTRUE);
    canvasCowLamK0->SaveAs(aSaveDirectory+"AvgSepCfsCowboysAndSailors/LamK0_AvgSepCfsCowboysAndSailors"+aCentralityTag+".pdf");
    canvasCowALamK0->SaveAs(aSaveDirectory+"AvgSepCfsCowboysAndSailors/ALamK0_AvgSepCfsCowboysAndSailors"+aCentralityTag+".pdf");
  }
*/

  //-----ROOT file to save
  if(SaveRootFile)
  {
    TFile *myFile = new TFile(aSaveDirectory+aSaveRootFileName, "RECREATE");
    cLamK0Analysis->SaveAll(myFile);
    myFile->Close();
  }

}
