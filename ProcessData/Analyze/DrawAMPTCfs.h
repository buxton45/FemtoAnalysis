#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TList.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "CanvasPartition.h"
#include "CfLite.h"
#include "CfHeavy.h"

//_________________________________________________________________________________________
TObjArray* GetAnDir(AnalysisType aAnType, CentralityType aAnDirCentType, TString aFileLocation)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TDirectoryFile *tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  TString tFemtoListName;
  if(aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP) tFemtoListName = TString("cLamcKch");
  else if(aAnType==kLamK0 || aAnType==kALamK0) tFemtoListName = TString("cLamK0");
  else if(aAnType==kXiKchP || aAnType==kAXiKchM || aAnType==kXiKchM || aAnType==kAXiKchP) tFemtoListName = TString("cXicKch");
  else assert(0);
  tFemtoListName += TString("_femtolist");

  TList *tFemtolist = (TList*)tDirFile->Get(tFemtoListName);
  tFemtolist->SetOwner();
  //----------------------------------
  TString tAnDirName;
  if(aAnDirCentType != kMB) tAnDirName = TString(cAnalysisBaseTags[aAnType]) + TString(cCentralityTags[aAnDirCentType]);
  else tAnDirName = TString(cAnalysisBaseTags[aAnType]) + TString("_0100");
  tAnDirName.ReplaceAll("0010","010");

  TObjArray *tAnDir = (TObjArray*)tFemtolist->FindObject(tAnDirName)->Clone();
  tAnDir->SetOwner();

  //----------------------------------
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtolist object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtolist);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }

  tFemtolist->Delete();
  delete tFemtolist;

  tDirFile->Close();
  delete tDirFile;

  tFile->Close();
  delete tFile;


  //----------------------------------

  return tAnDir;
}

//_________________________________________________________________________________________
TH1* Get1dHist(TObjArray* aAnDir, TString aHistName, TString aNewName="")
{
  TH1* tHist = (TH1*)aAnDir->FindObject(aHistName);
  tHist->SetDirectory(0);

  //-----make sure tHisto is retrieved
  if(!tHist) {cout << "1dHist NOT FOUND!!!:  Name:  " << aHistName << endl;}
  assert(tHist);
  //----------------------------------

  TH1 *ReturnHist = (TH1*)tHist->Clone(aNewName);
    ReturnHist->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHist->GetSumw2N()) {ReturnHist->Sumw2();}

  return (TH1*)ReturnHist;
}

//_________________________________________________________________________________________
CfLite* BuildCfLite(TString aFileLocationBase, TString aCentralityFile, AnalysisType aAnType, CentralityType aAnDirCentType=kMB, int aRebin=1, double aMinNorm=0.32, double aMaxNorm=0.40)
{
  TObjArray* tAnDir = GetAnDir(aAnType, aAnDirCentType, aFileLocationBase);

  TH1* tNum = Get1dHist(tAnDir, TString::Format("NumKStarCf_%s", cAnalysisBaseTags[aAnType]));
  TH1* tDen = Get1dHist(tAnDir, TString::Format("DenKStarCf_%s", cAnalysisBaseTags[aAnType]));

  CfLite* tCfLite = new CfLite(TString::Format("CfLite_%s_%s", cAnalysisBaseTags[aAnType], aCentralityFile.Data()), 
                               TString::Format("CfLite_%s_%s", cAnalysisBaseTags[aAnType], aCentralityFile.Data()), 
                               tNum, tDen, aMinNorm, aMaxNorm);
  tCfLite->Rebin(aRebin);

  delete tAnDir;

  return tCfLite;
}

//_________________________________________________________________________________________
bool CheckCentralityFilesAgainstAnDirCentType(const vector<TString> &aCentralityFiles, CentralityType aAnDirCentType)
{
  if(aAnDirCentType == kMB) return true;
  else
  {
    for(unsigned int iCentFile=0; iCentFile < aCentralityFiles.size(); iCentFile++)
    {
      if     (aAnDirCentType == k0010) {if(!(aCentralityFiles[iCentFile].EqualTo("0005") || aCentralityFiles[iCentFile].EqualTo("0510"))) return false;}
      else if(aAnDirCentType == k1030) {if(!(aCentralityFiles[iCentFile].EqualTo("1020") || aCentralityFiles[iCentFile].EqualTo("2030"))) return false;}
      else if(aAnDirCentType == k3050) {if(!(aCentralityFiles[iCentFile].EqualTo("3040") || aCentralityFiles[iCentFile].EqualTo("4050"))) return false;}
      else return false;
    }
  }
  return true;
}


//_________________________________________________________________________________________
CfHeavy* BuildCfHeavy(TString aCfName, const vector<TString> &aFileLocationBases, const vector<TString> &aCentralityFiles, const vector<AnalysisType> &aAnTypes, CentralityType aAnDirCentType=kMB, int aRebin=1, double aMinNorm=0.32, double aMaxNorm=0.40)
{
  //NOTE:  aFileLocationBases should align with aCentralityFiles
  assert(aFileLocationBases.size() == aCentralityFiles.size());
  assert(CheckCentralityFilesAgainstAnDirCentType(aCentralityFiles, aAnDirCentType));

  vector<CfLite*> tCfLiteVec(0);
  for(unsigned int iFile=0; iFile<aFileLocationBases.size(); iFile++)
  {
    for(unsigned int iAn=0; iAn<aAnTypes.size(); iAn++)
    {
      CfLite* tCfLite = BuildCfLite(aFileLocationBases[iFile], aCentralityFiles[iFile], aAnTypes[iAn], aAnDirCentType, aRebin, aMinNorm, aMaxNorm);
      tCfLiteVec.push_back(tCfLite);
    }
  }

  CfHeavy* tCfHeavy = new CfHeavy(aCfName, aCfName, tCfLiteVec, aMinNorm, aMaxNorm);
  return tCfHeavy;
}


//_________________________________________________________________________________________
TH1* TypicalGetAMPTCf(TString aResultsDate, CentralityType aAnDirCentType, AnalysisType aAnType, TString aCentralityFile, bool aCombineAllcLamcKch, bool aCombineConjugates, bool aCombineCentFiles, int aRebin=1, double aMinNorm=0.32, double aMaxNorm=0.40)
{
  if(aAnType==kLamK0 || aAnType==kALamK0) assert(!aCombineAllcLamcKch);

  //If data stored in separate directories for each centrality, 0010, 1030, 3050
  //    choose corresponding CentralityType
  //If data stored in one, large, 0100 directory, choose kMB
  if(aResultsDate.EqualTo("20180312")) aAnDirCentType = kMB;
  //---------------------------------------------------

  AnalysisType tConjAnType;
  if     (aAnType==kLamK0)    {tConjAnType=kALamK0;}
  else if(aAnType==kALamK0)   {tConjAnType=kLamK0;}
  else if(aAnType==kLamKchP)  {tConjAnType=kALamKchM;}
  else if(aAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(aAnType==kLamKchM)  {tConjAnType=kALamKchP;}
  else if(aAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);


  vector<AnalysisType> tAnTypesVec;
  if     (aCombineAllcLamcKch) tAnTypesVec = {kLamKchP, kALamKchM, kLamKchM, kALamKchP};
  else if(aCombineConjugates)  tAnTypesVec = {aAnType, tConjAnType};
  else                         tAnTypesVec = {aAnType};

  vector<TString> tCentralityFilesVec;
  if(!aCombineCentFiles) tCentralityFilesVec = {aCentralityFile};
  else
  {
    if     (aCentralityFile.EqualTo("0005") || aCentralityFile.EqualTo("0510")) tCentralityFilesVec = vector<TString>{"0005", "0510"};
    else if(aCentralityFile.EqualTo("1020") || aCentralityFile.EqualTo("2030")) tCentralityFilesVec = vector<TString>{"1020", "2030"};
    else if(aCentralityFile.EqualTo("3040") || aCentralityFile.EqualTo("4050")) tCentralityFilesVec = vector<TString>{"3040", "4050"};
    else if(aCentralityFile.EqualTo("5060") || aCentralityFile.EqualTo("6070") || aCentralityFile.EqualTo("7080")) tCentralityFilesVec = vector<TString>{"5060", "6070", "7080"};
    else assert(0);
  }

//-----------------------------------------------------------------------------

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/AMPT/%s/",aResultsDate.Data());
  vector<TString> tFileLocationBasesVec(0);

  for(unsigned int i=0; i<tCentralityFilesVec.size(); i++) tFileLocationBasesVec.push_back(TString::Format("%sAMPT_%s.root",tDirectoryBase.Data(), tCentralityFilesVec[i].Data()));

  TString tHeavyCfName = "CfHeavy";
  for(unsigned int i=0; i<tAnTypesVec.size(); i++) tHeavyCfName += TString::Format("_%s", cAnalysisBaseTags[tAnTypesVec[i]]);
  for(unsigned int iCentFile=0; iCentFile<tCentralityFilesVec.size(); iCentFile++) tHeavyCfName += TString::Format("_%s", tCentralityFilesVec[iCentFile].Data());
  CfHeavy* tCfHeavy = BuildCfHeavy(tHeavyCfName, tFileLocationBasesVec, tCentralityFilesVec, tAnTypesVec, aAnDirCentType, aRebin, aMinNorm, aMaxNorm);

  return (TH1*)tCfHeavy->GetHeavyCf();


}


