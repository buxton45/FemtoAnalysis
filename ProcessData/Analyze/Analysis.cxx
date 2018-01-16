///////////////////////////////////////////////////////////////////////////
// Analysis:                                                             //
///////////////////////////////////////////////////////////////////////////


#include "Analysis.h"

#ifdef __ROOT__
ClassImp(Analysis)
#endif




//GLOBAL!

const double LambdaMass = 1.115683, KaonMass = 0.493677;

//______________________________________________________________________________________________________________
bool gBgReject;
double gBgFitLow[2];
double gBgFitHigh[2];

double BgFitFunction(double *x, double *par)
{
  if( gBgReject && !(x[0]>gBgFitLow[0] && x[0]<gBgFitLow[1]) && !(x[0]>gBgFitHigh[0] && x[0]<gBgFitHigh[1]) )
    {
      TF1::RejectPoint();
      return 0;
    }
  
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
}

//______________________________________________________________________________________________________________
bool gRejectOmega;
double gRejectOmegaLow = 0.19;
double gRejectOmegaHigh = 0.23;

double MomResFitFunction(double *x, double *par)
{
  if(gRejectOmega && x[0] > gRejectOmegaLow && x[0] < gRejectOmegaHigh)
  {
    TF1::RejectPoint();
    return 0;
  }

  //return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);

  //return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3];

  //return exp(-par[0]*x[0])*par[1]*cos(par[2]*x[0]-par[3]) + par[4];

  //return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0)))*exp(-par[3]*x[0]) + par[4];


  double tGamma = par[1];
  double tOmega0 = par[2];
  double tOmega = sqrt(tOmega0*tOmega0 - tGamma*tGamma);

  return par[0]*exp(-tGamma*x[0])*cos(tOmega*x[0] - par[3]) + par[4];

}


//________________________________________________________________________________________________________________
double FitGaus(double *x, double *par)
{
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3];
}

//________________________________________________________________________________________________________________
double FitGausAndLine(double *x, double *par)
{
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3]*x[0] + par[4];
}

//________________________________________________________________________________________________________________
double FitTwoGaus(double *x, double *par)
{
  if(x[0] < 0) return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0)));
  else return par[3]*exp(-0.5*(pow((x[0]-par[4])/par[5],2.0)));
}

//________________________________________________________________________________________________________________
double FitTwoExp(double *x, double *par)
{
  if(x[0] < 0) return par[0]*exp((x[0]+par[1])/par[2]) + par[3];
  else return par[4]*exp(-(x[0]+par[5])/par[6]) + par[7];
}

//________________________________________________________________________________________________________________
double FitLorentz(double *x, double *par) {
  return (0.5*par[0]*par[1]/TMath::Pi()) /
    TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2])
   + .25*par[1]*par[1]);
}



//________________________________________________________________________________________________________________
double FitQuadratic(double *x, double *par)
{
  return par[0]*x[0]*x[0];
}

//________________________________________________________________________________________________________________
double FitPoly(double *x, double *par)
{
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4) + par[5]*pow(x[0],5) + par[6]*pow(x[0],6) + par[7]*pow(x[0],7) + par[8]*pow(x[0],8);
}

//________________________________________________________________________________________________________________
double FitGausAndTwoExp(double *x, double *par)
{
  double tGaus = par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0)));
  double tExpLeft = par[3]*exp((x[0]+par[4])/par[5]);
  double tExpRight = par[6]*exp(-(x[0]+par[7])/par[8]);
  double tRegionCutLeft = -0.014;
  double tRegionCutRight = 0.009;

  if(x[0] >= tRegionCutLeft && x[0] <= tRegionCutRight) return tGaus;
  if(x[0] < tRegionCutLeft) return tGaus+tExpLeft;
  if(x[0] > tRegionCutRight) return tGaus+tExpRight;
  else return 0;

/*
  if(x[0] > 0) return tGaus+tExpRight;
  else return tGaus+tExpLeft;
*/
}

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
Analysis::Analysis(TString aAnalysisName, vector<PartialAnalysis*> &aPartialAnalysisCollection, bool aCombineConjugates) :
  fAnalysisRunType(kTrain),
  fCombineConjugates(aCombineConjugates),
  fAnalysisName(aAnalysisName),
  fPartialAnalysisCollection(aPartialAnalysisCollection),

  fAnalysisType(fPartialAnalysisCollection[0]->GetAnalysisType()),
  fCentralityType(fPartialAnalysisCollection[0]->GetCentralityType()),

  fParticleTypes(2),
  fDaughterParticleTypes(0),


  fNPartialAnalysis(fPartialAnalysisCollection.size()),

  fNEventsPass(0),
  fNEventsFail(0),
  fNPart1Pass(0),
  fNPart1Fail(0),
  fNPart2Pass(0),
  fNPart2Fail(0),
  fNKStarNumEntries(0),

  fKStarHeavyCf(0),
  fKStarHeavyCfMCTrue(0),

  fModelKStarHeavyCfTrue(0),
  fModelKStarHeavyCfTrueIdeal(0),
  fModelKStarHeavyCfFake(0),
  fModelKStarHeavyCfFakeIdeal(0),
  fModelKStarHeavyCfFakeIdealSmeared(0),
  fModelCfTrueIdealCfTrueRatio(0),
  fMomResFit(0),
  fModelCfFakeIdealCfFakeRatio(0),
  fMomResFitFake(0),

  fModelKStarHeavyCfTrueUnitWeights(0),
  fModelKStarHeavyCfTrueIdealUnitWeights(0),

//  fModelKStarTrueVsRecTotal(0),
  fModelKStarTrueVsRecSameTot(0),
  fModelKStarTrueVsRecRotSameTot(0),
  fModelKStarTrueVsRecMixedTot(0),
  fModelKStarTrueVsRecRotMixedTot(0),

  fMomResMatrixFitSame(0),
  fMomResMatrixFitMixed(0),

  fDaughterPairTypes(0),
  fAvgSepHeavyCfs(kMaxNDaughterPairTypes),  //make size equal to total number of pair types so SepCfs can be stored in consistent matter
                                            //ex, for LamKchP analysis, {[0],[1],[2],[3]} = empty, {[4],[5]} = full, {[6]} = empty

  fSepHeavyCfs(kMaxNDaughterPairTypes),
  fAvgSepCowSailHeavyCfs(kMaxNDaughterPairTypes),

  fKStar2dHeavyCfKStarOut(0),
  fKStar1dHeavyCfKStarOutPos(0),
  fKStar1dHeavyCfKStarOutNeg(0),
  fKStar1dHeavyKStarOutPosNegRatio(0),

  fKStar2dHeavyCfKStarSide(0),
  fKStar1dHeavyCfKStarSidePos(0),
  fKStar1dHeavyCfKStarSideNeg(0),
  fKStar1dHeavyKStarSidePosNegRatio(0),

  fKStar2dHeavyCfKStarLong(0),
  fKStar1dHeavyCfKStarLongPos(0),
  fKStar1dHeavyCfKStarLongNeg(0),
  fKStar1dHeavyKStarLongPosNegRatio(0),

  fPurityCollection(0),

  fPart1MassFail(0)


{
  //make sure partial analyses in collection have same pair type (AnalysisType) and centrality (CentralityType)
  for(int i=1; i<fNPartialAnalysis; i++)
  {
    if(!fCombineConjugates) assert(fPartialAnalysisCollection[i-1]->GetAnalysisType() == fPartialAnalysisCollection[i]->GetAnalysisType());
    assert(fPartialAnalysisCollection[i-1]->GetCentralityType() == fPartialAnalysisCollection[i]->GetCentralityType());
  }

  //fAnalysisType = fPartialAnalysisCollection[0]->GetAnalysisType();
  //fCentralityType = fPartialAnalysisCollection[0]->GetCentralityType();

  //Don't need to make sure they have same daughter pair types, because analysis types are same
  fDaughterPairTypes = fPartialAnalysisCollection[0]->GetDaughterPairTypes();

  fParticleTypes = fPartialAnalysisCollection[0]->GetParticleTypes();
  fDaughterParticleTypes = fPartialAnalysisCollection[0]->GetDaughterParticleTypes();

  for(int i=0; i<fNPartialAnalysis; i++)
  {
    fNEventsPass += fPartialAnalysisCollection[i]->GetNEventsPass();
    fNEventsFail += fPartialAnalysisCollection[i]->GetNEventsFail();

    fNPart1Pass += fPartialAnalysisCollection[i]->GetNPart1Pass();
    fNPart1Fail += fPartialAnalysisCollection[i]->GetNPart1Fail();

    fNPart2Pass += fPartialAnalysisCollection[i]->GetNPart2Pass();
    fNPart2Fail += fPartialAnalysisCollection[i]->GetNPart2Fail();

    fNKStarNumEntries += fPartialAnalysisCollection[i]->GetNKStarNumEntries();
  }

  //---------------------------------
  TString tPart1MassFailName;
  if(fParticleTypes[0]==kLam)  {tPart1MassFailName = "LambdaMass_" + TString(cParticleTags[fParticleTypes[0]]) + "_Fail";}
  else if(fParticleTypes[0]==kALam) {tPart1MassFailName = "AntiLambdaMass_" + TString(cParticleTags[fParticleTypes[0]]) + "_Fail";}

  for(int i=0; i<fNPartialAnalysis; i++)
  {
    if(i==0) {fPart1MassFail = (TH1*)fPartialAnalysisCollection[i]->GetPart1MassFail()->Clone(tPart1MassFailName);}
    else{fPart1MassFail->Add(fPartialAnalysisCollection[i]->GetPart1MassFail());}
  }

}


//________________________________________________________________________________________________________________
Analysis::Analysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, TString aDirNameModifier) :
  fAnalysisRunType(aRunType),
  fCombineConjugates(false),
  fAnalysisName(0),
  fPartialAnalysisCollection(0),

  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),

  fParticleTypes(2),
  fDaughterParticleTypes(0),


  fNPartialAnalysis(aNPartialAnalysis),

  fNEventsPass(0),
  fNEventsFail(0),
  fNPart1Pass(0),
  fNPart1Fail(0),
  fNPart2Pass(0),
  fNPart2Fail(0),
  fNKStarNumEntries(0),

  fKStarHeavyCf(0),
  fKStarHeavyCfMCTrue(0),

  fModelKStarHeavyCfTrue(0),
  fModelKStarHeavyCfTrueIdeal(0),
  fModelKStarHeavyCfFake(0),
  fModelKStarHeavyCfFakeIdeal(0),
  fModelKStarHeavyCfFakeIdealSmeared(0),
  fModelCfTrueIdealCfTrueRatio(0),
  fMomResFit(0),
  fModelCfFakeIdealCfFakeRatio(0),
  fMomResFitFake(0),

  fModelKStarHeavyCfTrueUnitWeights(0),
  fModelKStarHeavyCfTrueIdealUnitWeights(0),


//  fModelKStarTrueVsRecTotal(0),
  fModelKStarTrueVsRecSameTot(0),
  fModelKStarTrueVsRecRotSameTot(0),
  fModelKStarTrueVsRecMixedTot(0),
  fModelKStarTrueVsRecRotMixedTot(0),

  fMomResMatrixFitSame(0),
  fMomResMatrixFitMixed(0),

  fDaughterPairTypes(0),
  fAvgSepHeavyCfs(kMaxNDaughterPairTypes),  //make size equal to total number of pair types so SepCfs can be stored in consistent matter
                                            //ex, for LamKchP analysis, {[0],[1],[2],[3]} = empty, {[4],[5]} = full, {[6]} = empty
  fSepHeavyCfs(kMaxNDaughterPairTypes),
  fAvgSepCowSailHeavyCfs(kMaxNDaughterPairTypes),

  fKStar2dHeavyCfKStarOut(0),
  fKStar1dHeavyCfKStarOutPos(0),
  fKStar1dHeavyCfKStarOutNeg(0),
  fKStar1dHeavyKStarOutPosNegRatio(0),

  fKStar2dHeavyCfKStarSide(0),
  fKStar1dHeavyCfKStarSidePos(0),
  fKStar1dHeavyCfKStarSideNeg(0),
  fKStar1dHeavyKStarSidePosNegRatio(0),

  fKStar2dHeavyCfKStarLong(0),
  fKStar1dHeavyCfKStarLongPos(0),
  fKStar1dHeavyCfKStarLongNeg(0),
  fKStar1dHeavyKStarLongPosNegRatio(0),

  fPurityCollection(0),
 
  fPart1MassFail(0)


{

  fAnalysisName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  int iStart;
  if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys) iStart=0;
  else iStart = 2;

  for(int i=iStart; i<fNPartialAnalysis+iStart; i++)
  {
    BFieldType tBFieldType = static_cast<BFieldType>(i);

    TString tFileLocation = aFileLocationBase + cBFieldTags[tBFieldType];
    tFileLocation += ".root";

    TString tPartialAnalysisName = fAnalysisName + cBFieldTags[tBFieldType];

    PartialAnalysis* tPartialAnalysis = new PartialAnalysis(tFileLocation, tPartialAnalysisName, fAnalysisType, tBFieldType, fCentralityType, fAnalysisRunType, aDirNameModifier);

    fPartialAnalysisCollection.push_back(tPartialAnalysis);
  } 

  fDaughterPairTypes = fPartialAnalysisCollection[0]->GetDaughterPairTypes();

  fParticleTypes = fPartialAnalysisCollection[0]->GetParticleTypes();
  fDaughterParticleTypes = fPartialAnalysisCollection[0]->GetDaughterParticleTypes();

  for(int i=0; i<fNPartialAnalysis; i++)
  {
    fNEventsPass += fPartialAnalysisCollection[i]->GetNEventsPass();
    fNEventsFail += fPartialAnalysisCollection[i]->GetNEventsFail();

    fNPart1Pass += fPartialAnalysisCollection[i]->GetNPart1Pass();
    fNPart1Fail += fPartialAnalysisCollection[i]->GetNPart1Fail();

    fNPart2Pass += fPartialAnalysisCollection[i]->GetNPart2Pass();
    fNPart2Fail += fPartialAnalysisCollection[i]->GetNPart2Fail();

    fNKStarNumEntries += fPartialAnalysisCollection[i]->GetNKStarNumEntries();
  }

  //---------------------------------
  TString tPart1MassFailName;
  if(fParticleTypes[0]==kLam)  {tPart1MassFailName = "LambdaMass_" + TString(cParticleTags[fParticleTypes[0]]) + "_Fail";}
  else if(fParticleTypes[0]==kALam) {tPart1MassFailName = "AntiLambdaMass_" + TString(cParticleTags[fParticleTypes[0]]) + "_Fail";}

  if(fAnalysisRunType != kTrainSys)  //TrainSys analyses DO NOT include FAIL cut monitors
  {
    for(int i=0; i<fNPartialAnalysis; i++)
    {
      if(i==0) {fPart1MassFail = (TH1*)fPartialAnalysisCollection[i]->GetPart1MassFail()->Clone(tPart1MassFailName);}
      else{fPart1MassFail->Add(fPartialAnalysisCollection[i]->GetPart1MassFail());}
    }
  }

}

//________________________________________________________________________________________________________________
Analysis::~Analysis()
{
  cout << "Analysis object (name: " << fAnalysisName << " ) is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
TH1* Analysis::SimpleAddTH1Collection(TString tHistosName)
{
  TH1* tReturnHist;
  for(int i=0; i<fNPartialAnalysis; i++)
  {
    if(i==0) tReturnHist = (TH1*)fPartialAnalysisCollection[i]->Get1dHisto(tHistosName,tHistosName+TString::Format("_%d",i))->Clone();
    else tReturnHist->Add((TH1*)fPartialAnalysisCollection[i]->Get1dHisto(tHistosName,tHistosName+TString::Format("_%d",i)));
  }
  return tReturnHist;
}


//________________________________________________________________________________________________________________
vector<ParticleType> Analysis::GetCorrectDaughterParticleTypes(DaughterPairType aDaughterPairType)
{
  vector<ParticleType> tDaughterParticleTypes(0);

  //In fDaughterParticleTypes[i][j], i=MotherId (0,1) and j=DaughterId(0=+, 1=-)

  if( (aDaughterPairType==kPosPos) || (aDaughterPairType==kPosNeg) || (aDaughterPairType==kNegPos) || (aDaughterPairType==kNegNeg) )
  {
    assert(fDaughterParticleTypes.size() == 2);
    tDaughterParticleTypes.resize(2);

    if(aDaughterPairType==kPosPos)
    {
      tDaughterParticleTypes[0] = fDaughterParticleTypes[0][0];
      tDaughterParticleTypes[1] = fDaughterParticleTypes[1][0];
    }
    else if(aDaughterPairType==kPosNeg)
    {
      tDaughterParticleTypes[0] = fDaughterParticleTypes[0][0];
      tDaughterParticleTypes[1] = fDaughterParticleTypes[1][1];
    }
    else if(aDaughterPairType==kNegPos)
    {
      tDaughterParticleTypes[0] = fDaughterParticleTypes[0][1];
      tDaughterParticleTypes[1] = fDaughterParticleTypes[1][0];
    }
    else if(aDaughterPairType==kNegNeg)
    {
      tDaughterParticleTypes[0] = fDaughterParticleTypes[0][1];
      tDaughterParticleTypes[1] = fDaughterParticleTypes[1][1];
    }
  }

  else if( (aDaughterPairType==kTrackPos) || (aDaughterPairType==kTrackNeg) || (aDaughterPairType==kTrackBac) )
  {
    assert(fDaughterParticleTypes.size() == 1);
    tDaughterParticleTypes.resize(1);

    if(aDaughterPairType==kTrackPos)
    {
      tDaughterParticleTypes[0] = fDaughterParticleTypes[0][0];
    }
    else if(aDaughterPairType==kTrackNeg)
    {
      tDaughterParticleTypes[0] = fDaughterParticleTypes[0][1];
    }
    else if(aDaughterPairType==kTrackBac)
    {
      tDaughterParticleTypes[0] = fDaughterParticleTypes[0][2];
    }
  }

  return tDaughterParticleTypes;
}


//________________________________________________________________________________________________________________
TString Analysis::GetDaughtersHistoTitle(DaughterPairType aDaughterPairType, bool aIsBacPion)
{
  vector<ParticleType> tDaughterParticleTypes = GetCorrectDaughterParticleTypes(aDaughterPairType);
  TString tReturnTitle;

  if(tDaughterParticleTypes.size() == 2)
  {
    tReturnTitle = TString(cRootParticleTags[tDaughterParticleTypes[0]]) + "(" + TString(cRootParticleTags[fParticleTypes[0]]) + ") - " + TString(cRootParticleTags[tDaughterParticleTypes[1]]) + "(" + TString(cRootParticleTags[fParticleTypes[1]]) + ")";
  }

  else if(tDaughterParticleTypes.size() == 1)
  {
    if(fAnalysisType!=kXiKchP && fAnalysisType!=kAXiKchM && fAnalysisType!=kXiKchM && fAnalysisType!=kAXiKchP)
    {
      tReturnTitle = TString(cRootParticleTags[tDaughterParticleTypes[0]]) + "(" + TString(cRootParticleTags[fParticleTypes[0]]) + ") - " + TString(cRootParticleTags[fParticleTypes[1]]);
    }
    else
    {
      if(aIsBacPion) tReturnTitle = TString(cRootParticleTags[tDaughterParticleTypes[0]]) + "(" + TString(cRootParticleTags[fParticleTypes[0]]) + ") - " + TString(cRootParticleTags[fParticleTypes[1]]);
      else
      {
        tReturnTitle = TString(cRootParticleTags[tDaughterParticleTypes[0]]);
        if(fParticleTypes[0]==kXi) tReturnTitle += TString("(") + TString(cRootParticleTags[0]);
        else if(fParticleTypes[0]==kAXi) tReturnTitle += TString("(") + TString(cRootParticleTags[1]);
        else assert(0);
        tReturnTitle += TString("(") + TString(cRootParticleTags[fParticleTypes[0]]) + TString(")) - ") + TString(cRootParticleTags[fParticleTypes[1]]);
      }
    }
  }

  return tReturnTitle;
}


//________________________________________________________________________________________________________________
void Analysis::BuildKStarHeavyCf(double aMinNorm, double aMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated
    //fPartialAnalysisCollection[iAnaly]->GetKStarCf()->BuildCf(aMinNorm,aMaxNorm);
    fPartialAnalysisCollection[iAnaly]->GetKStarCf()->Rebin(aRebin,aMinNorm,aMaxNorm);  //CfLite::Rebin calls BuildCf

    //CfLite *tTempCfLite = fPartialAnalysisCollection[iAnaly]->GetKStarCf();
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetKStarCf());
  }

  TString tCfBaseName = "KStarHeavyCf_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fKStarHeavyCf = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);

}

//________________________________________________________________________________________________________________
void Analysis::DrawKStarHeavyCf(TPad* aPad, int aMarkerColor, TString aOption, int aMarkerStyle, double aXMin, double aXMax, double aYMin, double aYMax)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fKStarHeavyCf->GetHeavyCf();

  double tXmin = 0.;
  double tXmax = 0.5;

  double tYmin = 0.82;
  double tYmax = 1.02;

  if(aXMin>-1. && aXMax>-1. && aYMin>-1. && aYMax>-1.)
  {
    tXmin = aXMin;
    tXmax = aXMax;

    tYmin = aYMin;
    tYmax = aYMax;
  }

  TAxis *xax1 = tCfToDraw->GetXaxis();
    xax1->SetTitle("#it{k}* (GeV/#it{c})");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
    xax1->SetRangeUser(tXmin,tXmax);
  TAxis *yax1 = tCfToDraw->GetYaxis();
    yax1->SetRangeUser(tYmin,tYmax);
    yax1->SetTitle("#it{C}(#it{k}*)");
    yax1->SetTitleSize(0.045);
    yax1->SetTitleOffset(1.0);
    //yax1->CenterTitle();


  //------------------------------------------------------
  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);
  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  //tCfToDraw->SetTitle(tTitle);



  //------------------------------------------------------
  TLine *line = new TLine(tXmin,1,tXmax,1);
  line->SetLineColor(14);

  tCfToDraw->Draw(aOption);
  line->Draw();

/*
  TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
  leg1->SetFillColor(0);

  leg1->AddEntry(tCfToDraw, tTitle,"lp");
  leg1->Draw();
*/

}

//________________________________________________________________________________________________________________
void Analysis::SaveAllKStarHeavyCf(TFile* aFile)
{
  assert(aFile->IsOpen());
  fKStarHeavyCf->SaveAllCollectionsAndCf("",TString(cAnalysisBaseTags[fAnalysisType]),aFile);
}












//________________________________________________________________________________________________________________
void Analysis::BuildKStarHeavyCfMCTrue(double aMinNorm, double aMaxNorm)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated
    fPartialAnalysisCollection[iAnaly]->BuildKStarCfMCTrue(aMinNorm,aMaxNorm);

    //CfLite *tTempCfLite = fPartialAnalysisCollection[iAnaly]->GetKStarCf();
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetKStarCfMCTrue());
  }

  TString tCfBaseName = "KStarHeavyCfMCTrue_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fKStarHeavyCfMCTrue = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);

}

//________________________________________________________________________________________________________________
void Analysis::DrawKStarHeavyCfMCTrue(TPad* aPad, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fKStarHeavyCfMCTrue->GetHeavyCf();

  TAxis *xax1 = tCfToDraw->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
    xax1->SetRangeUser(0.,0.5);
  TAxis *yax1 = tCfToDraw->GetYaxis();
    yax1->SetRangeUser(0.9,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();


  //------------------------------------------------------
  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);
  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  //tCfToDraw->SetTitle(tTitle);



  //------------------------------------------------------
  TLine *line = new TLine(0,1,1,1);
  line->SetLineColor(14);

  tCfToDraw->Draw(aOption);
  line->Draw();

/*
  TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
  leg1->SetFillColor(0);

  leg1->AddEntry(tCfToDraw, tTitle,"lp");
  leg1->Draw();
*/

}


//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarHeavyCfTrue(double aMinNorm, double aMaxNorm)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated, then push
    fPartialAnalysisCollection[iAnaly]->BuildModelKStarCfTrue(aMinNorm,aMaxNorm);
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetModelKStarCfTrue());
  }

  TString tCfBaseName = "ModelKStarHeavyCfTrue_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfTrue = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void Analysis::DrawModelKStarHeavyCfTrue(TPad* aPad, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fModelKStarHeavyCfTrue->GetHeavyCf();

  TAxis *xax1 = tCfToDraw->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
    xax1->SetRangeUser(0.,0.5);
  TAxis *yax1 = tCfToDraw->GetYaxis();
    yax1->SetRangeUser(0.9,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();
  //------------------------------------------------------
  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);
  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  //tCfToDraw->SetTitle(tTitle);

  //------------------------------------------------------
  TLine *line = new TLine(0,1,1,1);
  line->SetLineColor(14);

  tCfToDraw->Draw(aOption);
  line->Draw();

/*
  TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
  leg1->SetFillColor(0);

  leg1->AddEntry(tCfToDraw, tTitle,"lp");
  leg1->Draw();
*/

}

//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarHeavyCfTrueIdeal(double aMinNorm, double aMaxNorm)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated, then push
    fPartialAnalysisCollection[iAnaly]->BuildModelKStarCfTrueIdeal(aMinNorm,aMaxNorm);
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetModelKStarCfTrueIdeal());
  }

  TString tCfBaseName = "ModelKStarHeavyCfTrueIdeal_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfTrueIdeal = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarHeavyCfFake(double aMinNorm, double aMaxNorm)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated, then push
    fPartialAnalysisCollection[iAnaly]->BuildModelKStarCfFake(aMinNorm,aMaxNorm);
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetModelKStarCfFake());
  }

  TString tCfBaseName = "ModelKStarHeavyCfFake_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfFake = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarHeavyCfFakeIdeal(double aMinNorm, double aMaxNorm)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated, then push
    fPartialAnalysisCollection[iAnaly]->BuildModelKStarCfFakeIdeal(aMinNorm,aMaxNorm);
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetModelKStarCfFakeIdeal());
  }

  TString tCfBaseName = "ModelKStarHeavyCfFakeIdeal_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfFakeIdeal = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarHeavyCfFakeIdealSmeared(TH2* aMomResMatrix, double aMinNorm, double aMaxNorm, int aRebinFactor)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated, then push
    fPartialAnalysisCollection[iAnaly]->BuildModelKStarCfFakeIdealSmeared(aMomResMatrix,aMinNorm,aMaxNorm,aRebinFactor);
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetModelKStarCfFakeIdealSmeared());
  }

  TString tCfBaseName = "ModelKStarHeavyCfFakeIdealSmeared_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfFakeIdealSmeared = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void Analysis::BuildModelCfTrueIdealCfTrueRatio(double aMinNorm, double aMaxNorm, int aRebinFactor)
{
  //First, build the necessary Cfs
  BuildModelKStarHeavyCfTrue(aMinNorm,aMaxNorm);
  BuildModelKStarHeavyCfTrueIdeal(aMinNorm,aMaxNorm);

  TString tName = "ModelCfTrueIdealCfTrueRatio_" + TString(cAnalysisBaseTags[fAnalysisType]);

  fModelKStarHeavyCfTrueIdeal->Rebin(aRebinFactor);
  fModelKStarHeavyCfTrue->Rebin(aRebinFactor);

  fModelCfTrueIdealCfTrueRatio = (TH1*)fModelKStarHeavyCfTrueIdeal->GetHeavyCf()->Clone(tName);
  fModelCfTrueIdealCfTrueRatio->SetTitle(tName);
  fModelCfTrueIdealCfTrueRatio->Divide(fModelKStarHeavyCfTrue->GetHeavyCf());
}

//________________________________________________________________________________________________________________
void Analysis::BuildModelCfFakeIdealCfFakeRatio(double aMinNorm, double aMaxNorm, int aRebinFactor)
{
  //First, build the necessary Cfs
  BuildModelKStarHeavyCfFake(aMinNorm,aMaxNorm);
  BuildModelKStarHeavyCfFakeIdeal(aMinNorm,aMaxNorm);

  TString tName = "ModelCfFakeIdealCfFakeRatio_" + TString(cAnalysisBaseTags[fAnalysisType]);

  fModelKStarHeavyCfFakeIdeal->Rebin(aRebinFactor);
  fModelKStarHeavyCfFake->Rebin(aRebinFactor);

  fModelCfFakeIdealCfFakeRatio = (TH1*)fModelKStarHeavyCfFakeIdeal->GetHeavyCf()->Clone(tName);
  fModelCfFakeIdealCfFakeRatio->SetTitle(tName);
  fModelCfFakeIdealCfFakeRatio->Divide(fModelKStarHeavyCfFake->GetHeavyCf());
}



//________________________________________________________________________________________________________________
void Analysis::FitModelCfTrueIdealCfTrueRatio()
{
  double tFitLow = 0.;
  double tFitHigh = 0.2;

  bool tRejectOmega = false;
  if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) tRejectOmega = true;
  gRejectOmega = tRejectOmega;

  TF1* tFitBgd1 = new TF1("tFitBgd1",MomResFitFunction,tFitLow,tFitHigh,5);

  if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM)
  {
    tFitBgd1->SetParameter(0,0.073);
    tFitBgd1->SetParameter(1,92.9);
    tFitBgd1->SetParameter(2,93.5);
    tFitBgd1->SetParameter(3,-1.61);
    tFitBgd1->SetParameter(4,1.);
  }

  if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP)
  {
    tFitBgd1->SetParameter(0,0.005);
    tFitBgd1->SetParameter(1,70.0);
    tFitBgd1->SetParameter(2,80.);
    tFitBgd1->SetParameter(3,0.);
    tFitBgd1->SetParameter(4,1.);
  }


/*
    if(fAnalysisType==kLamKchP)
    {
      tFitBgd1->SetParameter(0,-64.2);
      tFitBgd1->SetParameter(1,-0.432);
      tFitBgd1->SetParameter(2,0.100);
      tFitBgd1->SetParameter(3,1.0);
    }
    else if(fAnalysisType==kALamKchP)
    {
      tFitBgd1->SetParameter(0,-0.000696);
      tFitBgd1->SetParameter(1,0.00517);
      tFitBgd1->SetParameter(2,0.0550);
      tFitBgd1->SetParameter(3,1.);
    }
    else if(fAnalysisType==kLamKchM)
    {
      tFitBgd1->SetParameter(0,-0.000737);
      tFitBgd1->SetParameter(1,0.00981);
      tFitBgd1->SetParameter(2,0.0510);
      tFitBgd1->SetParameter(3,1.);
    }
    else if(fAnalysisType==kALamKchM)
    {
      tFitBgd1->SetParameter(0,-59.4);
      tFitBgd1->SetParameter(1,-0.458);
      tFitBgd1->SetParameter(2,0.108);
      tFitBgd1->SetParameter(3,1.);
    }

    else
    {
      tFitBgd1->SetParameter(0,-0.01);
      tFitBgd1->SetParameter(1,0.0);
      tFitBgd1->SetParameter(2,0.1);
      tFitBgd1->SetParameter(3,1.);
    }
*/
  fModelCfTrueIdealCfTrueRatio->Fit("tFitBgd1","0R");

  if(tRejectOmega == true)  //get fit function without missing RejectOmega region
  {
    gRejectOmega = false;
    TF1 *tFitBgd2 = new TF1("tFitBgd2",MomResFitFunction,tFitLow,tFitHigh,5);
    tFitBgd2->SetParameters(tFitBgd1->GetParameters());
    fMomResFit = tFitBgd2;
  }
  else fMomResFit = tFitBgd1;

}

//________________________________________________________________________________________________________________
void Analysis::FitModelCfFakeIdealCfFakeRatio()
{
  double tFitLow = 0.;
  double tFitHigh = 0.2;

  bool tRejectOmega = false;
  if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) tRejectOmega = true;
  gRejectOmega = tRejectOmega;

  TF1* tFitBgd1 = new TF1("tFitBgd1",MomResFitFunction,tFitLow,tFitHigh,5);

  if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM)
  {
    tFitBgd1->SetParameter(0,0.073);
    tFitBgd1->SetParameter(1,92.9);
    tFitBgd1->SetParameter(2,93.5);
    tFitBgd1->SetParameter(3,-1.61);
    tFitBgd1->SetParameter(4,1.);
  }

  if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP)
  {
    tFitBgd1->SetParameter(0,0.005);
    tFitBgd1->SetParameter(1,70.0);
    tFitBgd1->SetParameter(2,80.);
    tFitBgd1->SetParameter(3,0.);
    tFitBgd1->SetParameter(4,1.);
  }


/*
    tFitBgd1->SetParameter(0,-0.01);
    //tFitBgd1->SetParameter(1,0.0);
    tFitBgd1->FixParameter(1,0.0);
    tFitBgd1->SetParameter(2,0.1);
    tFitBgd1->SetParameter(3,1.);
*/

  fModelCfFakeIdealCfFakeRatio->Fit("tFitBgd1","0R");

  if(tRejectOmega == true)  //get fit function without missing RejectOmega region
  {
    gRejectOmega = false;
    TF1 *tFitBgd2 = new TF1("tFitBgd2",MomResFitFunction,tFitLow,tFitHigh,5);
    tFitBgd2->SetParameters(tFitBgd1->GetParameters());
    fMomResFitFake = tFitBgd2;
  }
  else fMomResFitFake = tFitBgd1;

}

//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarHeavyCfTrueUnitWeights(double aMinNorm, double aMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated, then push
    fPartialAnalysisCollection[iAnaly]->BuildModelKStarCfTrueUnitWeights(aMinNorm,aMaxNorm);
    fPartialAnalysisCollection[iAnaly]->GetModelKStarCfTrueUnitWeights()->Rebin(aRebin);
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetModelKStarCfTrueUnitWeights());
  }

  TString tCfBaseName = "ModelKStarHeavyCfTrueUnitWeights_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfTrueUnitWeights = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarHeavyCfTrueIdealUnitWeights(double aMinNorm, double aMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated, then push
    fPartialAnalysisCollection[iAnaly]->BuildModelKStarCfTrueIdealUnitWeights(aMinNorm,aMaxNorm);
    fPartialAnalysisCollection[iAnaly]->GetModelKStarCfTrueIdealUnitWeights()->Rebin(aRebin);
    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetModelKStarCfTrueIdealUnitWeights());
  }

  TString tCfBaseName = "ModelKStarHeavyCfTrueIdealUnitWeights_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfTrueIdealUnitWeights = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}


//________________________________________________________________________________________________________________
void Analysis::NormalizeTH2ByTotalEntries(TH2* aHist)
{
  double tNorm = aHist->GetEntries();
  aHist->Scale(1./tNorm);
}


//________________________________________________________________________________________________________________
void Analysis::NormalizeTH2EachColumn(TH2* aHist)
{
  int tNbinsX = aHist->GetNbinsX();
  int tNbinsY = aHist->GetNbinsY();

  for(int i=1; i<=tNbinsX; i++)
  {
    double tScale = aHist->Integral(i,i,1,tNbinsY);
    if(tScale > 0.)
    {
      for(int j=1; j<=tNbinsY; j++)
      {
        double tNewContent = (1.0/tScale)*aHist->GetBinContent(i,j);
        aHist->SetBinContent(i,j,tNewContent);
      }
    }
  }
}


//________________________________________________________________________________________________________________
void Analysis::NormalizeTH2EachRow(TH2* aHist)
{
  int tNbinsX = aHist->GetNbinsX();
  int tNbinsY = aHist->GetNbinsY();

  for(int j=1; j<=tNbinsY; j++)
  {
    double tScale = aHist->Integral(1,tNbinsX,j,j);
    if(tScale > 0.)
    {
      for(int i=1; i<=tNbinsX; i++)
      {
        double tNewContent = (1.0/tScale)*aHist->GetBinContent(i,j);
        aHist->SetBinContent(i,j,tNewContent);
      }
    }
  }
}



/*
//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarTrueVsRecTotal()
{
  TString tName = "ModelKStarTrueVsRecTotal_" + TString(cAnalysisBaseTags[fAnalysisType]);
  for(int i=0; i<fNPartialAnalysis; i++)
  {
    fPartialAnalysisCollection[i]->BuildModelKStarTrueVsRec();
    if(i==0) 
    {
      fModelKStarTrueVsRecTotal = (TH2*)fPartialAnalysisCollection[i]->GetModelKStarTrueVsRec()->Clone(tName);
      if(!fModelKStarTrueVsRecTotal->GetSumw2N()) {fModelKStarTrueVsRecTotal->Sumw2();}
    }
    else fModelKStarTrueVsRecTotal->Add((TH2*)fPartialAnalysisCollection[i]->GetModelKStarTrueVsRec());
  }
}
*/

//________________________________________________________________________________________________________________
void Analysis::BuildModelKStarTrueVsRecTotal(KStarTrueVsRecType aType)
{
  TString tName = "ModelKStarTrueVsRec" + TString(cKStarTrueVsRecTypeTags[aType]) + "_" + TString(cAnalysisBaseTags[fAnalysisType]);

  TH2* tPre = (TH2*)fPartialAnalysisCollection[0]->GetModelKStarTrueVsRec(aType);
  if(!tPre)
  {
    fPartialAnalysisCollection[0]->BuildModelKStarTrueVsRec(aType);
    tPre = (TH2*)fPartialAnalysisCollection[0]->GetModelKStarTrueVsRec(aType);
  }
  TH2* tReturnHisto = (TH2*)tPre->Clone(tName);

  for(int i=1; i<fNPartialAnalysis; i++)
  {
    TH2* tToAdd = (TH2*)fPartialAnalysisCollection[i]->GetModelKStarTrueVsRec(aType);
    if(!tToAdd)
    {
      fPartialAnalysisCollection[i]->BuildModelKStarTrueVsRec(aType);
      tToAdd = (TH2*)fPartialAnalysisCollection[i]->GetModelKStarTrueVsRec(aType);
    }
    tReturnHisto->Add(tToAdd);
  }

  switch(aType) {
  case kSame:
    fModelKStarTrueVsRecSameTot = tReturnHisto;
    break;

  case kRotSame:
    fModelKStarTrueVsRecRotSameTot = tReturnHisto;
    break;

  case kMixed:
    fModelKStarTrueVsRecMixedTot = tReturnHisto;
    break;

  case kRotMixed:
    fModelKStarTrueVsRecRotMixedTot = tReturnHisto;
    break;


  default:
    cout << "ERROR: Analysis::BuildModelKStarTrueVsRecTotal:  Invalide KStarTrueVsRecType aType = "
         << aType << " selected" << endl;
    assert(0);
    break;
  }

}

//________________________________________________________________________________________________________________
void Analysis::BuildAllModelKStarTrueVsRecTotal()
{
  BuildModelKStarTrueVsRecTotal(kSame);
  BuildModelKStarTrueVsRecTotal(kRotSame);
  BuildModelKStarTrueVsRecTotal(kMixed);
  BuildModelKStarTrueVsRecTotal(kRotMixed);
}

//________________________________________________________________________________________________________________
TH2* Analysis::GetModelKStarTrueVsRecTotal(KStarTrueVsRecType aType)
{
  switch(aType) {
  case kSame:
    if(!fModelKStarTrueVsRecSameTot) BuildModelKStarTrueVsRecTotal(aType);
    return fModelKStarTrueVsRecSameTot;
    break;

  case kRotSame:
    if(!fModelKStarTrueVsRecRotSameTot) BuildModelKStarTrueVsRecTotal(aType);
    return fModelKStarTrueVsRecRotSameTot;
    break;

  case kMixed:
    if(!fModelKStarTrueVsRecMixedTot) BuildModelKStarTrueVsRecTotal(aType);
    return fModelKStarTrueVsRecMixedTot;
    break;

  case kRotMixed:
    if(!fModelKStarTrueVsRecRotMixedTot) BuildModelKStarTrueVsRecTotal(aType);
    return fModelKStarTrueVsRecRotMixedTot;
    break;


  default:
    cout << "ERROR: Analysis::GetModelKStarTrueVsRecTotal:  Invalide KStarTrueVsRecType aType = "
         << aType << " selected" << endl;
    assert(0);
  }

  return 0;
}

//________________________________________________________________________________________________________________
void Analysis::BuildMomResMatrixFit(KStarTrueVsRecType aType, double aKStarLow, double aKStarHigh)
{
  TH2* tKTrueVsRecRot;
  if(aType == kSame || aType == kRotSame) tKTrueVsRecRot = GetModelKStarTrueVsRecTotal(kRotSame);
  else tKTrueVsRecRot = GetModelKStarTrueVsRecTotal(kRotMixed);

  int tBinXLow = tKTrueVsRecRot->GetXaxis()->FindBin(0.);
  int tBinXHigh = tKTrueVsRecRot->GetXaxis()->FindBin(0.2);

  TString tNameDiffDist1 = "DiffDist1" + TString(cKStarTrueVsRecTypeTags[aType]) + "_" + TString(cAnalysisBaseTags[fAnalysisType]);
  TH1D* tDiffDist = tKTrueVsRecRot->ProjectionY(tNameDiffDist1,tBinXLow,tBinXHigh);
    double tMaxValueDiffDist = tDiffDist->GetBinContent(tDiffDist->GetMaximumBin());
cout << "\t\t\t\t\t\t\t tMaxValueDiffDist bin = " << tDiffDist->GetMaximumBin() << endl;
cout << "\t\t\t\t\t\t\t tMaxValueDiffDist = " << tMaxValueDiffDist << endl;
/*
  TF1* tFitDiffDist = new TF1("tFitDiffDist",FitGaus,-0.1,0.1,4);
    tFitDiffDist->SetParameter(0,tMaxValueDiffDist);
    tFitDiffDist->SetParameter(1,0.0001307);
    tFitDiffDist->SetParameter(2,0.00305);
    tFitDiffDist->SetParameter(3,3.);
*/

  TF1* tFitDiffDist = new TF1("tFitDiffDist",FitGausAndTwoExp,-0.1,0.1,9);
    tFitDiffDist->SetParameter(0,tMaxValueDiffDist);
    tFitDiffDist->SetParameter(1,0.0001307);
    tFitDiffDist->SetParameter(2,0.00305);

    tFitDiffDist->SetParameter(3,1.);
    tFitDiffDist->SetParameter(4,1.);
    tFitDiffDist->SetParameter(5,0.01);

    tFitDiffDist->SetParameter(6,1.);
    tFitDiffDist->SetParameter(7,-1.);
    tFitDiffDist->SetParameter(8,0.01);

/*
  TF1* tFitDiffDist = new TF1("tFitDiffDist",FitGausAndLine,-0.1,0.1,5);
cout << "\t\t\t\t\t\t\t\t tDiffDist->FindBin(0.08) = " << tDiffDist->FindBin(0.08) << endl;
cout << "\t\t\t\t\t\t\t\t tDiffDist->GetBinContent(tDiffDist->FindBin(0.08)) = " << tDiffDist->GetBinContent(tDiffDist->FindBin(0.08)) << endl;
cout << "\t\t\t\t\t\t\t\t tDiffDist->FindBin(-0.08) = " << tDiffDist->FindBin(-0.08) << endl;
cout << "\t\t\t\t\t\t\t\t tDiffDist->GetBinContent(tDiffDist->FindBin(-0.08)) = " << tDiffDist->GetBinContent(tDiffDist->FindBin(-0.08)) << endl;
    double tSlope = (tDiffDist->GetBinContent(tDiffDist->FindBin(0.08))-tDiffDist->GetBinContent(tDiffDist->FindBin(-0.08)))/0.16;
cout << "\t\t\t\t\t\t\t tSlope = " << tSlope << endl;
    tFitDiffDist->SetParameter(0,tMaxValueDiffDist);
    tFitDiffDist->SetParameter(1,0.0001307);
    tFitDiffDist->SetParameter(2,0.00305);
    tFitDiffDist->SetParameter(3,tSlope);
    tFitDiffDist->SetParameter(4,2.);
*/
/*
  TF1* tFitDiffDist = new TF1("tFitDiffDist",FitLorentz,-0.1,0.1,3);
    tFitDiffDist->SetParameter(1,0.63662/tMaxValueDiffDist);
*/
/*
  TF1* tFitDiffDist = new TF1("tFitDiffDist",FitTwoGaus,-0.1,0.1,6);
    tFitDiffDist->SetParameter(0,tMaxValueDiffDist);
    tFitDiffDist->SetParameter(1,0.0001307);
    tFitDiffDist->SetParameter(2,0.00305);
    tFitDiffDist->SetParameter(3,tMaxValueDiffDist);
    tFitDiffDist->SetParameter(4,0.0001307);
    tFitDiffDist->SetParameter(5,0.00305);
*/
/*
  TF1* tFitDiffDist = new TF1("tFitDiffDist",FitTwoExp,-0.1,0.1,8);
    tFitDiffDist->SetParameter(0,tMaxValueDiffDist);
    tFitDiffDist->SetParameter(1,0.0001307);
    tFitDiffDist->SetParameter(2,0.00305);
    tFitDiffDist->SetParameter(3,4.);
    tFitDiffDist->SetParameter(4,tMaxValueDiffDist);
    tFitDiffDist->SetParameter(5,0.0001307);
    tFitDiffDist->SetParameter(6,0.00305);
    tFitDiffDist->SetParameter(7,1.);
*/
  tDiffDist->Fit("tFitDiffDist","0R");

  TF1* tFitDiffDistFull = new TF1("tFitDiffDistFull",FitGausAndTwoExp,-0.2,0.2,9);
  tFitDiffDistFull->SetParameters(tFitDiffDist->GetParameters());


  TString tNamePairDist1 = "PairDist1" + TString(cKStarTrueVsRecTypeTags[aType]) + "_" + TString(cAnalysisBaseTags[fAnalysisType]);
  TH1D* tPairDist = tKTrueVsRecRot->ProjectionX(tNamePairDist1,1,tKTrueVsRecRot->GetNbinsY());
  //TF1* tFitPairDist = new TF1("tFitPairDist",FitQuadratic,0.,0.15,1);
  TF1* tFitPairDist = new TF1("tFitPairDist",FitPoly,0.,1.0,9);
  tPairDist->Fit("tFitPairDist","0R");



  TString tCanName = "CanMatrixFit_" + TString(cAnalysisBaseTags[fAnalysisType]);
  TCanvas* tCanMatrixFit = new TCanvas(tCanName,tCanName);
  tCanMatrixFit->Divide(2,1);

  tCanMatrixFit->cd(1);
    gPad->SetLogy();
    tDiffDist->GetXaxis()->SetRangeUser(-0.3,0.3);
    tDiffDist->Draw();
    tFitDiffDistFull->Draw("same");

  tCanMatrixFit->cd(2);
    tPairDist->Draw();
    tFitPairDist->Draw("same");

//  tCanMatrixFit->SaveAs("~/Analysis/Presentations/Group Meetings/20160310/MatrixFit_LamKchP.eps");

  TCanvas *tCanKTrueVsRecRot = new TCanvas("CanKTrueVsRecRot","CanKTrueVsRecRot");
  tCanKTrueVsRecRot->cd();
  gPad->SetLogz();
  tKTrueVsRecRot->Draw("colz");
//  tCanKTrueVsRecRot->SaveAs("~/Analysis/Presentations/Group Meetings/20160310/TrueVsRecRot_LamKchP.eps");
  //-----------------------------------------------------
  TH2D* tMomResMatrixFit = new TH2D("tMomResMatrixFit","tMomResMatrixFit",200,0.,0.2,200,0.,0.2);
  double tSum, tDiff;
  double tKTrue, tKRec;

  for(int i=0; i<100000000; i++)
  {
    tSum = tFitPairDist->GetRandom();
    tDiff = tFitDiffDist->GetRandom();

    tKTrue = tSum - tDiff/sqrt(2);
    tKRec = tSum + tDiff/sqrt(2);

    tMomResMatrixFit->Fill(tKTrue,tKRec);
  }

  TCanvas *tCanMomResMatrixFit = new TCanvas("CanMomResMatrixFit","CanMomResMatrixFit");
  tCanMomResMatrixFit->cd();
  gPad->SetLogz();
  tMomResMatrixFit->Draw("colz");
//  tCanMomResMatrixFit->SaveAs("~/Analysis/Presentations/Group Meetings/20160310/NewMatrix_LamKchP.eps");

  if(aType == kSame || aType == kRotSame) fMomResMatrixFitSame = tMomResMatrixFit;
  else if(aType == kMixed || aType == kRotMixed) fMomResMatrixFitMixed = tMomResMatrixFit;
}

//________________________________________________________________________________________________________________
TH2* Analysis::GetMomResMatrixFit(KStarTrueVsRecType aType)
{
  if(aType == kSame || aType == kRotSame) return fMomResMatrixFitSame;
  else return fMomResMatrixFitMixed;
}

//________________________________________________________________________________________________________________
void Analysis::BuildAvgSepHeavyCf(DaughterPairType aDaughterPairType, double aMinNorm, double aMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  //first, make sure all partial analyses have the same number of daughter 
  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //make sure everything is updated
    fPartialAnalysisCollection[iAnaly]->BuildAvgSepCf(aDaughterPairType,aMinNorm,aMaxNorm);

    //make sure the Cf exists
    assert(fPartialAnalysisCollection[iAnaly]->GetAvgSepCf(aDaughterPairType));

    tTempCfLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetAvgSepCf(aDaughterPairType));
  }

  TString tCfBaseName = "AvgSepCf";
  TString tCfName = tCfBaseName + cDaughterPairTags[aDaughterPairType] + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  CfHeavy *tCfHeavy = new CfHeavy(tCfName,tCfName,tTempCfLiteCollection,aMinNorm,aMaxNorm);
  if(aRebin != 1) tCfHeavy->Rebin(aRebin);

  fAvgSepHeavyCfs[aDaughterPairType] = tCfHeavy;
}



//________________________________________________________________________________________________________________
void Analysis::BuildAllAvgSepHeavyCfs(double aMinNorm, double aMaxNorm, int aRebin)
{
  int tNCfs = fDaughterPairTypes.size();

  for(int i=0; i<tNCfs; i++)
  {
    //make sure everything is updated

    BuildAvgSepHeavyCf(fDaughterPairTypes[i],aMinNorm,aMaxNorm,aRebin);
  }

}



//________________________________________________________________________________________________________________
CfHeavy* Analysis::GetAvgSepHeavyCf(DaughterPairType aDaughterPairType, int aRebin)
{
  if(aRebin != 1) fAvgSepHeavyCfs[aDaughterPairType]->Rebin(aRebin);
  return fAvgSepHeavyCfs[aDaughterPairType];
}

//________________________________________________________________________________________________________________
void Analysis::DrawAvgSepHeavyCf(DaughterPairType aDaughterPairType, TPad *aPad, bool aIsBacPion)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fAvgSepHeavyCfs[aDaughterPairType]->GetHeavyCf();

  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  //------------------------------------------------------------------

  tCfToDraw->GetYaxis()->SetRangeUser(-0.5,5.);
  TString tTitle = GetDaughtersHistoTitle(aDaughterPairType, aIsBacPion);

  tCfToDraw->GetXaxis()->SetTitle("Avg. Sep. (cm)");
  tCfToDraw->GetYaxis()->SetTitle("C(Avg. Sep)");

  tCfToDraw->SetTitle(tTitle);
  tCfToDraw->SetMarkerStyle(20);
  tCfToDraw->Draw();
  line->Draw();

  TLegend *tLeg = new TLegend(0.55,0.65,0.85,0.85);
    tLeg->AddEntry(tCfToDraw,tTitle,"p");
    tLeg->Draw();
}


//________________________________________________________________________________________________________________
void Analysis::SaveAllAvgSepHeavyCfs(TFile* aFile)
{
  assert(aFile->IsOpen());

  TString tPreName = "AvgSep";
  TString tPostName;


  for(unsigned int iDaughterPairs=0; iDaughterPairs<fDaughterPairTypes.size(); iDaughterPairs++)
  {
    tPostName = TString(cDaughterPairTags[fDaughterPairTypes[iDaughterPairs]]) + "_" + TString(cAnalysisBaseTags[fAnalysisType]);
    fAvgSepHeavyCfs[fDaughterPairTypes[iDaughterPairs]]->SaveAllCollectionsAndCf(tPreName,tPostName,aFile);
  }


}


//________________________________________________________________________________________________________________
void Analysis::BuildKStar2dHeavyCfs(double aMinNorm, double aMaxNorm)
{
  TString tCfBaseNameCommon = "KStar2dHeavyCf";

  vector<Cf2dLite*> tTempCf2dLiteCollection;

  //-----Build all KStar2dCfs first
  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    fPartialAnalysisCollection[iAnaly]->BuildKStar2dCfs();
  }


  //-----KStarOut
  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    assert(fPartialAnalysisCollection[iAnaly]->GetKStar2dCfKStarOut());
    tTempCf2dLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetKStar2dCfKStarOut());
  }

  TString tCfBaseNameKStarOut = tCfBaseNameCommon + "KStarOut";
  TString tCfDaughtersBaseNameKStarOut = tCfBaseNameKStarOut + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType]+"_";

  fKStar2dHeavyCfKStarOut = new Cf2dHeavy(tCfDaughtersBaseNameKStarOut,tTempCf2dLiteCollection,aMinNorm,aMaxNorm);

  fKStar1dHeavyCfKStarOutNeg = fKStar2dHeavyCfKStarOut->GetDaughterHeavyCf(0);
  fKStar1dHeavyCfKStarOutPos = fKStar2dHeavyCfKStarOut->GetDaughterHeavyCf(1);

  TString tRatioBaseNameKStarOut = "KStar1dHeavyKStarOutPosNegRatio";
  TString tRatioNameKStarOut = tRatioBaseNameKStarOut + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  fKStar1dHeavyKStarOutPosNegRatio = (TH1*)fKStar1dHeavyCfKStarOutPos->GetHeavyCf()->Clone(tRatioNameKStarOut);
  fKStar1dHeavyKStarOutPosNegRatio->SetTitle(tRatioNameKStarOut);
  fKStar1dHeavyKStarOutPosNegRatio->Divide(fKStar1dHeavyCfKStarOutNeg->GetHeavyCf());

  //-----KStarSide
  tTempCf2dLiteCollection.clear();

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    assert(fPartialAnalysisCollection[iAnaly]->GetKStar2dCfKStarSide());
    tTempCf2dLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetKStar2dCfKStarSide());
  }

  TString tCfBaseNameKStarSide = tCfBaseNameCommon + "KStarSide";
  TString tCfDaughtersBaseNameKStarSide = tCfBaseNameKStarSide + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType]+"_";

  fKStar2dHeavyCfKStarSide = new Cf2dHeavy(tCfDaughtersBaseNameKStarSide,tTempCf2dLiteCollection,aMinNorm,aMaxNorm);

  fKStar1dHeavyCfKStarSideNeg = fKStar2dHeavyCfKStarSide->GetDaughterHeavyCf(0);
  fKStar1dHeavyCfKStarSidePos = fKStar2dHeavyCfKStarSide->GetDaughterHeavyCf(1);

  TString tRatioBaseNameKStarSide = "KStar1dHeavyKStarSidePosNegRatio";
  TString tRatioNameKStarSide = tRatioBaseNameKStarSide + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  fKStar1dHeavyKStarSidePosNegRatio = (TH1*)fKStar1dHeavyCfKStarSidePos->GetHeavyCf()->Clone(tRatioNameKStarSide);
  fKStar1dHeavyKStarSidePosNegRatio->SetTitle(tRatioNameKStarSide);
  fKStar1dHeavyKStarSidePosNegRatio->Divide(fKStar1dHeavyCfKStarSideNeg->GetHeavyCf());

  //-----KStarLong
  tTempCf2dLiteCollection.clear();

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    assert(fPartialAnalysisCollection[iAnaly]->GetKStar2dCfKStarLong());
    tTempCf2dLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetKStar2dCfKStarLong());
  }

  TString tCfBaseNameKStarLong = tCfBaseNameCommon + "KStarLong";
  TString tCfDaughtersBaseNameKStarLong = tCfBaseNameKStarLong + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType]+"_";

  fKStar2dHeavyCfKStarLong = new Cf2dHeavy(tCfDaughtersBaseNameKStarLong,tTempCf2dLiteCollection,aMinNorm,aMaxNorm);

  fKStar1dHeavyCfKStarLongNeg = fKStar2dHeavyCfKStarLong->GetDaughterHeavyCf(0);
  fKStar1dHeavyCfKStarLongPos = fKStar2dHeavyCfKStarLong->GetDaughterHeavyCf(1);

  TString tRatioBaseNameKStarLong = "KStar1dHeavyKStarLongPosNegRatio";
  TString tRatioNameKStarLong = tRatioBaseNameKStarLong + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  fKStar1dHeavyKStarLongPosNegRatio = (TH1*)fKStar1dHeavyCfKStarLongPos->GetHeavyCf()->Clone(tRatioNameKStarLong);
  fKStar1dHeavyKStarLongPosNegRatio->SetTitle(tRatioNameKStarLong);
  fKStar1dHeavyKStarLongPosNegRatio->Divide(fKStar1dHeavyCfKStarLongNeg->GetHeavyCf());



}

//________________________________________________________________________________________________________________
void Analysis::RebinKStar2dHeavyCfs(int aRebinFactor)
{

  //-----KStarOut
  fKStar2dHeavyCfKStarOut->Rebin(aRebinFactor);

  //refresh objects which derive from fKStar2dHeavyCfKStarOut
  fKStar1dHeavyCfKStarOutNeg = fKStar2dHeavyCfKStarOut->GetDaughterHeavyCf(0);
  fKStar1dHeavyCfKStarOutPos = fKStar2dHeavyCfKStarOut->GetDaughterHeavyCf(1);

  TString tRatioBaseNameKStarOut = "KStar1dHeavyKStarOutPosNegRatio";
  TString tRatioNameKStarOut = tRatioBaseNameKStarOut + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  fKStar1dHeavyKStarOutPosNegRatio = (TH1*)fKStar1dHeavyCfKStarOutPos->GetHeavyCf()->Clone(tRatioNameKStarOut);
  fKStar1dHeavyKStarOutPosNegRatio->SetTitle(tRatioNameKStarOut);
  fKStar1dHeavyKStarOutPosNegRatio->Divide(fKStar1dHeavyCfKStarOutNeg->GetHeavyCf());

  //-----KStarSide
  fKStar2dHeavyCfKStarSide->Rebin(aRebinFactor);

  //refresh objects which derive from fKStar2dHeavyCfKStarSide
  fKStar1dHeavyCfKStarSideNeg = fKStar2dHeavyCfKStarSide->GetDaughterHeavyCf(0);
  fKStar1dHeavyCfKStarSidePos = fKStar2dHeavyCfKStarSide->GetDaughterHeavyCf(1);

  TString tRatioBaseNameKStarSide = "KStar1dHeavyKStarSidePosNegRatio";
  TString tRatioNameKStarSide = tRatioBaseNameKStarSide + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  fKStar1dHeavyKStarSidePosNegRatio = (TH1*)fKStar1dHeavyCfKStarSidePos->GetHeavyCf()->Clone(tRatioNameKStarSide);
  fKStar1dHeavyKStarSidePosNegRatio->SetTitle(tRatioNameKStarSide);
  fKStar1dHeavyKStarSidePosNegRatio->Divide(fKStar1dHeavyCfKStarSideNeg->GetHeavyCf());

  //-----KStarLong
  fKStar2dHeavyCfKStarLong->Rebin(aRebinFactor);

  //refresh objects which derive from fKStar2dHeavyCfKStarLong
  fKStar1dHeavyCfKStarLongNeg = fKStar2dHeavyCfKStarLong->GetDaughterHeavyCf(0);
  fKStar1dHeavyCfKStarLongPos = fKStar2dHeavyCfKStarLong->GetDaughterHeavyCf(1);

  TString tRatioBaseNameKStarLong = "KStar1dHeavyKStarLongPosNegRatio";
  TString tRatioNameKStarLong = tRatioBaseNameKStarLong + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  fKStar1dHeavyKStarLongPosNegRatio = (TH1*)fKStar1dHeavyCfKStarLongPos->GetHeavyCf()->Clone(tRatioNameKStarLong);
  fKStar1dHeavyKStarLongPosNegRatio->SetTitle(tRatioNameKStarLong);
  fKStar1dHeavyKStarLongPosNegRatio->Divide(fKStar1dHeavyCfKStarLongNeg->GetHeavyCf());
  
}


//________________________________________________________________________________________________________________
void Analysis::DrawKStar2dHeavyCfKStarOutRatio(TPad* aPad, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();

  TString tTitleBase = "Cf(KStarOut+)/Cf(KStarOut-) ";
  TString tTitle = tTitleBase + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  fKStar1dHeavyKStarOutPosNegRatio->SetTitle(tTitle);

  fKStar1dHeavyKStarOutPosNegRatio->GetXaxis()->SetTitle("k* (GeV/c)");
  fKStar1dHeavyKStarOutPosNegRatio->GetYaxis()->SetTitle("Cf(k*_{Out}+)/Cf(k*_{Out}-)");

  fKStar1dHeavyKStarOutPosNegRatio->GetYaxis()->SetRangeUser(0.95,1.11);

  fKStar1dHeavyKStarOutPosNegRatio->SetMarkerStyle(aMarkerStyle);
  fKStar1dHeavyKStarOutPosNegRatio->SetMarkerColor(aMarkerColor);
  fKStar1dHeavyKStarOutPosNegRatio->SetLineColor(aMarkerColor);

  fKStar1dHeavyKStarOutPosNegRatio->Draw(aOption);

  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);
  line->Draw();

  //---------------------------------------------
  TPaveText *tPaveText = new TPaveText(0.15,0.65,0.55,0.85,"NDC");
  tPaveText->SetFillColor(0);
  tPaveText->SetTextAlign(12);

  TString tPart1Name = TString(cRootParticleTags[fParticleTypes[0]]);
  TString tPart2Name = TString(cRootParticleTags[fParticleTypes[1]]);

  TString tInfo = "Particle Ordering:";
  TString tPart1Info = "P1 = " + tPart1Name;
  TString tPart2Info = "P2 = " + tPart2Name;

  tPaveText->AddText(tInfo);
  tPaveText->AddText(tPart1Info);
  tPaveText->AddText(tPart2Info);
  tPaveText->Draw();
}

//________________________________________________________________________________________________________________
void Analysis::DrawKStar2dHeavyCfKStarSideRatio(TPad* aPad, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();

  TString tTitleBase = "Cf(KStarSide+)/Cf(KStarSide-) ";
  TString tTitle = tTitleBase + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  fKStar1dHeavyKStarSidePosNegRatio->SetTitle(tTitle);

  fKStar1dHeavyKStarSidePosNegRatio->GetXaxis()->SetTitle("k* (GeV/c)");
  fKStar1dHeavyKStarSidePosNegRatio->GetYaxis()->SetTitle("Cf(k*_{Side}+)/Cf(k*_{Side}-)");

  fKStar1dHeavyKStarSidePosNegRatio->GetYaxis()->SetRangeUser(0.95,1.11);

  fKStar1dHeavyKStarSidePosNegRatio->SetMarkerStyle(aMarkerStyle);
  fKStar1dHeavyKStarSidePosNegRatio->SetMarkerColor(aMarkerColor);
  fKStar1dHeavyKStarSidePosNegRatio->SetLineColor(aMarkerColor);

  fKStar1dHeavyKStarSidePosNegRatio->Draw(aOption);

  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);
  line->Draw();

  //---------------------------------------------
  TPaveText *tPaveText = new TPaveText(0.15,0.65,0.55,0.85,"NDC");
  tPaveText->SetFillColor(0);
  tPaveText->SetTextAlign(12);

  TString tPart1Name = TString(cRootParticleTags[fParticleTypes[0]]);
  TString tPart2Name = TString(cRootParticleTags[fParticleTypes[1]]);

  TString tInfo = "Particle Ordering:";
  TString tPart1Info = "P1 = " + tPart1Name;
  TString tPart2Info = "P2 = " + tPart2Name;

  tPaveText->AddText(tInfo);
  tPaveText->AddText(tPart1Info);
  tPaveText->AddText(tPart2Info);
  tPaveText->Draw();
}

//________________________________________________________________________________________________________________
void Analysis::DrawKStar2dHeavyCfKStarLongRatio(TPad* aPad, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();

  TString tTitleBase = "Cf(KStarLong+)/Cf(KStarLong-) ";
  TString tTitle = tTitleBase + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  fKStar1dHeavyKStarLongPosNegRatio->SetTitle(tTitle);

  fKStar1dHeavyKStarLongPosNegRatio->GetXaxis()->SetTitle("k* (GeV/c)");
  fKStar1dHeavyKStarLongPosNegRatio->GetYaxis()->SetTitle("Cf(k*_{Long}+)/Cf(k*_{Long}-)");

  fKStar1dHeavyKStarLongPosNegRatio->GetYaxis()->SetRangeUser(0.95,1.11);

  fKStar1dHeavyKStarLongPosNegRatio->SetMarkerStyle(aMarkerStyle);
  fKStar1dHeavyKStarLongPosNegRatio->SetMarkerColor(aMarkerColor);
  fKStar1dHeavyKStarLongPosNegRatio->SetLineColor(aMarkerColor);

  fKStar1dHeavyKStarLongPosNegRatio->Draw(aOption);

  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);
  line->Draw();

  //---------------------------------------------
  TPaveText *tPaveText = new TPaveText(0.15,0.65,0.55,0.85,"NDC");
  tPaveText->SetFillColor(0);
  tPaveText->SetTextAlign(12);

  TString tPart1Name = TString(cRootParticleTags[fParticleTypes[0]]);
  TString tPart2Name = TString(cRootParticleTags[fParticleTypes[1]]);

  TString tInfo = "Particle Ordering:";
  TString tPart1Info = "P1 = " + tPart1Name;
  TString tPart2Info = "P2 = " + tPart2Name;

  tPaveText->AddText(tInfo);
  tPaveText->AddText(tPart1Info);
  tPaveText->AddText(tPart2Info);
  tPaveText->Draw();
}



//________________________________________________________________________________________________________________
void Analysis::DrawKStar2dHeavyCfRatios(TPad* aPad)
{
  aPad->cd();

  TString tTitleBase = "Cf+/Cf- ";
  TString tTitle = tTitleBase + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  fKStar1dHeavyKStarOutPosNegRatio->SetTitle(tTitle);

  fKStar1dHeavyKStarOutPosNegRatio->GetXaxis()->SetTitle("k* (GeV/c)");
  fKStar1dHeavyKStarOutPosNegRatio->GetYaxis()->SetTitle("Cf+/Cf-");

  fKStar1dHeavyKStarOutPosNegRatio->GetYaxis()->SetRangeUser(0.94,1.11);

  //-----
  fKStar1dHeavyKStarOutPosNegRatio->SetMarkerStyle(20);
  fKStar1dHeavyKStarOutPosNegRatio->SetMarkerColor(1);
  fKStar1dHeavyKStarOutPosNegRatio->SetLineColor(1);
  fKStar1dHeavyKStarOutPosNegRatio->SetMarkerSize(0.50);
  fKStar1dHeavyKStarOutPosNegRatio->GetXaxis()->SetRangeUser(0.,0.5);
  fKStar1dHeavyKStarOutPosNegRatio->Draw();

  fKStar1dHeavyKStarSidePosNegRatio->SetMarkerStyle(20);
  fKStar1dHeavyKStarSidePosNegRatio->SetMarkerColor(2);
  fKStar1dHeavyKStarSidePosNegRatio->SetLineColor(2);
  fKStar1dHeavyKStarSidePosNegRatio->SetMarkerSize(0.50);
  fKStar1dHeavyKStarSidePosNegRatio->Draw("same");

  fKStar1dHeavyKStarLongPosNegRatio->SetMarkerStyle(20);
  fKStar1dHeavyKStarLongPosNegRatio->SetMarkerColor(3);
  fKStar1dHeavyKStarLongPosNegRatio->SetLineColor(3);
  fKStar1dHeavyKStarLongPosNegRatio->SetMarkerSize(0.50);
  fKStar1dHeavyKStarLongPosNegRatio->Draw("same");

  fKStar1dHeavyKStarOutPosNegRatio->Draw("same");
  //-----
  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);
  line->Draw();

  //---------------------------------------------
  TPaveText *tPaveText = new TPaveText(0.15,0.65,0.35,0.85,"NDC");
  tPaveText->SetFillColor(0);
  tPaveText->SetTextAlign(12);

  TString tPart1Name = TString(cRootParticleTags[fParticleTypes[0]]);
  TString tPart2Name = TString(cRootParticleTags[fParticleTypes[1]]);

  TString tInfo = "Particle Ordering:";
  TString tPart1Info = "P1 = " + tPart1Name;
  TString tPart2Info = "P2 = " + tPart2Name;

  tPaveText->AddText(tInfo);
  tPaveText->AddText(tPart1Info);
  tPaveText->AddText(tPart2Info);
  tPaveText->Draw();


  //---------------------------------------------
  TLegend *tLeg = new TLegend(0.65,0.65,0.85,0.85);
    tLeg->AddEntry(fKStar1dHeavyKStarOutPosNegRatio,"Out","p");
    tLeg->AddEntry(fKStar1dHeavyKStarSidePosNegRatio,"Side","p");
    tLeg->AddEntry(fKStar1dHeavyKStarLongPosNegRatio,"Long","p");
    tLeg->Draw();

}






//________________________________________________________________________________________________________________
void Analysis::BuildSepHeavyCfs(DaughterPairType aDaughterPairType, double aMinNorm, double aMaxNorm)
{
  vector<Cf2dLite*> tTempCf2dLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //make sure everything is updated
    fPartialAnalysisCollection[iAnaly]->BuildSepCfs(aDaughterPairType,aMinNorm,aMaxNorm);

    //make sure the Cf exists
    assert(fPartialAnalysisCollection[iAnaly]->GetSepCfs(aDaughterPairType));

    tTempCf2dLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetSepCfs(aDaughterPairType));

  }

  TString tCfDaughterBaseName = "SepCfs";
  TString tCfDaughterName = tCfDaughterBaseName + cDaughterPairTags[aDaughterPairType] + "_" + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType] + "_";

  Cf2dHeavy *tCf2dHeavy = new Cf2dHeavy(tCfDaughterName,tTempCf2dLiteCollection,aMinNorm,aMaxNorm);
  fSepHeavyCfs[aDaughterPairType] = tCf2dHeavy;

}




//________________________________________________________________________________________________________________
void Analysis::BuildAllSepHeavyCfs(double aMinNorm, double aMaxNorm)
{
  int tNCfs = fDaughterPairTypes.size();

  for(int i=0; i<tNCfs; i++)
  {
    BuildSepHeavyCfs(fDaughterPairTypes[i],aMinNorm,aMaxNorm);
  }
}



//________________________________________________________________________________________________________________
Cf2dHeavy* Analysis::GetSepHeavyCfs(DaughterPairType aDaughterPairType)
{
  return fSepHeavyCfs[aDaughterPairType];
}


//________________________________________________________________________________________________________________
void Analysis::DrawSepHeavyCfs(DaughterPairType aDaughterPairType, TPad* aPad)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  aPad->Divide(8,1,0,0);

  //----------------------------------------------------------------------
  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  double YRangeMin = 0.8;
  double YRangeMax = 2.0;
  //----------------------------------------------------------------------

  Cf2dHeavy* tCf2dHeavy = fSepHeavyCfs[aDaughterPairType];
  vector<CfHeavy*> tDerivedHeavyCfs = tCf2dHeavy->GetAllDaughterHeavyCfs();

  int tNTPCbins = tDerivedHeavyCfs.size();

  assert(tNTPCbins == 8);

  vector<TH1*> tHistosToDraw(tNTPCbins);
  TString tTitle;

  for(int i=0; i<tNTPCbins; i++)
  {
    tHistosToDraw[i] = tDerivedHeavyCfs[i]->GetHeavyCf();

    tTitle = GetDaughtersHistoTitle(aDaughterPairType) + " TPC";
    tTitle += i;

    tHistosToDraw[i]->SetTitle(tTitle);
    tHistosToDraw[i]->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
    tHistosToDraw[i]->SetMarkerStyle(20);
    tHistosToDraw[i]->SetMarkerSize(0.5);
  }

  //------------------------------------------------------------------


  for(int i=0; i<tNTPCbins; i++)
  {
    int aPadInt = i+1;
    aPad->cd(aPadInt);

    tHistosToDraw[i]->Draw();
    line->Draw();
  }

}

//________________________________________________________________________________________________________________
void Analysis::BuildAvgSepCowSailHeavyCfs(DaughterPairType aDaughterPairType, double aMinNorm, double aMaxNorm)
{
  vector<Cf2dLite*> tTempCf2dLiteCollection;

  for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    //make sure everything is updated
    fPartialAnalysisCollection[iAnaly]->BuildAvgSepCowSailCfs(aDaughterPairType,aMinNorm,aMaxNorm);

    //make sure the Cf exists
    assert(fPartialAnalysisCollection[iAnaly]->GetAvgSepCowSailCfs(aDaughterPairType));

    tTempCf2dLiteCollection.push_back(fPartialAnalysisCollection[iAnaly]->GetAvgSepCowSailCfs(aDaughterPairType));

  }

  TString tCfDaughterBaseName = "AvgSepCowSailCfs";
  TString tCfDaughterName = tCfDaughterBaseName + cDaughterPairTags[aDaughterPairType] + "_" + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType] + "_";

  Cf2dHeavy *tCf2dHeavy = new Cf2dHeavy(tCfDaughterName,tTempCf2dLiteCollection,aMinNorm,aMaxNorm);
  fAvgSepCowSailHeavyCfs[aDaughterPairType] = tCf2dHeavy;

}




//________________________________________________________________________________________________________________
void Analysis::BuildAllAvgSepCowSailHeavyCfs(double aMinNorm, double aMaxNorm)
{
  int tNCfs = fDaughterPairTypes.size();

  for(int i=0; i<tNCfs; i++)
  {
    BuildAvgSepCowSailHeavyCfs(fDaughterPairTypes[i],aMinNorm,aMaxNorm);
  }
}



//________________________________________________________________________________________________________________
Cf2dHeavy* Analysis::GetAvgSepCowSailHeavyCfs(DaughterPairType aDaughterPairType)
{
  return fAvgSepCowSailHeavyCfs[aDaughterPairType];
}


//________________________________________________________________________________________________________________
void Analysis::DrawAvgSepCowSailHeavyCfs(DaughterPairType aDaughterPairType, TPad* aPad)
{
  aPad->cd();

  //----------------------------------------------------------------------
  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  double YRangeMin = -0.5;
  double YRangeMax = 5.0;

  double aMarkerSize = 0.50;
  double aMarkerStyle = 20;
  //----------------------------------------------------------------------

  Cf2dHeavy* tCf2dHeavy = fAvgSepCowSailHeavyCfs[aDaughterPairType];
  vector<CfHeavy*> tDerivedHeavyCfs = tCf2dHeavy->GetAllDaughterHeavyCfs();

  int tNProjectionBins = tDerivedHeavyCfs.size();

  assert(tNProjectionBins == 2); //bin1 = momentum(particle2) > momentum(particle1)
                                 //bin2 = momentum(particle2) < momentum(particle1)

  vector<TH1*> tHistosToDraw(tNProjectionBins);
  TString tTitle;

  for(int i=0; i<tNProjectionBins; i++)
  {
    tHistosToDraw[i] = tDerivedHeavyCfs[i]->GetHeavyCf();

    tTitle = GetDaughtersHistoTitle(aDaughterPairType);
    tHistosToDraw[i]->SetTitle(tTitle);
    tHistosToDraw[i]->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
    tHistosToDraw[i]->SetMarkerStyle(aMarkerStyle);
    tHistosToDraw[i]->SetMarkerSize(aMarkerSize);
    tHistosToDraw[i]->SetMarkerColor(i+1);
  }

  tHistosToDraw[0]->Draw();
  tHistosToDraw[1]->Draw("same");
  line->Draw();

  //------------------------------------------------------

  vector<ParticleType> tDaughterParticleTypes =  GetCorrectDaughterParticleTypes(aDaughterPairType);

  TString tLegEntry1;
  TString tLegEntry2;

  TString tDaughterParticle1Name;
  TString tDaughterParticle2Name;

  if(tDaughterParticleTypes.size() == 1)
  {
    tDaughterParticle1Name = "P(" + TString(cRootParticleTags[tDaughterParticleTypes[0]]) + "(" + TString(cRootParticleTags[fParticleTypes[0]]) + ")";
    tDaughterParticle2Name = "P(" + TString(cRootParticleTags[fParticleTypes[1]]) + ")";
  }

  else if(tDaughterParticleTypes.size() == 2)
  {
    tDaughterParticle1Name = "P(" + TString(cRootParticleTags[tDaughterParticleTypes[0]]) + "(" + TString(cRootParticleTags[fParticleTypes[0]]) + ")";
    tDaughterParticle2Name = "P(" + TString(cRootParticleTags[tDaughterParticleTypes[1]]) + "(" + TString(cRootParticleTags[fParticleTypes[1]]) + ")";
  }


  tLegEntry1 = tDaughterParticle1Name + " < " + tDaughterParticle2Name;
  tLegEntry2 = tDaughterParticle1Name + " > " + tDaughterParticle2Name;

  TLegend *tLeg = new TLegend(0.7,0.7,0.9,0.9);
    tLeg->AddEntry(tHistosToDraw[0],tLegEntry1,"p");
    tLeg->AddEntry(tHistosToDraw[1],tLegEntry2,"p");
    tLeg->Draw();



}


//________________________________________________________________________________________________________________
void Analysis::BuildPurityCollection()
{
  if(fPurityCollection.size() != 0) {fPurityCollection.clear();}

  for(unsigned int i=0; i<fParticleTypes.size(); i++)
  {
    if( (fParticleTypes[i]==kLam) || (fParticleTypes[i]==kALam) || (fParticleTypes[i]==kK0) || (fParticleTypes[i]==kXi) || (fParticleTypes[i]==kAXi) )
    {
      vector<TH1*> tTempVec(0);
      for(int iAnaly=0; iAnaly<fNPartialAnalysis; iAnaly++)
      {
        TH1* tPurityHisto = fPartialAnalysisCollection[iAnaly]->GetPurityHisto(fParticleTypes[i]);
        tTempVec.push_back(tPurityHisto);
      }
      TString tBaseName = TString(cParticleTags[fParticleTypes[i]]);
      TString tName = tBaseName + "Purity_" + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

      Purity* tPurity = new Purity(tName,fParticleTypes[i],tTempVec);

      fPurityCollection.push_back(tPurity);
    }

    else{continue;}

  }


}



//________________________________________________________________________________________________________________
void Analysis::DrawAllPurityHistos(TPad* aPad)
{
  int tNPurityHistos = fPurityCollection.size();

  aPad->Divide(1,tNPurityHistos);

  for(int i=0; i<tNPurityHistos; i++)
  {
    fPurityCollection[i]->DrawPurityAndBgd((TPad*)aPad->cd(i+1));
  }

}


//________________________________________________________________________________________________________________
double Analysis::GetPurity(ParticleType aV0Type)
{
  double tReturnValue = 0.;
  assert(fPurityCollection.size() != 0);
  for(unsigned int i=0; i<fPurityCollection.size(); i++)
  {
    if(fPurityCollection[i]->GetParticleType() == aV0Type)
    {
      tReturnValue = fPurityCollection[i]->GetPurity();
      return tReturnValue;
    }
  }

  cout << "ERROR: Analysis::GetPurity: for aV0Type = " << aV0Type << " in fAnalysisType = " << fAnalysisType << " no purity was found!" << endl;
  assert(0);
  return tReturnValue;
}

//________________________________________________________________________________________________________________
TH1* Analysis::GetCombinedPurityHisto(ParticleType aV0Type)
{
  TH1* tReturnHisto;
  assert(fPurityCollection.size() != 0);
  for(unsigned int i=0; i<fPurityCollection.size(); i++)
  {
    if(fPurityCollection[i]->GetParticleType() == aV0Type)
    {
      tReturnHisto = fPurityCollection[i]->GetCombinedPurity();
      return tReturnHisto;
    }
  }

  cout << "ERROR: Analysis::GetPurity: for aV0Type = " << aV0Type << " in fAnalysisType = " << fAnalysisType << " no purity was found!" << endl;
  assert(0);
  return tReturnHisto;
}

//________________________________________________________________________________________________________________
Purity* Analysis::GetPurityObject(ParticleType aV0Type)
{
  Purity* tReturnPurity;
  assert(fPurityCollection.size() != 0);
  for(unsigned int i=0; i<fPurityCollection.size(); i++)
  {
    if(fPurityCollection[i]->GetParticleType() == aV0Type)
    {
      tReturnPurity = fPurityCollection[i];
      return tReturnPurity;
    }
  }

  cout << "ERROR: Analysis::GetPurity: for aV0Type = " << aV0Type << " in fAnalysisType = " << fAnalysisType << " no purity was found!" << endl;
  assert(0);
  return tReturnPurity;
}


//________________________________________________________________________________________________________________
void Analysis::OutputPassFailInfo()
{
  cout << fAnalysisName << " PassFailInfo: " << endl;
  cout << "\t" << "NEventsPass = " << fNEventsPass << endl;
  cout << "\t" << "NEventsFail = " << fNEventsFail << endl;
  cout << "\t" << "NPart1Pass = " << fNPart1Pass << endl;
  cout << "\t" << "NPart1Fail = " << fNPart1Fail << endl;
  cout << "\t" << "NPart2Pass = " << fNPart2Pass << endl;
  cout << "\t" << "NPart2Fail = " << fNPart2Fail << endl;
  cout << "\t" << "NKStarNumEntries = " << fNKStarNumEntries << endl << endl;
}

//________________________________________________________________________________________________________________
void Analysis::DrawPart1MassFail(TPad* aPad, bool aDrawWideRangeToo)
{
  assert(fAnalysisRunType != kTrainSys);  //TrainSys analyses DO NOT include FAIL cut monitors

  aPad->cd();

  if(aDrawWideRangeToo)
  {
    aPad->Divide(2,1,0.00001,0.00001);

    aPad->cd(2);
    gPad->SetMargin(0.05,0.05,0.1,0.1);

  
    fPart1MassFail->GetXaxis()->SetRange(fPart1MassFail->FindBin(1.),fPart1MassFail->FindBin(2.));
    fPart1MassFail->DrawCopy();
  }

  //--------------------------------
  gBgReject = true;

  gBgFitLow[0] = 1.09;
  gBgFitLow[1] = 1.108;

  gBgFitHigh[0] = 1.124;
  gBgFitHigh[1] = 1.150;

  double tROI[2];
  tROI[0] = LambdaMass-0.0038;
  tROI[1] = LambdaMass+0.0038;

  TF1 *fitBgd = new TF1("fitBgd",BgFitFunction,fPart1MassFail->GetBinLowEdge(1),fPart1MassFail->GetBinLowEdge(fPart1MassFail->GetNbinsX()+1),5);  //fit over entire range
  fPart1MassFail->Fit("fitBgd","0q");

  //--------------------------------
  gBgReject = false;

  char buffer[50];
  sprintf(buffer, "fitBgd_%s",fPart1MassFail->GetName());
  TF1 *fitBgd2 = new TF1(buffer,BgFitFunction,fPart1MassFail->GetBinLowEdge(1),fPart1MassFail->GetBinLowEdge(fPart1MassFail->GetNbinsX()+1),5);
  fitBgd2->SetParameters(fitBgd->GetParameters());

  //--------------------------------------------------------------------------------------------
  double tBgd = fitBgd2->Integral(tROI[0],tROI[1]);
  tBgd /= fPart1MassFail->GetBinWidth(0);  //divide by bin size
  cout << fPart1MassFail->GetName() << ": " << "Bgd = " << tBgd << endl;
  //-----
  double tSigpbgd = fPart1MassFail->Integral(fPart1MassFail->FindBin(tROI[0]),fPart1MassFail->FindBin(tROI[1]));
  cout << fPart1MassFail->GetName() << ": " << "Sig+Bgd = " << tSigpbgd << endl;
  //-----
  double tSig = tSigpbgd-tBgd;
  cout << fPart1MassFail->GetName() << ": " << "Sig = " << tSig << endl;
  //--------------------------------------------------------------------------------------------

  fPart1MassFail->GetXaxis()->SetRange(fPart1MassFail->FindBin(gBgFitLow[0]),fPart1MassFail->FindBin(gBgFitHigh[1]));

  double tHistoMaxValue = fPart1MassFail->GetMaximum();
  double tYRangeMin = fPart1MassFail->GetMinimum();

  TLine* lROImin = new TLine(tROI[0],tYRangeMin,tROI[0],tHistoMaxValue);
  TLine* lROImax = new TLine(tROI[1],tYRangeMin,tROI[1],tHistoMaxValue);

  if(aDrawWideRangeToo)
  {
    aPad->cd(1);
    gPad->SetMargin(0.05,0.05,0.1,0.1);
  }
  fPart1MassFail->Draw();
  fitBgd2->Draw("same");
  lROImin->Draw();
  lROImax->Draw();


  //--------------------------------------------------------------------------------------------

  TPaveText *tPaveText = new TPaveText(0.15,0.65,0.35,0.85,"NDC");
  tPaveText->SetFillColor(0);
  tPaveText->SetTextAlign(12);

  TString tAnalysisText = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  TString tHeaderText = TString(cRootParticleTags[fParticleTypes[0]]) + "_Fail";

  TString tText = "NLost = ";
  int tSigReduced = tSig/1000000;
  tText+=tSigReduced;
  tText+=" M";

  tPaveText->AddText(tAnalysisText);
  tPaveText->AddText(tHeaderText);
  tPaveText->AddText(tText);

  tPaveText->Draw();

}

//________________________________________________________________________________________________________________
void Analysis::GetMCKchPurity(bool aBeforePairCut)
{
  TH1* tPIdHisto = fPartialAnalysisCollection[0]->GetMCKchPurityHisto(aBeforePairCut);

  for(int iAnaly=1; iAnaly<fNPartialAnalysis; iAnaly++)
  {
    tPIdHisto->Add(fPartialAnalysisCollection[iAnaly]->GetMCKchPurityHisto(aBeforePairCut));
  }

  double tKchCounts = tPIdHisto->GetBinContent(tPIdHisto->FindBin(321));
  double tTotalCounts = tPIdHisto->GetEntries();
  double tKchPurity = tKchCounts/tTotalCounts;

  if(aBeforePairCut) {cout << TString(cParticleTags[fParticleTypes[1]]) << " Purity BeforePairCut = " << tKchPurity << endl;}
  else{cout << TString(cParticleTags[fParticleTypes[1]]) << " Purity AfterPairCut = " << tKchPurity << endl;}

}


//________________________________________________________________________________________________________________
TH1* Analysis::GetMassAssumingK0ShortHypothesis()
{
  //I will want this only for (Anti)Lambda, which is always particle1
  TString tHistoName = TString("K0ShortMass_") + TString(cParticleTags[fParticleTypes[0]]) + TString("_Pass");
  TH1* tReturnHist = SimpleAddTH1Collection(tHistoName);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1* Analysis::GetMassAssumingLambdaHypothesis()
{
  //I will want this only for K0Short, which is always particle2
  TString tHistoName = TString("LambdaMass_") + TString(cParticleTags[fParticleTypes[1]]) + TString("_Pass");
  TH1* tReturnHist = SimpleAddTH1Collection(tHistoName);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1* Analysis::GetMassAssumingAntiLambdaHypothesis()
{
  //I will want this only for K0Short, which is always particle2
  TString tHistoName = TString("AntiLambdaMass_") + TString(cParticleTags[fParticleTypes[1]]) + TString("_Pass");
  TH1* tReturnHist = SimpleAddTH1Collection(tHistoName);
  return tReturnHist;
}



//________________________________________________________________________________________________________________
TCanvas* Analysis::DrawKchdEdx(ParticleType aKchType, bool aLogz)
{
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);


  assert(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || fAnalysisType==kLamKchM || fAnalysisType==kALamKchP);

  TString tCanvasName = TString::Format("canDrawKchdEdx_%s%s_%s", cAnalysisBaseTags[fAnalysisType], 
                                        cCentralityTags[fCentralityType], cParticleTags[aKchType]);
  if(aLogz) tCanvasName += TString("_Logz");
  TCanvas* tReturnCan = new TCanvas(tCanvasName, tCanvasName);
  tReturnCan->SetRightMargin(0.11); //0.1 is standard
  tReturnCan->cd();

  TString tHistName = TString::Format("TPCdEdx_%s_Pass", cParticleTags[aKchType]);
  TString tHistNewName = TString::Format("dEdX_%s%s_%s", cAnalysisBaseTags[fAnalysisType], 
                                        cCentralityTags[fCentralityType], cParticleTags[aKchType]);
  TH2* tCombineddEdX = (TH2*)fPartialAnalysisCollection[0]->Get2dHisto(tHistName, tHistNewName);
  if(!tCombineddEdX->GetSumw2N()) tCombineddEdX->Sumw2();

  for(unsigned int i=1; i<fPartialAnalysisCollection.size(); i++) 
  {
    TString tHistNewName = TString::Format("dEdX_%s%s_%s_%d", cAnalysisBaseTags[fAnalysisType], 
                                          cCentralityTags[fCentralityType], cParticleTags[aKchType], i);
    tCombineddEdX->Add((TH2*)fPartialAnalysisCollection[0]->Get2dHisto(tHistName, tHistNewName));
  }

  if(aLogz)
  {
    gPad->SetLogz();
    tCombineddEdX->GetXaxis()->SetRangeUser(0.1, 2.05);
  }
  else tCombineddEdX->GetXaxis()->SetRangeUser(0., 2.05);

  tCombineddEdX->GetXaxis()->SetTitle("#it{p} (GeV/#it{c})");
    tCombineddEdX->GetXaxis()->SetTitleSize(0.04);
    tCombineddEdX->GetXaxis()->SetTitleOffset(1.1);

  tCombineddEdX->GetYaxis()->SetTitle("TPC d#it{E}/d#it{x}");
    tCombineddEdX->GetYaxis()->SetTitleSize(0.04);
    tCombineddEdX->GetYaxis()->SetTitleOffset(1.2);

  tCombineddEdX->Draw("colz");

  TLatex *tTex = new TLatex(1.2,450,"Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.044);
  tTex->SetLineWidth(2);
  tTex->Draw();

  tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetTextFont(62);
  tTex->SetTextSize(0.06);
  tTex->DrawLatex(1.2, 400, TString::Format("%s (%s)", cRootParticleTags[aKchType], cPrettyCentralityTags[fCentralityType]));

  return tReturnCan;
}



