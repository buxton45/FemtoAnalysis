////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  myAliFemtoModelCorrFctnKStar                                              //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
#ifdef __ROOT__
  /// \cond CLASSIMP
  ClassImp(myAliFemtoModelCorrFctnKStar);
  /// \endcond
#endif

#include "myAliFemtoModelCorrFctnKStar.h"
#include "AliFemtoModelHiddenInfo.h"
#include <TH1D.h>
#include <TH2D.h>

//_________________________________________
myAliFemtoModelCorrFctnKStar::myAliFemtoModelCorrFctnKStar() :
  AliFemtoModelCorrFctn(),
  fRemoveMisidentified(false),
  fAnalysisType(kLamK0),
  fPart1ID(3122),
  fPart2ID(310),
  fNumTrueUnitWeights(0),
  fNumTrueIdealUnitWeights(0),
  fKTrueKRecSame(0),
  fKTrueKRecMixed(0),
  fKTrueKRecRotSame(0),
  fKTrueKRecRotMixed(0)
{
  //default constructor
  fNumTrueUnitWeights = new TH1D("NumTrueUnitWeights","NumTrueUnitWeights",200,0.,1.);
  fNumTrueIdealUnitWeights = new TH1D("NumTrueIdealUnitWeights","NumTrueIdealUnitWeights",200,0.,1.);

  fKTrueKRecSame = new TH2D("fKTrueKRecSame","fKTrueKRecSame",200,0.,1.,200,0.,1.);
  fKTrueKRecMixed = new TH2D("fKTrueKRecMixed","fKTrueKRecMixed",200,0.,1.,200,0.,1.);
  fKTrueKRecRotSame = new TH2D("fKTrueKRecRotSame","fKTrueKRecRotSame",200,0.,1.,200,-0.5,0.5);
  fKTrueKRecRotMixed = new TH2D("fKTrueKRecRotMixed","fKTrueKRecRotMixed",200,0.,1.,200,-0.5,0.5);

  //-----------------

  fNumTrueUnitWeights->Sumw2();
  fNumTrueIdealUnitWeights->Sumw2();

  fKTrueKRecSame->Sumw2();
  fKTrueKRecMixed->Sumw2();
  fKTrueKRecRotSame->Sumw2();
  fKTrueKRecRotMixed->Sumw2();
}

myAliFemtoModelCorrFctnKStar::myAliFemtoModelCorrFctnKStar(const char *title, int aNbins, double aKStarLo, double aKStarHi) :
  AliFemtoModelCorrFctn(title,aNbins,aKStarLo,aKStarHi),
  fRemoveMisidentified(false),
  fAnalysisType(kLamK0),
  fPart1ID(3122),
  fPart2ID(310),
  fNumTrueUnitWeights(0),
  fNumTrueIdealUnitWeights(0),
  fKTrueKRecSame(0),
  fKTrueKRecMixed(0),
  fKTrueKRecRotSame(0),
  fKTrueKRecRotMixed(0)
{
  //normal constructor
  char buf[100];
  snprintf(buf , 100,  "NumTrueUnitWeights%s", title);
  fNumTrueUnitWeights = new TH1D(buf,buf,aNbins,aKStarLo,aKStarHi);
  snprintf(buf , 100,  "NumTrueIdealUnitWeights%s", title);
  fNumTrueIdealUnitWeights = new TH1D(buf,buf,aNbins,aKStarLo,aKStarHi);

  double tYRange = aKStarHi-aKStarLo;
  double tRotYMin = -0.5*tYRange;
  double tRotYMax = 0.5*tYRange;

  snprintf(buf , 100,  "fKTrueKRecSame%s", title);
  fKTrueKRecSame = new TH2D(buf,buf,aNbins,aKStarLo,aKStarHi,aNbins,aKStarLo,aKStarHi);
  snprintf(buf , 100,  "fKTrueKRecMixed%s", title);
  fKTrueKRecMixed = new TH2D(buf,buf,aNbins,aKStarLo,aKStarHi,aNbins,aKStarLo,aKStarHi);
  snprintf(buf , 100,  "fKTrueKRecRotSame%s", title);
  fKTrueKRecRotSame = new TH2D(buf,buf,aNbins,aKStarLo,aKStarHi,aNbins,tRotYMin,tRotYMax);
  snprintf(buf , 100,  "fKTrueKRecRotMixed%s", title);
  fKTrueKRecRotMixed = new TH2D(buf,buf,aNbins,aKStarLo,aKStarHi,aNbins,tRotYMin,tRotYMax);


  //-----------------

  fNumTrueUnitWeights->Sumw2();
  fNumTrueIdealUnitWeights->Sumw2();

  fKTrueKRecSame->Sumw2();
  fKTrueKRecMixed->Sumw2();
  fKTrueKRecRotSame->Sumw2();
  fKTrueKRecRotMixed->Sumw2();
}

//_________________________________________
myAliFemtoModelCorrFctnKStar::~myAliFemtoModelCorrFctnKStar()
{
  //destructor
}

//_________________________________________
myAliFemtoModelCorrFctnKStar::myAliFemtoModelCorrFctnKStar(const myAliFemtoModelCorrFctnKStar& aCorrFctn) :
  AliFemtoModelCorrFctn(aCorrFctn),
  fRemoveMisidentified(aCorrFctn.fRemoveMisidentified),
  fAnalysisType(aCorrFctn.fAnalysisType),
  fPart1ID(aCorrFctn.fPart1ID),
  fPart2ID(aCorrFctn.fPart2ID)
{
  //copy constructor
  if (aCorrFctn.fNumTrueUnitWeights)
    fNumTrueUnitWeights = new TH1D(*(aCorrFctn.fNumTrueUnitWeights));
  else fNumTrueUnitWeights = 0;

  if (aCorrFctn.fNumTrueIdealUnitWeights)
    fNumTrueIdealUnitWeights = new TH1D(*(aCorrFctn.fNumTrueIdealUnitWeights));
  else fNumTrueIdealUnitWeights = 0;

  if (aCorrFctn.fKTrueKRecSame)
    fKTrueKRecSame = new TH2D(*(aCorrFctn.fKTrueKRecSame));
  else fKTrueKRecSame = 0;

  if (aCorrFctn.fKTrueKRecMixed)
    fKTrueKRecMixed = new TH2D(*(aCorrFctn.fKTrueKRecMixed));
  else fKTrueKRecMixed = 0;

  if (aCorrFctn.fKTrueKRecRotSame)
    fKTrueKRecRotSame = new TH2D(*(aCorrFctn.fKTrueKRecRotSame));
  else fKTrueKRecRotSame = 0;

  if (aCorrFctn.fKTrueKRecRotMixed)
    fKTrueKRecRotMixed = new TH2D(*(aCorrFctn.fKTrueKRecRotMixed));
  else fKTrueKRecRotMixed = 0;

}

//_________________________________________
myAliFemtoModelCorrFctnKStar& myAliFemtoModelCorrFctnKStar::operator=(const myAliFemtoModelCorrFctnKStar& aCorrFctn)
{
  //assignment operator
  if (this == &aCorrFctn) return *this;

  AliFemtoModelCorrFctn::operator=(aCorrFctn);

  fRemoveMisidentified = aCorrFctn.fRemoveMisidentified;
  fAnalysisType = aCorrFctn.fAnalysisType;
  fPart1ID = aCorrFctn.fPart1ID;
  fPart2ID = aCorrFctn.fPart2ID;

  if (aCorrFctn.fNumTrueUnitWeights)
    fNumTrueUnitWeights = new TH1D(*(aCorrFctn.fNumTrueUnitWeights));
  else fNumTrueUnitWeights = 0;

  if (aCorrFctn.fNumTrueIdealUnitWeights)
    fNumTrueIdealUnitWeights = new TH1D(*(aCorrFctn.fNumTrueIdealUnitWeights));
  else fNumTrueIdealUnitWeights = 0;

  if (aCorrFctn.fKTrueKRecSame)
    fKTrueKRecSame = new TH2D(*(aCorrFctn.fKTrueKRecSame));
  else fKTrueKRecSame = 0;

  if (aCorrFctn.fKTrueKRecMixed)
    fKTrueKRecMixed = new TH2D(*(aCorrFctn.fKTrueKRecMixed));
  else fKTrueKRecMixed = 0;

  if (aCorrFctn.fKTrueKRecRotSame)
    fKTrueKRecRotSame = new TH2D(*(aCorrFctn.fKTrueKRecRotSame));
  else fKTrueKRecRotSame = 0;

  if (aCorrFctn.fKTrueKRecRotMixed)
    fKTrueKRecRotMixed = new TH2D(*(aCorrFctn.fKTrueKRecRotMixed));
  else fKTrueKRecRotMixed = 0;

  return *this;
}


//_________________________________________
void myAliFemtoModelCorrFctnKStar::SetPIDs(AnalysisType aAnalysisType)
{
  int tPiP = 211; int tPiM = -211;
  int tK0s = 310;
  int tKchP = 321; int tKchM = -321;
  int tLam = 3122; int tALam = -3122;
  int tXi = 3312; int tAXi = -3122;

  if(aAnalysisType == kLamK0) {fPart1ID = tLam; fPart2ID = tK0s;}
  else if(aAnalysisType == kALamK0) {fPart1ID = tALam; fPart2ID = tK0s;}

  else if(aAnalysisType == kLamKchP) {fPart1ID = tLam; fPart2ID = tKchP;}
  else if(aAnalysisType == kALamKchP) {fPart1ID = tALam; fPart2ID = tKchP;}
  else if(aAnalysisType == kLamKchM) {fPart1ID = tLam; fPart2ID = tKchM;}
  else if(aAnalysisType == kALamKchM) {fPart1ID = tALam; fPart2ID = tKchM;}

  else if(aAnalysisType == kXiKchP) {fPart1ID = tXi; fPart2ID = tKchP;}
  else if(aAnalysisType == kAXiKchP) {fPart1ID = tAXi; fPart2ID = tKchP;}
  else if(aAnalysisType == kXiKchM) {fPart1ID = tXi; fPart2ID = tKchM;}
  else if(aAnalysisType == kAXiKchM) {fPart1ID = tAXi; fPart2ID = tKchM;}

  else if(aAnalysisType == kLamLam) {fPart1ID = tLam; fPart2ID = tLam;}
  else if(aAnalysisType == kALamALam) {fPart1ID = tALam; fPart2ID = tALam;}
  else if(aAnalysisType == kLamALam) {fPart1ID = tLam; fPart2ID = tALam;}

  else if(aAnalysisType == kLamPiP) {fPart1ID = tLam; fPart2ID = tPiP;}
  else if(aAnalysisType == kALamPiP) {fPart1ID = tALam; fPart2ID = tPiP;}
  else if(aAnalysisType == kLamPiM) {fPart1ID = tLam; fPart2ID = tPiM;}
  else if(aAnalysisType == kALamPiM) {fPart1ID = tALam; fPart2ID = tPiM;}

  else{fPart1ID = tLam; fPart2ID = tK0s;}
}

//_________________________________________
void myAliFemtoModelCorrFctnKStar::SetAnalysisType(int aType)
{
  fAnalysisType = static_cast<AnalysisType>(aType);
  SetPIDs(fAnalysisType);
}


//_________________________________________
bool myAliFemtoModelCorrFctnKStar::TestPIDs(int aPID1, int aPID2)
{
  if(aPID1==fPart1ID && aPID2==fPart2ID) return true;
  else return false;
}


//_________________________________________
double myAliFemtoModelCorrFctnKStar::GetKStarTrue(AliFemtoPair* aPair)
{
  double tVerySmall = std::numeric_limits < double >::min();

  // Generate a simple femtoscopic weight coming simple Lednicky equation
  // The weight is generated using the TrueMomentum in the HiddenInfo

  AliFemtoParticle *tPart1 = (AliFemtoParticle*)aPair->Track1();
  AliFemtoParticle *tPart2 = (AliFemtoParticle*)aPair->Track2();

  if(tPart1 != NULL && tPart2 != NULL)
  {
    AliFemtoModelHiddenInfo *tPart1HiddenInfo = (AliFemtoModelHiddenInfo*)tPart1->GetHiddenInfo();
    AliFemtoModelHiddenInfo *tPart2HiddenInfo = (AliFemtoModelHiddenInfo*)tPart2->GetHiddenInfo();

    if(tPart1HiddenInfo != NULL && tPart2HiddenInfo != NULL)
    {
      double px1 = tPart1HiddenInfo->GetTrueMomentum()->x();
      double py1 = tPart1HiddenInfo->GetTrueMomentum()->y();
      double pz1 = tPart1HiddenInfo->GetTrueMomentum()->z();
      double mass1 = tPart1HiddenInfo->GetMass();
      double E1 = std::sqrt(mass1*mass1 + px1*px1 + py1*py1 + pz1*pz1);
      if( (E1-mass1) < tVerySmall) return -999;  //p1 has zero momentum!

      double px2 = tPart2HiddenInfo->GetTrueMomentum()->x();
      double py2 = tPart2HiddenInfo->GetTrueMomentum()->y();
      double pz2 = tPart2HiddenInfo->GetTrueMomentum()->z();
      double mass2 = tPart2HiddenInfo->GetMass();
      double E2 = std::sqrt(mass2*mass2 + px2*px2 + py2*py2 + pz2*pz2);
      if( (E2-mass2) < tVerySmall) return -999;  //p2 has zero momentum!
      //------------------------------------------------------------

      if(fRemoveMisidentified)
      {
        bool tPass = TestPIDs(tPart1HiddenInfo->GetPDGPid(),tPart2HiddenInfo->GetPDGPid());
        if(!tPass) return -999;
      }


      //------------------------------------------------------------
      double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);

      double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);

      double tKStar = 0.5*sqrt(tQinvSq);
      return tKStar;
    }

  }

  return -999;
}

//_________________________________________
void myAliFemtoModelCorrFctnKStar::AddRealPair(AliFemtoPair* aPair)
{
  double tWeight;
  if(fManager) tWeight = fManager->GetWeight(aPair);
  else tWeight = 1.0;

  double tKStarTrue = GetKStarTrue(aPair);
  double tKStarRec = aPair->KStar();

//  if(tWeight != 0 && tKStarTrue != 0 && tKStarTrue > -999)  //make sure we have a good particle
  if(tWeight > 0. && tKStarTrue > 0. && tKStarRec > 0.)  //make sure we have a good particle
  {
    fNumeratorTrue->Fill(tKStarRec,tWeight);
    fNumeratorTrueIdeal->Fill(tKStarTrue,tWeight);

    fNumTrueUnitWeights->Fill(tKStarRec,1.0);
    fNumTrueIdealUnitWeights->Fill(tKStarTrue,1.0);

    fKTrueKRecSame->Fill(tKStarTrue,tKStarRec);

    double tX = 0.5*(tKStarRec + tKStarTrue);
    double tY = (tKStarRec - tKStarTrue)/sqrt(2);
    fKTrueKRecRotSame->Fill(tX,tY);

  }
}

//_________________________________________
void myAliFemtoModelCorrFctnKStar::AddMixedPair(AliFemtoPair* aPair)
{
  double tWeight;
  if(fManager) tWeight = fManager->GetWeight(aPair);
  else tWeight = 1.0;

  double tKStarTrue = GetKStarTrue(aPair);
  double tKStarRec = aPair->KStar();

//  if(tWeight > 0 && tKStarTrue > 0 && tKStarTrue > -999)  //make sure we have a good particle
  if(tWeight > 0. && tKStarTrue > 0. && tKStarRec > 0.)  //make sure we have a good particle
  {
    fNumeratorFake->Fill(tKStarRec,tWeight);
    fDenominator->Fill(tKStarRec, 1.0);

    fNumeratorFakeIdeal->Fill(tKStarTrue,tWeight);
    fDenominatorIdeal->Fill(tKStarTrue, 1.0);

//    fQgenQrec->Fill(tKStarTrue,aPair->KStar());

    fKTrueKRecMixed->Fill(tKStarTrue,tKStarRec);

    double tX = 0.5*(tKStarRec + tKStarTrue);
    double tY = (tKStarRec - tKStarTrue)/sqrt(2);
    fKTrueKRecRotMixed->Fill(tX,tY);

  }
}

//_______________________
void myAliFemtoModelCorrFctnKStar::Write()
{
  // Write out data histos

  fNumeratorTrue->Write();
  fNumeratorFake->Write();
  fDenominator->Write();

  fNumeratorTrueIdeal->Write();
  fNumeratorFakeIdeal->Write();
  fDenominatorIdeal->Write();

  fNumTrueUnitWeights->Write();
  fNumTrueIdealUnitWeights->Write();

//  fQgenQrec->Write();

  fKTrueKRecSame->Write();
  fKTrueKRecMixed->Write();
  fKTrueKRecRotSame->Write();
  fKTrueKRecRotMixed->Write();
}

//_________________________
TList* myAliFemtoModelCorrFctnKStar::GetOutputList()
{
  // Prepare the list of objects to be written to the output
  TList *tOutputList = new TList();

  tOutputList->Add(fNumeratorTrue);
  tOutputList->Add(fNumeratorFake);
  tOutputList->Add(fDenominator);

  tOutputList->Add(fNumeratorTrueIdeal);
  tOutputList->Add(fNumeratorFakeIdeal);
  tOutputList->Add(fDenominatorIdeal);

  tOutputList->Add(fNumTrueUnitWeights);
  tOutputList->Add(fNumTrueIdealUnitWeights);

//  tOutputList->Add(fQgenQrec);

  tOutputList->Add(fKTrueKRecSame);
  tOutputList->Add(fKTrueKRecMixed);
  tOutputList->Add(fKTrueKRecRotSame);
  tOutputList->Add(fKTrueKRecRotMixed);


  return tOutputList;
}

