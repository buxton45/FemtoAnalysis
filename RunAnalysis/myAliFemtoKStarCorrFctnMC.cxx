///////////////////////////////////////////////////////////////////////////
//                                                                       //
// myAliFemtoKStarCorrFctnMC:                                                 //
// a simple Q-invariant correlation function                             //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "myAliFemtoKStarCorrFctnMC.h"
//#include "AliFemtoHisto.h"
#include <cstdio>

#ifdef __ROOT__
ClassImp(myAliFemtoKStarCorrFctnMC)
#endif

//____________________________
myAliFemtoKStarCorrFctnMC::myAliFemtoKStarCorrFctnMC(const char* title, const int& nbins, const float& KStarLo, const float& KStarHi):
  fNumeratorTrue(0),
  fDenominatorTrue(0)

{
  // set up fNumeratorTrue
  char tTitNumTrue[101] = "Num";
  strncat(tTitNumTrue,title, 100);
  fNumeratorTrue = new TH1D(tTitNumTrue,title,nbins,KStarLo,KStarHi);

  // set up fDenominatorTrue
  char tTitDenTrue[101] = "Den";
  strncat(tTitDenTrue,title, 100);
  fDenominatorTrue = new TH1D(tTitDenTrue,title,nbins,KStarLo,KStarHi);

  // to enable error bar calculation...
  fNumeratorTrue->Sumw2();
  fDenominatorTrue->Sumw2();

}

//____________________________
myAliFemtoKStarCorrFctnMC::myAliFemtoKStarCorrFctnMC(const myAliFemtoKStarCorrFctnMC& aCorrFctn) :
  AliFemtoCorrFctn(),
  fNumeratorTrue(0),
  fDenominatorTrue(0)
{
  // copy constructor
  fNumeratorTrue = new TH1D(*aCorrFctn.fNumeratorTrue);
  fDenominatorTrue = new TH1D(*aCorrFctn.fDenominatorTrue);
}
//____________________________
myAliFemtoKStarCorrFctnMC::~myAliFemtoKStarCorrFctnMC(){
  // destructor
  delete fNumeratorTrue;
  delete fDenominatorTrue;

}
//_________________________
myAliFemtoKStarCorrFctnMC& myAliFemtoKStarCorrFctnMC::operator=(const myAliFemtoKStarCorrFctnMC& aCorrFctn)
{
  // assignment operator
  if (this == &aCorrFctn)
    return *this;


  if (fNumeratorTrue) delete fNumeratorTrue;
  fNumeratorTrue = new TH1D(*aCorrFctn.fNumeratorTrue);
  if (fDenominatorTrue) delete fDenominatorTrue;
  fDenominatorTrue = new TH1D(*aCorrFctn.fDenominatorTrue);

  return *this;
}

//_________________________
void myAliFemtoKStarCorrFctnMC::Finish(){
  // here is where we should normalize, fit, etc...
  // we should NOT Draw() the histos (as I had done it below),
  // since we want to insulate ourselves from root at this level
  // of the code.  Do it instead at root command line with browser.
  //  fNumeratorTrue->Draw();
  //  fDenominatorTrue->Draw();


}


//____________________________
double myAliFemtoKStarCorrFctnMC::GetKStarTrue(AliFemtoPair* aPair)
{
  AliFemtoParticle *tPart1 = (AliFemtoParticle*)aPair->Track1();
  AliFemtoParticle *tPart2 = (AliFemtoParticle*)aPair->Track2();

  if(tPart1 != NULL && tPart2 != NULL)
  {
    AliFemtoModelHiddenInfo *tPart1Info = (AliFemtoModelHiddenInfo*)tPart1->GetHiddenInfo();
    AliFemtoModelHiddenInfo *tPart2Info = (AliFemtoModelHiddenInfo*)tPart2->GetHiddenInfo();

    if(tPart1Info != NULL && tPart2Info != NULL)
    {
      AliFemtoThreeVector *tPart1TrueMomentum = tPart1Info->GetTrueMomentum();
      double px1 = tPart1TrueMomentum->x();
      double py1 = tPart1TrueMomentum->y();
      double pz1 = tPart1TrueMomentum->z();
      double mass1 = tPart1Info->GetMass();
      double E1 = sqrt(mass1*mass1 + px1*px1 + py1*py1 + pz1*pz1);


      AliFemtoThreeVector *tPart2TrueMomentum = tPart2Info->GetTrueMomentum();
      double px2 = tPart2TrueMomentum->x();
      double py2 = tPart2TrueMomentum->y();
      double pz2 = tPart2TrueMomentum->z();
      double mass2 = tPart2Info->GetMass();
      double E2 = sqrt(mass2*mass2 + px2*px2 + py2*py2 + pz2*pz2);

      //------------------------------------------------------------

      double tMinvSq = (E1+E2)*(E1+E2) - (px1+px2)*(px1+px2) - (py1+py2)*(py1+py2) - (pz1+pz2)*(pz1+pz2);

      double tQinvSq = ((mass1*mass1 - mass2*mass2)*(mass1*mass1 - mass2*mass2))/tMinvSq + tMinvSq - 2.0*(mass1*mass1 + mass2*mass2);

      double tKStar = 0.5*sqrt(tQinvSq);

      return tKStar;
    }
  }

  return -999;
}



//____________________________
AliFemtoString myAliFemtoKStarCorrFctnMC::Report(){
  // construct report
  string stemp = "KStar Correlation Function Report:\n";
  char ctemp[100];
  snprintf(ctemp , 100, "Number of entries in fNumeratorTrue:\t%E\n",fNumeratorTrue->GetEntries());
  stemp += ctemp;
  snprintf(ctemp , 100, "Number of entries in fDenominatorTrue:\t%E\n",fDenominatorTrue->GetEntries());
  stemp += ctemp;
  //  stemp += mCoulombWeight->Report();
  AliFemtoString returnThis = stemp;
  return returnThis;
}
//____________________________
void myAliFemtoKStarCorrFctnMC::AddRealPair(AliFemtoPair* pair){
  // add true pair
  if (fPairCut)
    if (!fPairCut->Pass(pair)) return;

  double tKStarTrue = fabs(GetKStarTrue(pair));
  fNumeratorTrue->Fill(tKStarTrue);


}

//____________________________
void myAliFemtoKStarCorrFctnMC::AddMixedPair(AliFemtoPair* pair){
  // add mixed (background) pair
  if (fPairCut)
    if (!fPairCut->Pass(pair)) return;

  double weight = 1.0;
  double tKStarTrue = fabs(GetKStarTrue(pair));
  fDenominatorTrue->Fill(tKStarTrue,weight);
}
//____________________________
void myAliFemtoKStarCorrFctnMC::Write(){
  // Write out neccessary objects
  fNumeratorTrue->Write();
  fDenominatorTrue->Write();
}
//______________________________
TList* myAliFemtoKStarCorrFctnMC::GetOutputList()
{
  // Prepare the list of objects to be written to the output
  TList *tOutputList = new TList();

  tOutputList->Add(fNumeratorTrue);
  tOutputList->Add(fDenominatorTrue);

  return tOutputList;
}

