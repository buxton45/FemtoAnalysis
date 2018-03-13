///////////////////////////////////////////////////////////////////////////
// FitSharedAnalyses:                                                    //
///////////////////////////////////////////////////////////////////////////


#include "FitSharedAnalyses.h"

#ifdef __ROOT__
ClassImp(FitSharedAnalyses)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitSharedAnalyses::FitSharedAnalyses(vector<FitPairAnalysis*> &aVecOfFitPairAnalyses) :
  fMinuit(0),
  fFitType(kChi2PML),
  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType(kLinear),
  fNFitPairAnalysis(aVecOfFitPairAnalyses.size()),
  fNFitParamsPerAnalysis(0),
  fNFitNormParamsPerAnalysis(0),
  fFixNormParams(false),

  fFitPairAnalysisCollection(aVecOfFitPairAnalyses),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMinuitFitParametersMatrix(0),

  fFitChi2Histograms(0),

  fParallelSharedResidualCollection(nullptr)
{
  //set fFitPairAnalysisNumber in each FitPairAnalysis object
  for(int i=0; i<fNFitPairAnalysis; i++) {fFitPairAnalysisCollection[i]->SetFitPairAnalysisNumber(i);}

  //make sure pair analyses in collection have same fNFitParams and fNFitNormParams
  for(int i=1; i<fNFitPairAnalysis; i++)
  {
    assert(fFitPairAnalysisCollection[i-1]->GetNFitParams() == fFitPairAnalysisCollection[i]->GetNFitParams());
    assert(fFitPairAnalysisCollection[i-1]->GetNFitNormParams() == fFitPairAnalysisCollection[i]->GetNFitNormParams());
  }
  fNFitParamsPerAnalysis = fFitPairAnalysisCollection[0]->GetNFitParams();
  fNFitNormParamsPerAnalysis = fFitPairAnalysisCollection[0]->GetNFitNormParams();

  fMinuit = new TMinuit(50);
  fFitChi2Histograms = new FitChi2Histograms();
}



//________________________________________________________________________________________________________________
FitSharedAnalyses::FitSharedAnalyses(vector<FitPairAnalysis*> &aVecOfFitPairAnalyses, vector<ParameterType> &aVecOfSharedParameterTypes) :
  fMinuit(0),
  fFitType(kChi2PML),
  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType(kLinear),
  fNFitPairAnalysis(aVecOfFitPairAnalyses.size()),
  fNFitParamsPerAnalysis(0),
  fNFitNormParamsPerAnalysis(0),
  fFixNormParams(false),

  fFitPairAnalysisCollection(aVecOfFitPairAnalyses),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMinuitFitParametersMatrix(0),

  fFitChi2Histograms(0),

  fParallelSharedResidualCollection(nullptr)
{
  //set fFitPairAnalysisNumber in each FitPairAnalysis object
  for(int i=0; i<fNFitPairAnalysis; i++) {fFitPairAnalysisCollection[i]->SetFitPairAnalysisNumber(i);}

  //make sure pair analyses in collection have same fNFitParams and fNFitNormParams
  for(int i=1; i<fNFitPairAnalysis; i++)
  {
    assert(fFitPairAnalysisCollection[i-1]->GetNFitParams() == fFitPairAnalysisCollection[i]->GetNFitParams());
    assert(fFitPairAnalysisCollection[i-1]->GetNFitNormParams() == fFitPairAnalysisCollection[i]->GetNFitNormParams());
  }
  fNFitParamsPerAnalysis = fFitPairAnalysisCollection[0]->GetNFitParams();
  fNFitNormParamsPerAnalysis = fFitPairAnalysisCollection[0]->GetNFitNormParams();

  fMinuit = new TMinuit(50);
  fFitChi2Histograms = new FitChi2Histograms();
}


//________________________________________________________________________________________________________________
FitSharedAnalyses::~FitSharedAnalyses()
{
  cout << "FitSharedAnalyses object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
void FitSharedAnalyses::SetParameter(ParameterType aParamType, int aAnalysisNumber, double aStartValue, double aLowerBound, double aUpperBound, bool aIsFixed)
{
  fFitPairAnalysisCollection[aAnalysisNumber]->GetFitParameter(aParamType)->SetStartValue(aStartValue);
  fFitPairAnalysisCollection[aAnalysisNumber]->GetFitParameter(aParamType)->SetLowerBound(aLowerBound);
  fFitPairAnalysisCollection[aAnalysisNumber]->GetFitParameter(aParamType)->SetUpperBound(aUpperBound);
  fFitPairAnalysisCollection[aAnalysisNumber]->GetFitParameter(aParamType)->SetFixed(aIsFixed);
}



//________________________________________________________________________________________________________________
void FitSharedAnalyses::CompareParameters(TString aAnalysisName1, FitParameter* aParam1, TString aAnalysisName2, FitParameter* aParam2)
{
  assert(aParam1->GetType() == aParam2->GetType());

  bool tAlwaysChoose1 = false;
  bool tAlwaysChoose2 = false;

  if(aParam1->GetStartValue() != aParam2->GetStartValue())
  {
    if(tAlwaysChoose1) {aParam2->SetStartValue(aParam1->GetStartValue());}
    else if(tAlwaysChoose2) {aParam1->SetStartValue(aParam2->GetStartValue());}
    else
    {
      cout << "StartValues do not agree for parameters to be shared: " << aParam1->GetName() << endl;
      cout << "The two values are: " << endl;
      cout << "\t (1): " << aParam1->GetStartValue() << " from the analysis " << aAnalysisName1 << endl;
      cout << "\t (2): " << aParam2->GetStartValue() << " from the analysis " << aAnalysisName2 << endl;
      cout << "Which value should be used?" << endl;
      cout << "\t Choose (1) or (2) to choose values for StartValues" << endl;
      cout << "\t Choose (11) or (22) to choose which parameter to use for all discrepancies" << endl;
      int tResponse;
      cin >> tResponse;
      assert( (tResponse == 1) || (tResponse == 2) || (tResponse == 11) || (tResponse == 22) );
      if(tResponse == 1) {aParam2->SetStartValue(aParam1->GetStartValue());}
      else if(tResponse == 2) {aParam1->SetStartValue(aParam2->GetStartValue());}
      else if(tResponse == 11) {tAlwaysChoose1 = true;}
      else if(tResponse == 22) {tAlwaysChoose2 = true;}
      else {cout << "Invalid selection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;}
    }
  }

  if(aParam1->GetLowerBound() != aParam2->GetLowerBound())
  {
    if(tAlwaysChoose1) {aParam2->SetLowerBound(aParam1->GetLowerBound());}
    else if(tAlwaysChoose2) {aParam1->SetLowerBound(aParam2->GetLowerBound());}
    else
    {
      cout << "LowerBounds do not agree for parameters to be shared: " << aParam1->GetName() << endl;
      cout << "The two values are: " << endl;
      cout << "\t (1): " << aParam1->GetLowerBound() << " from the analysis " << aAnalysisName1 << endl;
      cout << "\t (2): " << aParam2->GetLowerBound() << " from the analysis " << aAnalysisName2 << endl;
      cout << "Which value should be used?" << endl;
      cout << "\t Choose (1) or (2) to choose values for LowerBounds" << endl;
      cout << "\t Choose (11) or (22) to choose which parameter to use for all discrepancies" << endl;
      int tResponse;
      cin >> tResponse;
      assert( (tResponse == 1) || (tResponse == 2) || (tResponse == 11) || (tResponse == 22) );
      if(tResponse == 1) {aParam2->SetLowerBound(aParam1->GetLowerBound());}
      else if(tResponse == 2) {aParam1->SetLowerBound(aParam2->GetLowerBound());}
      else if(tResponse == 11) {tAlwaysChoose1 = true;}
      else if(tResponse == 22) {tAlwaysChoose2 = true;}
      else {cout << "Invalid selection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;}
    }
  }

  if(aParam1->GetUpperBound() != aParam2->GetUpperBound())
  {
    if(tAlwaysChoose1) {aParam2->SetUpperBound(aParam1->GetUpperBound());}
    else if(tAlwaysChoose2) {aParam1->SetUpperBound(aParam2->GetUpperBound());}
    else
    {
      cout << "UpperBounds do not agree for parameters to be shared: " << aParam1->GetName() << endl;
      cout << "The two values are: " << endl;
      cout << "\t (1): " << aParam1->GetUpperBound() << " from the analysis " << aAnalysisName1 << endl;
      cout << "\t (2): " << aParam2->GetUpperBound() << " from the analysis " << aAnalysisName2 << endl;
      cout << "Which value should be used?" << endl;
      cout << "\t Choose (1) or (2) to choose values for UpperBounds" << endl;
      cout << "\t Choose (11) or (22) to choose which parameter to use for all discrepancies" << endl;
      int tResponse;
      cin >> tResponse;
      assert( (tResponse == 1) || (tResponse == 2) || (tResponse == 11) || (tResponse == 22) );
      if(tResponse == 1) {aParam2->SetUpperBound(aParam1->GetUpperBound());}
      else if(tResponse == 2) {aParam1->SetUpperBound(aParam2->GetUpperBound());}
      else if(tResponse == 11) {tAlwaysChoose1 = true;}
      else if(tResponse == 22) {tAlwaysChoose2 = true;}
      else {cout << "Invalid selection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;}
    }
  }


  if(aParam1->IsFixed() != aParam2->IsFixed())
  {
    if(tAlwaysChoose1) {aParam2->SetFixed(aParam1->IsFixed());}
    else if(tAlwaysChoose2) {aParam1->SetFixed(aParam2->IsFixed());}
    else
    {
      cout << "IsFixed() do not agree for parameters to be shared: " << aParam1->GetName() << endl;
      cout << "The two values are: " << endl;
      cout << "\t (1): " << aParam1->IsFixed() << " from the analysis " << aAnalysisName1 << endl;
      cout << "\t (2): " << aParam2->IsFixed() << " from the analysis " << aAnalysisName2 << endl;
      cout << "Which value should be used?" << endl;
      cout << "\t Choose (1) or (2) to choose values for IsFixed" << endl;
      cout << "\t Choose (11) or (22) to choose which parameter to use for all discrepancies" << endl;
      int tResponse;
      cin >> tResponse;
      assert( (tResponse == 1) || (tResponse == 2) || (tResponse == 11) || (tResponse == 22) );
      if(tResponse == 1) {aParam2->SetFixed(aParam1->IsFixed());}
      else if(tResponse == 2) {aParam1->SetFixed(aParam2->IsFixed());}
      else if(tResponse == 11) {tAlwaysChoose1 = true;}
      else if(tResponse == 22) {tAlwaysChoose2 = true;}
      else {cout << "Invalid selection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;}
    }
  }




}


//________________________________________________________________________________________________________________
void FitSharedAnalyses::SetSharedParameter(ParameterType aParamType, bool aIsFixed)
{
  //Sets parameter for all analyses
  //If I only want to set for certain analyses, used the overloaded SetSharedParameter function with &aSharedAnalyses argument

  vector<int> tAllShared (fNFitPairAnalysis);
  for(int i=0; i<fNFitPairAnalysis; i++) {tAllShared[i] = i;}


  for(int i=0; i<fNFitPairAnalysis; i++)
  {
    fFitPairAnalysisCollection[i]->GetFitParameter(aParamType)->SetSharedGlobal(true,tAllShared);
    fFitPairAnalysisCollection[i]->GetFitParameter(aParamType)->SetFixed(aIsFixed);
    if(i != 0)
    {
      CompareParameters(fFitPairAnalysisCollection[0]->GetAnalysisName(), fFitPairAnalysisCollection[0]->GetFitParameter(aParamType), fFitPairAnalysisCollection[i]->GetAnalysisName(), fFitPairAnalysisCollection[i]->GetFitParameter(aParamType));

      fFitPairAnalysisCollection[i]->SetFitParameter(fFitPairAnalysisCollection[0]->GetFitParameter(aParamType));
    }

  }


}


//________________________________________________________________________________________________________________
void FitSharedAnalyses::SetSharedParameter(ParameterType aParamType, double aStartValue, double aLowerBound, double aUpperBound, bool aIsFixed)
{
  //Sets parameter for all analyses
  //If I only want to set for certain analyses, used the overloaded SetSharedParameter function with &aSharedAnalyses argument

  vector<int> tAllShared (fNFitPairAnalysis);
  for(int i=0; i<fNFitPairAnalysis; i++) {tAllShared[i] = i;}

  vector<FitParameter*> tempSharedParameters (fNFitPairAnalysis);

  //----------
  fFitPairAnalysisCollection[0]->GetFitParameter(aParamType)->SetSharedGlobal(true,tAllShared);
  fFitPairAnalysisCollection[0]->GetFitParameter(aParamType)->SetStartValue(aStartValue);
  fFitPairAnalysisCollection[0]->GetFitParameter(aParamType)->SetLowerBound(aLowerBound);
  fFitPairAnalysisCollection[0]->GetFitParameter(aParamType)->SetUpperBound(aUpperBound);
  fFitPairAnalysisCollection[0]->GetFitParameter(aParamType)->SetFixed(aIsFixed);

  for(int i=1; i<fNFitPairAnalysis; i++)
  {
    fFitPairAnalysisCollection[i]->GetFitParameter(aParamType)->SetSharedGlobal(true,tAllShared);
    fFitPairAnalysisCollection[i]->SetFitParameter(fFitPairAnalysisCollection[0]->GetFitParameter(aParamType));
  }


  //make sure the above for loop set all parameters equal
  for(int i=0; i<fNFitPairAnalysis; i++)
  {
    if(i != 0)
    {
      CompareParameters(fFitPairAnalysisCollection[0]->GetAnalysisName(), fFitPairAnalysisCollection[0]->GetFitParameter(aParamType), fFitPairAnalysisCollection[i]->GetAnalysisName(), fFitPairAnalysisCollection[i]->GetFitParameter(aParamType));
    }

  }

}



//________________________________________________________________________________________________________________
void FitSharedAnalyses::SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, bool aIsFixed)
{
  vector<FitParameter*> tempSharedParameters (aSharedAnalyses.size());

  for(unsigned int i=0; i<aSharedAnalyses.size(); i++)
  {

    fFitPairAnalysisCollection[aSharedAnalyses[i]]->GetFitParameter(aParamType)->SetSharedGlobal(true,aSharedAnalyses);
    fFitPairAnalysisCollection[aSharedAnalyses[i]]->GetFitParameter(aParamType)->SetFixed(aIsFixed);
    if(i != 0)
    {
      CompareParameters(fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetAnalysisName(), fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType), fFitPairAnalysisCollection[aSharedAnalyses[i]]->GetAnalysisName(), fFitPairAnalysisCollection[aSharedAnalyses[i]]->GetFitParameter(aParamType));

      fFitPairAnalysisCollection[aSharedAnalyses[i]]->SetFitParameter(fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType));
    }

  }


}




//________________________________________________________________________________________________________________
void FitSharedAnalyses::SetSharedParameter(ParameterType aParamType, const vector<int> &aSharedAnalyses, double aStartValue, double aLowerBound, double aUpperBound, bool aIsFixed)
{
  vector<FitParameter*> tempSharedParameters (aSharedAnalyses.size());

  //----------
  fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType)->SetSharedGlobal(true,aSharedAnalyses);
  fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType)->SetStartValue(aStartValue);
  fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType)->SetLowerBound(aLowerBound);
  fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType)->SetUpperBound(aUpperBound);
  fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType)->SetFixed(aIsFixed);

  for(unsigned int i=1; i<aSharedAnalyses.size(); i++)
  {
    fFitPairAnalysisCollection[aSharedAnalyses[i]]->GetFitParameter(aParamType)->SetSharedGlobal(true,aSharedAnalyses);
    fFitPairAnalysisCollection[aSharedAnalyses[i]]->SetFitParameter(fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType));
  } 


  //make sure the above for loop set all parameters equal
  for(unsigned int i=0; i<aSharedAnalyses.size(); i++)
  {
    if(i != 0)
    {
      CompareParameters(fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetAnalysisName(), fFitPairAnalysisCollection[aSharedAnalyses[0]]->GetFitParameter(aParamType), fFitPairAnalysisCollection[aSharedAnalyses[i]]->GetAnalysisName(), fFitPairAnalysisCollection[aSharedAnalyses[i]]->GetFitParameter(aParamType));
    }

  }


}

//________________________________________________________________________________________________________________
void FitSharedAnalyses::SetSharedAndFixedParameter(ParameterType aParamType, double aFixedValue)
{
  //Sets parameter fixed for all analyses
  vector<int> tAllShared (fNFitPairAnalysis);
  for(int i=0; i<fNFitPairAnalysis; i++) {tAllShared[i] = i;}


  for(int i=0; i<fNFitPairAnalysis; i++)
  {
    fFitPairAnalysisCollection[i]->GetFitParameter(aParamType)->SetSharedGlobal(true,tAllShared);
    fFitPairAnalysisCollection[i]->GetFitParameter(aParamType)->SetStartValue(aFixedValue);
    fFitPairAnalysisCollection[i]->GetFitParameter(aParamType)->SetFixed(true);

    if(i != 0)
    {
      CompareParameters(fFitPairAnalysisCollection[0]->GetAnalysisName(), fFitPairAnalysisCollection[0]->GetFitParameter(aParamType), fFitPairAnalysisCollection[i]->GetAnalysisName(), fFitPairAnalysisCollection[i]->GetFitParameter(aParamType));

      fFitPairAnalysisCollection[i]->SetFitParameter(fFitPairAnalysisCollection[0]->GetFitParameter(aParamType));
    }

  }


}



//________________________________________________________________________________________________________________
vector<FitParameter*> FitSharedAnalyses::GetDistinctParamsOfCommonType(ParameterType aParamType)
{
  vector<FitParameter*> tReturnVec;
  tReturnVec.clear();

  vector<int> tSharedSoSkip;
  tSharedSoSkip.clear();

  for(int iAnaly=0; iAnaly<fNFitPairAnalysis; iAnaly++)
  {
    FitParameter* tempParam = fFitPairAnalysisCollection[iAnaly]->GetFitParameter(aParamType);
    if(!tempParam->IsSharedGlobal()) {tReturnVec.push_back(tempParam);}

    else
    {
      bool tAlreadyShared = false;
      for(unsigned int iShared = 0; iShared < tSharedSoSkip.size(); iShared++)
      {
        if(iAnaly == tSharedSoSkip[iShared]) {tAlreadyShared = true;}
      }

      if(!tAlreadyShared)
      {
        tReturnVec.push_back(tempParam);
        vector<int> tempSharedWith = tempParam->GetSharedWithGlobal();
        for(unsigned int itemp = 0; itemp < tempSharedWith.size(); itemp++) {tSharedSoSkip.push_back(tempSharedWith[itemp]);}
      }
    }

  }

  return tReturnVec;
}


//________________________________________________________________________________________________________________
void FitSharedAnalyses::CreateMinuitParametersMatrix()
{
  fMinuitFitParametersMatrix.clear();  //TODO added so I could run ScattParamValues.C without creating tons of FitGenerator objects
  for(int iPar=0; iPar<fNFitParamsPerAnalysis; iPar++)
  {
    ParameterType tParamType = static_cast<ParameterType>(iPar);
    vector<FitParameter*> tempParamVec = GetDistinctParamsOfCommonType(tParamType);
    fMinuitFitParametersMatrix.push_back(tempParamVec);
  }

  vector<FitParameter*> tempNormVec;
  double tNormStartValue = 1.;
  for(int iAnaly=0; iAnaly<fNFitPairAnalysis; iAnaly++)
  {
    //tempNormVec.clear();
    tempNormVec = fFitPairAnalysisCollection[iAnaly]->GetFitNormParameters();
    assert((int)tempNormVec.size() == fNFitNormParamsPerAnalysis);
    assert((int)tempNormVec.size() == fFitPairAnalysisCollection[iAnaly]->GetNFitPartialAnalysis());  //always let each have own normalization, so I can't forsee
                                                                                                      //when this would ever fail

    for(int iPartAn=0; iPartAn<fFitPairAnalysisCollection[iAnaly]->GetNFitPartialAnalysis(); iPartAn++)
    {
      if(!fApplyNonFlatBackgroundCorrection && fFitType==kChi2PML)  //any other cases where norm != 1?
      {
        double tNumScale=0., tDenScale=0.;
        tNumScale = fFitPairAnalysisCollection[iAnaly]->GetFitPartialAnalysis(iPartAn)->GetKStarNumScale();
        tDenScale = fFitPairAnalysisCollection[iAnaly]->GetFitPartialAnalysis(iPartAn)->GetKStarDenScale();
        tNormStartValue = tNumScale/tDenScale;
      }
      else tNormStartValue = 1.;
      tempNormVec[iPartAn]->SetStartValue(tNormStartValue);

      if(fFixNormParams) tempNormVec[iPartAn]->SetFixed(true);
    }

    fMinuitFitParametersMatrix.push_back(tempNormVec);
  }

}


//________________________________________________________________________________________________________________
void FitSharedAnalyses::CreateMinuitParameter(int aMinuitParamNumber, FitParameter* aParam)
{
  int tErrFlg = 0;

  fMinuit->mnparm(aMinuitParamNumber,aParam->GetName(),aParam->GetStartValue(),aParam->GetStepSize(),aParam->GetLowerBound(), aParam->GetUpperBound(),tErrFlg);
  if(tErrFlg != 0) {cout << "Error setting minuit parameter #: " << aMinuitParamNumber << endl << "and name: " << aParam->GetName() << endl;}

  if(aParam->IsFixed()) {fMinuit->FixParameter(aMinuitParamNumber);}

  aParam->SetMinuitParamNumber(aMinuitParamNumber);

//TODO  if(aParam->GetType() != kNorm) fFitChi2Histograms->AddParameter(aMinuitParamNumber,aParam);
}



//________________________________________________________________________________________________________________
void FitSharedAnalyses::CreateMinuitParameters()
{
  //call AFTER all parameters have been shared!!!!!

  CreateMinuitParametersMatrix();
  assert((fNFitParamsPerAnalysis+fNFitPairAnalysis) == (int)fMinuitFitParametersMatrix.size());

  fNMinuitParams = 0;  //for some reason, this makes fNMinuitParams = 0 outside of this function?

  for(unsigned int iPar=0; iPar < fMinuitFitParametersMatrix.size(); iPar++)
  {
    vector<FitParameter*> tempVec = fMinuitFitParametersMatrix[iPar];
    for(unsigned int itemp=0; itemp < tempVec.size(); itemp++)
    {
      CreateMinuitParameter(fNMinuitParams,tempVec[itemp]);
      fNMinuitParams++;
    }
  }
  
// TODO  fFitChi2Histograms->InitiateHistograms();
}


//________________________________________________________________________________________________________________
void FitSharedAnalyses::ReturnFitParametersToAnalyses()
{
  for(int iAnaly=0; iAnaly < fNFitPairAnalysis; iAnaly++)
  {
    for(int iPar=0; iPar < fNFitParamsPerAnalysis; iPar++)
    {
      ParameterType tParamType = static_cast<ParameterType>(iPar);
      int tMinuitParamNumber = fFitPairAnalysisCollection[iAnaly]->GetFitParameter(tParamType)->GetMinuitParamNumber();

      fFitPairAnalysisCollection[iAnaly]->GetFitParameter(tParamType)->SetFitValue(fMinuitMinParams[tMinuitParamNumber]);
      fFitPairAnalysisCollection[iAnaly]->GetFitParameter(tParamType)->SetFitValueError(fMinuitParErrors[tMinuitParamNumber]);
    }

    for(int iNorm=0; iNorm<fNFitNormParamsPerAnalysis; iNorm++)
    {
      int tMinuitParamNumber = fFitPairAnalysisCollection[iAnaly]->GetFitNormParameter(iNorm)->GetMinuitParamNumber();
      fFitPairAnalysisCollection[iAnaly]->GetFitNormParameter(iNorm)->SetFitValue(fMinuitMinParams[tMinuitParamNumber]);
      fFitPairAnalysisCollection[iAnaly]->GetFitNormParameter(iNorm)->SetFitValueError(fMinuitParErrors[tMinuitParamNumber]);
    }

  }

}

//________________________________________________________________________________________________________________
double FitSharedAnalyses::GetKStarMinNorm()
{
  for(int i=1; i<fNFitPairAnalysis; i++) assert(fFitPairAnalysisCollection[i-1]->GetKStarMinNorm() == fFitPairAnalysisCollection[i]->GetKStarMinNorm());
  return fFitPairAnalysisCollection[0]->GetKStarMinNorm();
}


//________________________________________________________________________________________________________________
double FitSharedAnalyses::GetKStarMaxNorm()
{
  for(int i=1; i<fNFitPairAnalysis; i++) assert(fFitPairAnalysisCollection[i-1]->GetKStarMaxNorm() == fFitPairAnalysisCollection[i]->GetKStarMaxNorm());
  return fFitPairAnalysisCollection[0]->GetKStarMaxNorm();
}


//________________________________________________________________________________________________________________
double FitSharedAnalyses::GetMinBgdFit()
{
  for(int i=1; i<fNFitPairAnalysis; i++) assert(fFitPairAnalysisCollection[i-1]->GetMinBgdFit() == fFitPairAnalysisCollection[i]->GetMinBgdFit());
  return fFitPairAnalysisCollection[0]->GetMinBgdFit();
}


//________________________________________________________________________________________________________________
double FitSharedAnalyses::GetMaxBgdFit()
{
  for(int i=1; i<fNFitPairAnalysis; i++) assert(fFitPairAnalysisCollection[i-1]->GetMaxBgdFit() == fFitPairAnalysisCollection[i]->GetMaxBgdFit());
  return fFitPairAnalysisCollection[0]->GetMaxBgdFit();
}


//________________________________________________________________________________________________________________
BinInfoTransformMatrix FitSharedAnalyses::BuildBinInfoTransformMatrix(AnalysisType aDaughterAnType, AnalysisType aParentResType, CentralityType aCentType, TH2D* aTransformMatrix)
{
  BinInfoTransformMatrix tReturn;

  tReturn.daughterAnalysisType = aDaughterAnType;
  tReturn.parentResidualType = aParentResType;
  tReturn.centralityType = aCentType;
  tReturn.global2DMatrixPosition = -1;
  tReturn.globalOffset = -1;

  tReturn.nBinsX = aTransformMatrix->GetNbinsX();
  tReturn.nBinsY = aTransformMatrix->GetNbinsY();

  tReturn.binWidthX = aTransformMatrix->GetXaxis()->GetBinWidth(1);
  tReturn.binWidthY = aTransformMatrix->GetYaxis()->GetBinWidth(1);

  tReturn.minX = aTransformMatrix->GetXaxis()->GetBinLowEdge(1);
  tReturn.maxX = aTransformMatrix->GetXaxis()->GetBinUpEdge(aTransformMatrix->GetNbinsX());
  tReturn.minY = aTransformMatrix->GetYaxis()->GetBinLowEdge(1);
  tReturn.maxY = aTransformMatrix->GetYaxis()->GetBinUpEdge(aTransformMatrix->GetNbinsY());

  return tReturn;
}


//________________________________________________________________________________________________________________
td2dVec FitSharedAnalyses::BuildTransformVec(TH2D* aTransformMatrix)
{
  td2dVec tTransformVec(0);

  tTransformVec.resize(aTransformMatrix->GetNbinsX(), td1dVec(aTransformMatrix->GetNbinsY(), 0));

  for(int i=1; i<=aTransformMatrix->GetNbinsX(); i++)
  {
    for(int j=1; j<=aTransformMatrix->GetNbinsY(); j++)
    {
      tTransformVec[i-1][j-1] = aTransformMatrix->GetBinContent(i, j);
    }
  }

  return tTransformVec;
}


//________________________________________________________________________________________________________________
void FitSharedAnalyses::BuildParallelSharedResidualCollection()
{
  fParallelSharedResidualCollection = new ParallelSharedResidualCollection();

  for(unsigned int iPairAn=0; iPairAn<fFitPairAnalysisCollection.size(); iPairAn++)
  {
    ResidualCollection* tResColl = fFitPairAnalysisCollection[iPairAn]->GetResidualCollection();
    vector<NeutralResidualCf> tNeutralResiduals = tResColl->GetNeutralCollection();
    vector<SimpleChargedResidualCf> tChargedResiduals = tResColl->GetChargedCollection();

    for(int iRes=0; iRes<tNeutralResiduals.size(); iRes++)
    {
      BinInfoTransformMatrix tBinInfo = BuildBinInfoTransformMatrix(fFitPairAnalysisCollection[iPairAn]->GetAnalysisType(),
                                                                    tNeutralResiduals[iRes].GetResidualType(),
                                                                    fFitPairAnalysisCollection[iPairAn]->GetCentralityType(),
                                                                    tNeutralResiduals[iRes].GetTransformMatrix());
      td2dVec tTransformVec = BuildTransformVec(tNeutralResiduals[iRes].GetTransformMatrix());
      fParallelSharedResidualCollection->LoadTransformMatrix(tTransformVec, tBinInfo);
    }
    for(int iRes=0; iRes<tChargedResiduals.size(); iRes++)
    {
      BinInfoTransformMatrix tBinInfo = BuildBinInfoTransformMatrix(fFitPairAnalysisCollection[iPairAn]->GetAnalysisType(),
                                                                    tChargedResiduals[iRes].GetResidualType(),
                                                                    fFitPairAnalysisCollection[iPairAn]->GetCentralityType(),
                                                                    tChargedResiduals[iRes].GetTransformMatrix());
      td2dVec tTransformVec = BuildTransformVec(tChargedResiduals[iRes].GetTransformMatrix());
      fParallelSharedResidualCollection->LoadTransformMatrix(tTransformVec, tBinInfo);
    }
  }

}



