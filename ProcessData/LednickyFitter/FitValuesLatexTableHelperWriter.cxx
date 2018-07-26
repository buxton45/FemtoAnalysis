/* FitValuesLatexTableHelperWriter.cxx */

#include "FitValuesLatexTableHelperWriter.h"

#ifdef __ROOT__
ClassImp(FitValuesLatexTableHelperWriter)
#endif




//________________________________________________________________________________________________________________
FitValuesLatexTableHelperWriter::FitValuesLatexTableHelperWriter(TString aMasterFileLocation, TString aResultsDate, AnalysisType aAnType) : 
  FitValuesWriter(aMasterFileLocation, aResultsDate, aAnType),
  fResType(kInclude3Residuals)
{

}





//________________________________________________________________________________________________________________
FitValuesLatexTableHelperWriter::~FitValuesLatexTableHelperWriter()
{
  //no-op
}


//________________________________________________________________________________________________________________
TString FitValuesLatexTableHelperWriter::GetTwoLetterID(TString aFitInfoTString, IncludeResidualsType aResType)
{
  TString tCommon1 = "_MomResCrctn";
  TString tCommon2 = "_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly";

  TString tReturnID = "";
  if     (aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Aa");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Ba");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Ca");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Da");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Ea");
  }
  //--------------------------------------------------------------
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Ab");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Bb");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Cb");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Db");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Eb");
  }
  //--------------------------------------------------------------
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Ac");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Bc");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Cc");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Dc");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Ec");
  }
  //--------------------------------------------------------------
  //--------------------------------------------------------------
  else if(aFitInfoTString.EqualTo(TString::Format("%s%s%s_StavCf", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Ad");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s%s%s_StavCf_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Bd");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s%s%s_StavCf_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Cd");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s%s%s_StavCf_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Dd");
  }
  else if(aFitInfoTString.EqualTo(TString::Format("%s%s%s_StavCf_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data())))
  {
    tReturnID = TString("Ed");
  }
  //--------------------------------------------------------------
  else
  {
    cout << "In FitValuesLatexTableHelperWriter::GetTwoLetterID: NO return ID found for aFitInfoTString = " << aFitInfoTString << " and aResType = " << cIncludeResidualsTypeTags[aResType] << endl;
    assert(0);
  }

  return tReturnID;
}


//________________________________________________________________________________________________________________
TString FitValuesLatexTableHelperWriter::GetFitInfoTStringFromTwoLetterID(TString aTwoLetterID, IncludeResidualsType aResType)
{
  TString tCommon1 = "_MomResCrctn";
  TString tCommon2 = "_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly";

  TString tReturnFitInfoTString = "";

  if     (aTwoLetterID.EqualTo("Aa")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ba")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ca")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Da")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ea")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ab")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Bb")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Cb")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Db")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Eb")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ac")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Bc")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Cc")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Dc")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ec")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ad")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Bd")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Cd")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Dd")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ed")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  //--------------------------------------------------------------
  else assert(0);

  return tReturnFitInfoTString;
}


//________________________________________________________________________________________________________________
TString FitValuesLatexTableHelperWriter::GetLatexTableOverallLabel(TString aFitInfoTString)
{
  TString tTwoLetterID = GetTwoLetterID(aFitInfoTString, kInclude3Residuals);
  TString tReturnLabel = "";

  if     (tTwoLetterID.EqualTo("Aa")) tReturnLabel = TString("FitGen_PolyNFB_NoLamShare");
  else if(tTwoLetterID.EqualTo("Ba")) tReturnLabel = TString("FitGen_PolyNFB_ShareLamConj");
  else if(tTwoLetterID.EqualTo("Ca")) tReturnLabel = TString("Dualie_PolyNFB_NoLamShare");
  else if(tTwoLetterID.EqualTo("Da")) tReturnLabel = TString("Dualie_PolyNFB_ShareLamConj");
  else if(tTwoLetterID.EqualTo("Ea")) tReturnLabel = TString("Dualie_PolyNFB_ShareSingleLam");
  //--------------------------------------------------------------
  else if(tTwoLetterID.EqualTo("Ab")) tReturnLabel = TString("FitGen_LinrNFB_NoLamShare");
  else if(tTwoLetterID.EqualTo("Bb")) tReturnLabel = TString("FitGen_LinrNFB_ShareLamConj");
  else if(tTwoLetterID.EqualTo("Cb")) tReturnLabel = TString("Dualie_LinrNFB_NoLamShare");
  else if(tTwoLetterID.EqualTo("Db")) tReturnLabel = TString("Dualie_LinrNFB_ShareLamConj");
  else if(tTwoLetterID.EqualTo("Eb")) tReturnLabel = TString("Dualie_LinrNFB_ShareSingleLam");
  //--------------------------------------------------------------
  else if(tTwoLetterID.EqualTo("Ac")) tReturnLabel = TString("FitGen_LinrNFB_StavCf_NoLamShare");
  else if(tTwoLetterID.EqualTo("Bc")) tReturnLabel = TString("FitGen_LinrNFB_StavCf_ShareLamConj");
  else if(tTwoLetterID.EqualTo("Cc")) tReturnLabel = TString("Dualie_LinrNFB_StavCf_NoLamShare");
  else if(tTwoLetterID.EqualTo("Dc")) tReturnLabel = TString("Dualie_LinrNFB_StavCf_ShareLamConj");
  else if(tTwoLetterID.EqualTo("Ec")) tReturnLabel = TString("Dualie_LinrNFB_StavCf_ShareSingleLam");
  //--------------------------------------------------------------
  else if(tTwoLetterID.EqualTo("Ad")) tReturnLabel = TString("FitGen_NoNFB_StavCf_NoLamShare");
  else if(tTwoLetterID.EqualTo("Bd")) tReturnLabel = TString("FitGen_NoNFB_StavCf_ShareLamConj");
  else if(tTwoLetterID.EqualTo("Cd")) tReturnLabel = TString("Dualie_NoNFB_StavCf_NoLamShare");
  else if(tTwoLetterID.EqualTo("Dd")) tReturnLabel = TString("Dualie_NoNFB_StavCf_ShareLamConj");
  else if(tTwoLetterID.EqualTo("Ed")) tReturnLabel = TString("Dualie_NoNFB_StavCf_ShareSingleLam");
  //--------------------------------------------------------------
  else assert(0);

  return tReturnLabel;
}

//________________________________________________________________________________________________________________
vector<TString> FitValuesLatexTableHelperWriter::GetFitInfoTStringAndLatexTableOverallLabel(TString aTwoLetterID, IncludeResidualsType aResType)
{
  vector<TString> tReturnVec(0);
  TString tFitInfoTString = GetFitInfoTStringFromTwoLetterID(aTwoLetterID, aResType);
  TString tLatexTableOverallLabel = GetLatexTableOverallLabel(aTwoLetterID);

  tReturnVec = vector<TString>{tFitInfoTString, tLatexTableOverallLabel};
  return tReturnVec;
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriter::WriteLatexTableHelperEntry(ostream &aOut, TString aMasterFileLocation, TString aFitInfoTString, AnalysisType aAnType)
{
  FitParameter *tPar_lam0010, *tPar_lam1030, *tPar_lam3050;
  FitParameter *tPar_R0010, *tPar_R1030, *tPar_R3050;
  FitParameter *tPar_Ref0, *tPar_Imf0, *tPar_d0;

  tPar_lam0010 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k0010, kLambda);
  tPar_lam1030 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k1030, kLambda);
  tPar_lam3050 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k3050, kLambda);

  tPar_R0010 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k0010, kRadius);
  tPar_R1030 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k1030, kRadius);
  tPar_R3050 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k3050, kRadius);

  tPar_Ref0 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k0010, kRef0);
  tPar_Imf0 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k0010, kImf0);
  tPar_d0 = GetFitParameter(aMasterFileLocation, aFitInfoTString, aAnType, k0010, kd0);


  //------------------------------------------------------------------
  aOut << "\\newarray\AaLamKchP" << endl;
  aOut << "\\readarray{AaLamKchP}{" << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & ", tPar_lam0010->GetFitValue(), tPar_R0010->GetFitValue()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & ", tPar_lam1030->GetFitValue(), tPar_R1030->GetFitValue()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & ", tPar_lam3050->GetFitValue(), tPar_R3050->GetFitValue()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f}", tPar_Ref0->GetFitValue(), tPar_Imf0->GetFitValue(), tPar_d0->GetFitValue()) << endl << endl;


}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriter::WriteLatexTableHelperSection(ostream &aOut, TString aMasterFileLocation, TString aFitInfoTString)
{
  WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aFitInfoTString, kLamKchP);
  WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aFitInfoTString, kALamKchM);
  WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aFitInfoTString, kLamKchM);
  WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aFitInfoTString, kALamKchP);
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriter::WriteLatexTableHelperHeader(ostream &aOut)
{
  aOut << "\%NOTE: Arrays seem to only be able to be named with alphabetical characters" << endl;
  aOut << "\%      No '_' or event numbers" << endl;
  aOut << endl;
  aOut << "\% ----------Shorthand definitions----------" << endl;
  aOut << "\% A = FitGen_NoLamShare" << endl;
  aOut << "\% B = FitGen_ShareLamConj" << endl;
  aOut << endl;
  aOut << "\% C = Dualie_NoLamShare" << endl;
  aOut << "\% D = Dualie_ShareLamConj" << endl;
  aOut << "\% E = Dualie_ShareSingleLam" << endl;
  aOut << endl;
  aOut << "\%   ------------------" << endl;
  aOut << "\% a = PolyNFB" << endl;
  aOut << "\% b = LinrNFB" << endl;
  aOut << "\% c = LinrNFB_StavCf" << endl;
  aOut << "\% d = NoNFB_StavCf" << endl;
  aOut << endl;
  aOut << "\% ----------How arrays are stored----------" << endl;
  aOut << "\% aArray = {lam0010(1), R0010(2)," << endl;
  aOut << "\%           lam1030(3), R1030(4)," << endl;
  aOut << "\%           lam3050(5), R3050(6)," << endl;
  aOut << "\%           ReF0(7), ImF0(8), d0(9)}" << endl;
  aOut << endl;
  aOut << endl;
  aOut << "\%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << endl;
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriter::WriteLatexTableHelper(TString aHelperLocation, TString aMasterFileLocation, IncludeResidualsType aResType)
{
  std::ofstream tOut;
  tOut.open(aHelperLocation);
  WriteLatexTableHelperHeader(tOut);
  //-------------------------------------------
  cout << "\% --------------- Aa = " << GetLatexTableOverallLabel("Aa") << endl;




}

