/* FitGeneratorAndDraw.h */
/* Purpose is to separate out all of the draw methods, which were previously located in 
    FitGenerator class, from the other methods directly relating to the fit
*/

#ifndef FITGENERATORANDDRAW_H
#define FITGENERATORANDDRAW_H


#include "FitGenerator.h"
#include "FitValuesWriterwSysErrs.h"
#include "TMarker.h"

class FitGeneratorAndDraw : public FitGenerator {

public:
  FitGeneratorAndDraw(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="", bool aUseStavCf=false);

  FitGeneratorAndDraw(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType=kMB, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="", bool aUseStavCf=false);
  virtual ~FitGeneratorAndDraw();


  static void SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, 
                 float aTitleSize=0.05, float aTitleOffset=0.9, bool aCenterTitle=false, float aLabelSize=0.03, float aLabelOffset=0.005, int aNdivisions=510);
  static void SetupAxis(TAxis* aAxis, TString aTitle, 
                 float aTitleSize=0.05, float aTitleOffset=0.9, bool aCenterTitle=false, float aLabelSize=0.03, float aLabelOffset=0.005, int aNdivisions=510);

  void CreateParamInitValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15);
  void CreateParamFinalValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15);
  void CreateParamFinalValuesText(AnalysisType aAnType, CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const td1dVec &aSysErrors, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15, bool aDrawAll=true);
  void CreateParamFinalValuesTextTwoColumns(CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const td1dVec &aSysErrors, double aText1Xmin=0.75, double aText1Ymin=0.75, double aText1Width=0.15, double aText1Height=0.10, bool aDrawText1=true, double aText2Xmin=0.50, double aText2Ymin=0.75, double aText2Width=0.15, double aText2Height=0.10, bool aDrawText2=true, double aTextFont=63, double aTextSize=15);
  void AddTextCorrectionInfo(CanvasPartition *aCanPart, int aNx, int aNy, bool aMomResCorrect, bool aNonFlatCorrect, double aTextXmin=0.75, double aTextYmin=0.75, double aTextWidth=0.15, double aTextHeight=0.10, double aTextFont=63, double aTextSize=15);
  void AddColoredLinesLabels(CanvasPartition *aCanPart, int aNx, int aNy, bool aZoomROP=true, double aScaleFactor=1.);
  void AddColoredLinesLabelsAndData(CanvasPartition *aCanPart, int aNx, int aNy, bool aZoomROP=true, double aScaleFactor=1.);

  void DrawSingleKStarCf(TPad* aPad, int aPairAnNumber, double aYmin=0.9, double aYmax=1.1, double aXmin=0.0, double aXmax=0.5, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  void DrawSingleKStarCfwFit(TPad* aPad, int aPairAnNumber, double aYmin=0.9, double aYmax=1.1, double aXmin=0.0, double aXmax=0.5, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  virtual TCanvas* DrawKStarCfs(bool aSaveImage=false, bool aDrawSysErrors=true);

  static TH1D* Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle = "tCf");
  static td1dVec GetSystErrs(IncludeResidualsType aIncResType, AnalysisType aAnType, CentralityType aCentType);
  static td1dVec GetSystErrs(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType);

  virtual void BuildKStarCfswFitsPanel_PartAn(CanvasPartition* aCanPart, int aAnalysisNumber, BFieldType aBFieldType, int tColumn, int tRow, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawDataOnTop);
  virtual CanvasPartition* BuildKStarCfswFitsCanvasPartition_PartAn(BFieldType aBFieldType, TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aZoomROP=true);
  virtual TCanvas* DrawKStarCfswFits_PartAn(BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aSaveImage=false, bool aZoomROP=true);

  virtual void BuildKStarCfswFitsPanel(CanvasPartition* aCanPart, int aAnalysisNumber, int tColumn, int tRow, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawSysErrors, bool aDrawDataOnTop);
  virtual CanvasPartition* BuildKStarCfswFitsCanvasPartition(TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aDrawSysErrors=true, bool aZoomROP=true, bool aSuppressFitInfoOutput=false, bool aLabelLines=false);
  virtual TCanvas* DrawKStarCfswFits(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aSuppressFitInfoOutput=false, bool aLabelLines=false);

  double GetWeightedAnalysisNorm(FitPairAnalysis* aPairAn);
  virtual TCanvas* DrawResiduals(int aAnalysisNumber, CentralityType aCentralityType=k0010, TString aCanvasName="Residuals", bool aSaveImage=false);
  virtual TObjArray* DrawAllResiduals(bool aSaveImage=false);

  template <typename T>
  TCanvas* GetResidualsWithTransformMatrices(int aAnalysisNumber, T& aResidual, td1dVec &aParamsOverall, int aOffset=0);
  template <typename T>
  TCanvas* GetResidualsWithTransformMatricesv2(int aAnalysisNumber, T& aResidual, td1dVec &aParamsOverall, int aOffset=0);
  virtual TObjArray* DrawResidualsWithTransformMatrices(int aAnalysisNumber, bool aSaveImage=false, bool aDrawv2=false);
  virtual TObjArray* DrawAllResidualsWithTransformMatrices(bool aSaveImage=false, bool aDrawv2=false);

  void CheckCorrectedCf_PartAn(int aAnalysisNumber, BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType);
  virtual TCanvas* DrawSingleKStarCfwFitAndResiduals_PartAn(int aAnalysisNumber, BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage=false, bool aZoomROP=true, bool aOutputCheckCorrectedCf=false);
  virtual TObjArray* DrawAllSingleKStarCfwFitAndResiduals_PartAn(BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage=false, bool aZoomROP=true, bool aOutputCheckCorrectedCf=false);

  void CheckCorrectedCf(int aAnalysisNumber, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType);
  virtual TCanvas* DrawSingleKStarCfwFitAndResiduals(int aAnalysisNumber, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aOutputCheckCorrectedCf=false);
  virtual TObjArray* DrawAllSingleKStarCfwFitAndResiduals(bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aOutputCheckCorrectedCf=false);
  virtual TCanvas* DrawKStarCfswFitsAndResiduals(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aZoomResiduals=false);

  virtual TCanvas* DrawModelKStarCfs(bool aSaveImage=false);  //TODO add option to choose true, fake, no weight, etc.


  static CfHeavy* CombineTwoHeavyCfs(CfHeavy *aCf1, CfHeavy* aCf2);
  static TH1* CombineTwoHists(TH1* aHist1, TH1* aHist2, double aNorm1, double aNorm2);
  virtual void BuildKStarCfswFitsPanel_CombineConj(CanvasPartition* aCanPart, int aAnNumber, int aConjAnNumber, int tColumn, int tRow, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawSysErrors, bool aDrawDataOnTop);
  virtual CanvasPartition* BuildKStarCfswFitsCanvasPartition_CombineConj(TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aDrawSysErrors=true, bool aZoomROP=true, bool aSuppressFitInfoOutput=false, bool aLabelLines=false);
  virtual TCanvas* DrawKStarCfswFits_CombineConj(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aSuppressFitInfoOutput=false, bool aLabelLines=false);

  virtual void BuildKStarCfsPanel_CombineConj(CanvasPartition* aCanPart, int aAnNumber, int aConjAnNumber, int tColumn, int tRow, bool aDrawSysErrors, bool aDrawDataOnTop);

  //inline 


protected:




#ifdef __ROOT__
  ClassDef(FitGeneratorAndDraw, 1)
#endif
};




#endif
