from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import yaml
import joblib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

MODEL_URI = str(config['paths']['load_model'])

appEq1 = FastAPI(title='Turkish Music Emotion API', version='1.0')
MODEL_PATH = Path(__file__).resolve().parents[2] / config['paths']['working_model']
model = joblib.load(MODEL_PATH)

class InputData(BaseModel):
    RMSenergyMean: float = Field(..., alias="_RMSenergy_Mean")
    LowenergyMean: float = Field(..., alias="_Lowenergy_Mean")
    FluctuationMean: float = Field(..., alias="_Fluctuation_Mean")
    TempoMean: float = Field(..., alias="_Tempo_Mean")
    MFCCMean1: float = Field(..., alias="_MFCC_Mean_1")
    MFCCMean2: float = Field(..., alias="_MFCC_Mean_2")
    MFCCMean3: float = Field(..., alias="_MFCC_Mean_3")
    MFCCMean4: float = Field(..., alias="_MFCC_Mean_4")
    MFCCMean5: float = Field(..., alias="_MFCC_Mean_5")
    MFCCMean6: float = Field(..., alias="_MFCC_Mean_6")
    MFCCMean7: float = Field(..., alias="_MFCC_Mean_7")
    MFCCMean8: float = Field(..., alias="_MFCC_Mean_8")
    MFCCMean9: float = Field(..., alias="_MFCC_Mean_9")
    MFCCMean10: float = Field(..., alias="_MFCC_Mean_10")
    MFCCMean11: float = Field(..., alias="_MFCC_Mean_11")
    MFCCMean12: float = Field(..., alias="_MFCC_Mean_12")
    MFCCMean13: float = Field(..., alias="_MFCC_Mean_13")
    RoughnessMean: float = Field(..., alias="_Roughness_Mean")
    RoughnessSlope: float = Field(..., alias="_Roughness_Slope")
    ZeroCrossingrateMean: float = Field(..., alias="_Zero-crossingrate_Mean")
    AttackTimeMean: float = Field(..., alias="_AttackTime_Mean")
    AttackTimeSlope: float = Field(..., alias="_AttackTime_Slope")
    RolloffMean: float = Field(..., alias="_Rolloff_Mean")
    EventdensityMean: float = Field(..., alias="_Eventdensity_Mean")
    PulseclarityMean: float = Field(..., alias="_Pulseclarity_Mean")
    BrightnessMean: float = Field(..., alias="_Brightness_Mean")
    SpectralcentroidMean: float = Field(..., alias="_Spectralcentroid_Mean")
    SpectralspreadMean: float = Field(..., alias="_Spectralspread_Mean")
    SpectralskewnessMean: float = Field(..., alias="_Spectralskewness_Mean")
    SpectralkurtosisMean: float = Field(..., alias="_Spectralkurtosis_Mean")
    SpectralflatnessMean: float = Field(..., alias="_Spectralflatness_Mean")
    EntropyofSpectrumMean: float = Field(..., alias="_EntropyofSpectrum_Mean")
    ChromagramMean1: float = Field(..., alias="_Chromagram_Mean_1")
    ChromagramMean2: float = Field(..., alias="_Chromagram_Mean_2")
    ChromagramMean3: float = Field(..., alias="_Chromagram_Mean_3")
    ChromagramMean4: float = Field(..., alias="_Chromagram_Mean_4")
    ChromagramMean5: float = Field(..., alias="_Chromagram_Mean_5")
    ChromagramMean6: float = Field(..., alias="_Chromagram_Mean_6")
    ChromagramMean7: float = Field(..., alias="_Chromagram_Mean_7")
    ChromagramMean8: float = Field(..., alias="_Chromagram_Mean_8")
    ChromagramMean9: float = Field(..., alias="_Chromagram_Mean_9")
    ChromagramMean10: float = Field(..., alias="_Chromagram_Mean_10")
    ChromagramMean11: float = Field(..., alias="_Chromagram_Mean_11")
    ChromagramMean12: float = Field(..., alias="_Chromagram_Mean_12")
    HarmonicChangeDetectionFunctionMean: float = Field(..., alias="_HarmonicChangeDetectionFunction_Mean")
    HarmonicChangeDetectionFunctionStd: float = Field(..., alias="_HarmonicChangeDetectionFunction_Std")
    HarmonicChangeDetectionFunctionSlope: float = Field(..., alias="_HarmonicChangeDetectionFunction_Slope")
    HarmonicChangeDetectionFunctionPeriodFreq: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodFreq")
    HarmonicChangeDetectionFunctionPeriodAmp: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodAmp")
    HarmonicChangeDetectionFunctionPeriodEntropy: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodEntropy")

@appEq1.get("/")

def home():
    return {"status": "API Disponible", "modelo": "Turkish Music Emotion Equipo1"}

@appEq1.post("/predict")

def predict(data: InputData):
    df = pd.DataFrame([data.model_dump(by_alias=True)])

    pred = model.predict(df)
    return {"prediction": pred.tolist()}