import pandas as pd

def carga_datos(modificado, original):
  df1 = pd.read_csv(modificado)
  df2 = pd.read_csv(original)

  return df1, df2