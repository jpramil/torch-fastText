import s3fs
import sys
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from utils import add_libelles, clean_and_tokenize_df, stratified_split_rare_labels
from sklearn.preprocessing import LabelEncoder
sys.path.append("../")
from torchFastText import torchFastText
from torchFastText.preprocess import clean_text_feature
sys.path.append("./notebooks")

def get_data():
    fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
    anon=True,
    )

    df = (
    pq.ParquetDataset(
        "projet-ape/extractions/20241027_sirene4.parquet",
        filesystem=fs,
    )
    .read_pandas()
    .to_pandas()
    ).sample(frac=0.001).fillna(np.nan)

    with fs.open("projet-ape/data/naf2008.csv") as file:
        naf2008 = pd.read_csv(file, sep=";")

    categorical_features = ["evenement_type", "cj",  "activ_nat_et", "liasse_type", "activ_surf_et", "activ_perm_et"]
    text_feature = "libelle"
    y = "apet_finale"
    textual_features = None

    df = add_libelles(df, naf2008, y, text_feature, textual_features, categorical_features)

    df["libelle_processed"] = clean_text_feature(df["libelle"])

    encoder = LabelEncoder()
    df["apet_finale"] = encoder.fit_transform(df["apet_finale"])

    df, _ = clean_and_tokenize_df(df, text_feature="libelle_processed")
    df = df.sample(frac=1).reset_index(drop=True) # shuffling
    X = df[["libelle_processed", "EVT", "CJ", "NAT", "TYP", "CRT", "SRF"]].values
    y = df["apet_finale"].values

    X_train, X_test, y_train, y_test = stratified_split_rare_labels(X, y)
    assert set(range(len(naf2008["code"]))) == set(np.unique(y_train))
    return X_train, X_test, y_train, y_test