from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType

import pandas as pd
import numpy as np

import os

from Predict_lc import PredictLightCurve
from io_utils import create_alert_data_obj
from fink_science.conversion import mag2fluxcal_snana

from fink_science import __file__
from fink_science.utilities import load_scikit_model
from fink_science.random_forest_snia.classifier_bazin import fit_all_bands
from fink_science.random_forest_snia.classifier_sigmoid import get_sigmoid_features_dev

from fink_science.tester import spark_unit_tests
from astropy.table import Table

import pickle

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_pca(jd, fid, magpsf, sigmapsf, model=None, bands=None, num_pc_components= None, min_flux_threshold=None) -> pd.Series:

    if bands is None:
        bands = ['g', 'r']
    if num_pc_components is None:
        num_pc_components = 3
    if min_flux_threshold is None:
        min_flux_threshold = 200
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 3
    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])

    #need to define this later
    else:
        clf = load_scikit_model("models/pickle_model.pkl")

    #else:
    #    curdir = os.path.dirname(os.path.abspath(__file__))
    #    model = curdir + 'models/pickle_model.pkl'
    #
    #    clf = load_scikit_model(model)

        #remember to initialize bands and

    test_features = []
    ids = pd.Series(range(len(jd)))
    for id in ids[mask]:
        # compute flux and flux error
        data = [mag2fluxcal_snana(*args) for args in zip(
            magpsf[id],
            sigmapsf[id])]
        flux, error = np.transpose(data)

        # make a Pandas DataFrame with exploded series
        pdf_id = [id] * len(flux)
        pdf = pd.DataFrame.from_dict({
            'SNID': [int(i) for i in pdf_id],
            'MJD': [int(i) for i in jd[id]],
            'FLUXCAL': flux,
            'FLUXCALERR': error,
            'FLT': pd.Series(fid[id]).replace({1: 'g', 2: 'r'})
        })

        pdf = Table.from_pandas(pdf)
        data_obj = create_alert_data_obj(pdf, bands)

        # move to dataframe class
        pc = PredictLightCurve(data_obj, object_id=pdf['SNID'][0])
        coeff_dict, num_pts_dict = pc.predict_lc_coeff(current_date=None,
                                                       num_pc_components=3,
                                                       decouple_pc_bands=False,
                                                       decouple_prediction_bands=True,
                                                       min_flux_threshold=min_flux_threshold,
                                                       bands=bands,
                                                       band_choice='u')

        features = np.zeros((num_pc_components + 1) * len(bands))
        for i, band in enumerate(bands):
            for j in range(num_pc_components):
                if j == 0:
                    features[i * 4] = num_pts_dict[band]
                features[i * 4 + j + 1] = coeff_dict[band][j]

        test_features.append(features)

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # Take only probabilities to be KN
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities.T[1]

    return pd.Series(to_return)


