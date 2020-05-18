import numpy as np


def flux2mag(flux, flux_err, zpt):

    flux = np.array(flux)
    flux_err = np.array(flux_err)
    zpt = np.array(zpt)
    flux[np.where(flux < .00001)] = .00001
    mag = 2.5 * np.log(flux) + zpt
    mag_err = 1.08574 * flux_err / flux

    return mag, mag_err
