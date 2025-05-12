# Estimating Stellar Temperatures and Luminosities from UBV Photometry

**Author:** Moulik Mishra  
**Institution:** Shiv Nadar University  
**Project Type:** Observational Astrophysics & Computational Modeling  
**Programming Language:** Python  
**Status:** Completed

---

## Overview

This project uses real stellar photometric data to estimate fundamental physical properties of stars—namely, effective temperature, bolometric luminosity, and radius. UBV magnitudes and parallax-based distances are obtained from public catalogs (APASS DR9 and SIMBAD) using the `astroquery` API. Photometric color indices (B–V) are then mapped to temperature using empirical relations. Luminosities and radii are computed via the Stefan–Boltzmann law, with optional bolometric corrections applied.

The pipeline is fully automated, robust to missing data, and includes fallbacks for offline or failed queries. It also includes support for synthetic data simulation and visualization via Hertzsprung–Russell (HR) diagrams.

---

## Features

- Accesses **real stellar data** via Vizier/APASS and SIMBAD
- Computes:
  - Effective temperature from B–V color
  - Bolometric luminosity (absolute or normalized)
  - Stellar radius (via Stefan–Boltzmann law)
- Applies **Wien's displacement law** and **bolometric correction**
- Handles missing or uncertain data gracefully
- Generates:
  - HR diagram (Teff vs. Luminosity)
  - Summary statistics
  - Annotated CSV output
- Modular class-based structure for easy extension

---

## Project Structure

- `ubv_photometry.py` — Main script containing the `StellarPropertiesPipeline` class and example usage
- `results.csv` — Output table of all stars with computed properties
- `hr_diagram.png` — HR diagram generated from computed values

---

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scipy astroquery astropy
2. Run the script:
   ```bash
   python ubv_photometry.py
3. If the script fails to fetch data (due to network issues), it will automatically fall back to synthetic test stars.

## Output

- **results.csv**: A table containing:
  - Star name
  - B, V, B-V
  - Distance (pc)
  - Estimated effective temperature (K)
  - Luminosity (solar units)
  - Radius (solar units)

- **hr_diagram.png**: Plot of log(Luminosity) vs. log(Temperature), annotated

## Scientific Background

- **Color–Temperature Relation**: B–V color index is correlated with stellar temperature via empirical fits.

- **Stefan-Boltzmann Law**

- **Wien’s Law**

- **Bolometric Correction**: Adjusts visual magnitudes to estimate total luminosity.

## Future Work

- Add extinction correction using reddening maps or dust models
- Expand to additional photometric systems (e.g., Gaia BP/RP)
- Compare results with evolutionary tracks or isochrones
- Deploy as a web app for public/educational use

## Contact

**Email**: mm748@snu.edu.in  
**Affiliation**: Department of Physics, Shiv Nadar University
