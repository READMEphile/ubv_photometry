import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
import astropy.constants as const
from scipy.interpolate import interp1d
import warnings
import logging
import requests
from requests.exceptions import Timeout, RequestException
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Vizier timeout
Vizier.TIMEOUT = 30

class StellarPropertiesPipeline:
    def __init__(self):
        """Initialize the pipeline with physical constants and interpolators"""
        # Physical constants
        self.stefan_boltzmann = const.sigma_sb.value  # W m^-2 K^-4
        self.solar_temperature = 5778  # K
        self.solar_luminosity = 3.828e26  # W
        self.solar_radius = 6.957e8  # m

        # Setup SIMBAD
        self.custom_simbad = Simbad()
        self.custom_simbad.add_votable_fields('parallax', 'sptype', 'flux(V)', 'flux(B)')
        self.custom_simbad.TIMEOUT = 30

        # Color-temperature and bolometric correction relations (Flower 1996)
        self._setup_interpolators()
        self.stars_data = None
        self.simbad_cache = {}  # Cache for SIMBAD results

    def _setup_interpolators(self):
        """Setup interpolation functions for B-V to temperature and bolometric correction"""
        bv_points = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 
                              0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        temp_points = np.array([33000, 26000, 20000, 14000, 10000, 8600, 7600, 7000, 6600, 6200, 
                                5800, 5400, 5000, 4600, 4200, 4000, 3800, 3600, 3400, 3200, 3000, 2800, 2600, 2400, 2200])
        bc_points = np.array([-3.6, -3.0, -2.4, -1.2, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 
                              -0.1, -0.15, -0.3, -0.5, -0.7, -1.0, -1.4, -1.8, -2.2, -2.6, -3.0, -3.4, -3.8, -4.2, -4.6])
        self.bv_to_temp = interp1d(bv_points, temp_points, kind='cubic', bounds_error=False, fill_value="extrapolate")
        self.bv_to_bc = interp1d(bv_points, bc_points, kind='cubic', bounds_error=False, fill_value="extrapolate")

    def _check_network(self):
        """Check network connectivity to Vizier"""
        try:
            response = requests.get("http://vizier.u-strasbg.fr/viz-bin/VizieR", timeout=5)
            logger.info(f"Network check to Vizier: status code {response.status_code}")
            return response.status_code == 200
        except RequestException as e:
            logger.error(f"Network connectivity to Vizier failed: {e}")
            return False

    def query_vizier(self, limit=300):
        """Query APASS DR9 catalog for bright stars"""
        logger.info(f"Querying APASS DR9 for up to {limit} stars")
        try:
            vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag', 'Vmag'], row_limit=limit)
            result = vizier.query_constraints(catalog="II/336/apass9", Vmag="<10")
            if len(result) > 0:
                data = result[0].to_pandas()
                logger.info(f"Retrieved {len(data)} stars from APASS DR9. Columns: {list(data.columns)}")
                return data
            logger.warning("No results from APASS DR9")
            return None
        except Timeout:
            logger.error("Timeout querying APASS DR9")
            return None
        except Exception as e:
            logger.error(f"Error querying APASS DR9: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _query_simbad_single(self, coords):
        """Query SIMBAD for a single star with retry"""
        cache_key = f"{coords.ra.deg:.6f}_{coords.dec.deg:.6f}"
        if cache_key in self.simbad_cache:
            logger.debug(f"Cache hit for SIMBAD query at {cache_key}")
            return self.simbad_cache[cache_key]
        
        try:
            result = self.custom_simbad.query_region(coords, radius=10*u.arcsec)
            if result is not None and len(result) > 0:
                data = {
                    'parallax': result['PLX_VALUE'][0] if 'PLX_VALUE' in result.colnames and result['PLX_VALUE'][0] is not None else np.nan,
                    'spectral_type': result['SP_TYPE'][0] if 'SP_TYPE' in result.colnames and result['SP_TYPE'][0] else ''
                }
                self.simbad_cache[cache_key] = data
                return data
            return {'parallax': np.nan, 'spectral_type': ''}
        except Exception as e:
            logger.warning(f"Error querying SIMBAD for star at {coords}: {e}")
            return {'parallax': np.nan, 'spectral_type': ''}

    def query_simbad(self, stars):
        """Query SIMBAD for parallaxes and spectral types"""
        logger.info(f"Querying SIMBAD for {len(stars)} stars")
        stars['parallax'] = np.nan
        stars['spectral_type'] = ''
        success_count = 0
        for idx, star in stars.iterrows():
            try:
                coords = SkyCoord(ra=star['ra']*u.degree, dec=star['dec']*u.degree)
                result = self._query_simbad_single(coords)
                stars.at[idx, 'parallax'] = result['parallax']
                stars.at[idx, 'spectral_type'] = result['spectral_type']
                if not np.isnan(result['parallax']):
                    success_count += 1
            except Exception as e:
                logger.warning(f"Failed to process SIMBAD query for star at {coords}: {e}")
        logger.info(f"Retrieved valid parallaxes for {success_count} stars")
        return stars

    def create_simulated_data(self, num_stars=100):
        """Generate simulated star data"""
        logger.info(f"Generating simulated data for {num_stars} stars")
        np.random.seed(42)
        stars = pd.DataFrame({
            'ra': np.random.uniform(0, 360, num_stars),
            'dec': np.random.uniform(-90, 90, num_stars),
            'B-V': np.clip(np.random.normal(0.8, 0.4, num_stars), -0.4, 2.0),
            'Vmag': np.clip(np.random.normal(7.0, 1.5, num_stars), 0, 12),
            'parallax': np.clip(np.random.lognormal(2, 1, num_stars), 2, 200)
        })
        stars['Bmag'] = stars['Vmag'] + stars['B-V']
        stars['distance_pc'] = 1000 / stars['parallax']
        stars['spectral_type'] = stars['B-V'].apply(
            lambda bv: next((t for v, t in [(-0.3, 'O'), (0.0, 'B'), (0.3, 'A'), (0.5, 'F'), 
                                             (0.8, 'G'), (1.3, 'K'), (float('inf'), 'M')] if bv < v), 'M')
        )
        return stars

    def fetch_star_data(self, limit=300):
        """Fetch star data from APASS or generate simulated data"""
        if self._check_network():
            data = self.query_vizier(limit)
            if data is not None and len(data) > 0:
                stars = pd.DataFrame({
                    'ra': data['RAJ2000'],
                    'dec': data['DEJ2000'],
                    'Bmag': data['Bmag'],
                    'Vmag': data['Vmag']
                })
                stars['B-V'] = stars['Bmag'] - stars['Vmag']
                stars = self.query_simbad(stars)
                # Keep all stars, even without parallaxes
                stars['distance_pc'] = np.where(stars['parallax'] > 0, 1000 / stars['parallax'], np.nan)
                self.stars_data = stars
                valid_stars = len(stars[stars['parallax'] > 0])
                logger.info(f"Processed {len(stars)} stars, {valid_stars} with valid parallaxes")
                if valid_stars == 0:
                    logger.warning("No stars with valid parallaxes. Using simulated data")
                    self.stars_data = self.create_simulated_data(num_stars=min(limit, 100))
                return self.stars_data
        logger.warning("Falling back to simulated data due to network issues")
        self.stars_data = self.create_simulated_data(num_stars=min(limit, 100))
        return self.stars_data

    def estimate_properties(self):
        """Estimate stellar properties using B-V, parallaxes, and physical laws"""
        if self.stars_data is None or len(self.stars_data) == 0:
            logger.error("No star data available")
            return None
        stars = self.stars_data.copy()
        
        # Effective temperature from B-V
        stars['teff'] = self.bv_to_temp(stars['B-V'])
        
        # Initialize columns
        stars['M_V'] = np.nan
        stars['BC'] = np.nan
        stars['M_bol'] = np.nan
        stars['luminosity_lsun'] = np.nan
        stars['radius_rsun'] = np.nan
        
        # Calculate properties only for stars with valid parallaxes and distances
        valid = (stars['parallax'] > 0) & (~stars['distance_pc'].isna())
        if valid.sum() > 0:
            stars.loc[valid, 'M_V'] = stars.loc[valid, 'Vmag'] - 5 * np.log10(stars.loc[valid, 'distance_pc']) + 5
            stars.loc[valid, 'BC'] = self.bv_to_bc(stars.loc[valid, 'B-V'])
            stars.loc[valid, 'M_bol'] = stars.loc[valid, 'M_V'] + stars.loc[valid, 'BC']
            stars.loc[valid, 'luminosity_lsun'] = 10**((4.74 - stars.loc[valid, 'M_bol']) / 2.5)
            stars.loc[valid, 'radius_rsun'] = np.sqrt(stars.loc[valid, 'luminosity_lsun']) * (self.solar_temperature / stars.loc[valid, 'teff'])**2
        
        self.stars_data = stars
        logger.info(f"Estimated properties for {valid.sum()} stars with valid parallaxes")
        return stars

    def create_hr_diagram(self):
        """Create Hertzsprung-Russell diagram for stars with valid luminosities"""
        if self.stars_data is None or len(self.stars_data) == 0:
            logger.error("No data available for plotting")
            return
        valid = self.stars_data['luminosity_lsun'].notna()
        if valid.sum() == 0:
            logger.error("No stars with valid luminosities for HR diagram")
            return
        plt.figure(figsize=(10, 8))
        plt.scatter(self.stars_data.loc[valid, 'teff'], 
                   self.stars_data.loc[valid, 'luminosity_lsun'], 
                   c=self.stars_data.loc[valid, 'radius_rsun'], cmap='viridis', 
                   norm=plt.Normalize(vmin=0.1, vmax=100), s=50, alpha=0.7)
        plt.colorbar(label='Radius ($R_\\odot$)')
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().invert_xaxis()
        plt.xlabel('Effective Temperature (K)')
        plt.ylabel('Luminosity ($L_\\odot$)')
        plt.title('Hertzsprung-Russell Diagram')
        plt.scatter([self.solar_temperature], [1], c='yellow', s=150, edgecolor='orange', label='Sun')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('hr_diagram.png', dpi=300)
        plt.close()
        logger.info("HR diagram saved as 'hr_diagram.png'")

    def generate_summary(self):
        """Generate summary statistics"""
        if self.stars_data is None or len(self.stars_data) == 0:
            logger.error("No data available for summary")
            return
        valid = self.stars_data['luminosity_lsun'].notna()
        summary = [
            f"\n===== Stellar Sample Summary =====",
            f"Total stars: {len(self.stars_data)}",
            f"Stars with valid parallaxes: {valid.sum()}",
            f"Temperature (K): Min={self.stars_data['teff'].min():.1f}, Max={self.stars_data['teff'].max():.1f}, Mean={self.stars_data['teff'].mean():.1f}",
        ]
        if valid.sum() > 0:
            summary.extend([
                f"Luminosity (L☉): Min={self.stars_data.loc[valid, 'luminosity_lsun'].min():.3f}, Max={self.stars_data.loc[valid, 'luminosity_lsun'].max():.3f}, Mean={self.stars_data.loc[valid, 'luminosity_lsun'].mean():.3f}",
                f"Radius (R☉): Min={self.stars_data.loc[valid, 'radius_rsun'].min():.3f}, Max={self.stars_data.loc[valid, 'radius_rsun'].max():.3f}, Mean={self.stars_data.loc[valid, 'radius_rsun'].mean():.3f}",
                f"Distance (pc): Min={self.stars_data.loc[valid, 'distance_pc'].min():.1f}, Max={self.stars_data.loc[valid, 'distance_pc'].max():.1f}, Mean={self.stars_data.loc[valid, 'distance_pc'].mean():.1f}"
            ])
        print("\n".join(summary))
        self.stars_data.to_csv('stellar_properties.csv', index=False)
        logger.info("Exported data to 'stellar_properties.csv'")

    def run(self, num_stars=300):
        """Run the full pipeline"""
        logger.info("Starting stellar properties pipeline")
        self.fetch_star_data(limit=num_stars)
        if self.stars_data is None or len(self.stars_data) == 0:
            logger.error("Failed to retrieve star data")
            return
        self.estimate_properties()
        self.create_hr_diagram()
        self.generate_summary()

if __name__ == "__main__":
    pipeline = StellarPropertiesPipeline()
    pipeline.run(num_stars=300)