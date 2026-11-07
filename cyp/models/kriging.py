
import numpy as np
import gstools as gs


class SpatioTemporalResidualKriging:
    

    EARTH_RADIUS_KM = 6371.0088

    def __init__(
        self,
        r_loc_km=100.0,
        r_year=2.0,
        nugget_fraction=0.25,
        n_neighbors=100,
        jitter=1e-8,
    ):

        self.r_loc_km = float(r_loc_km)
        self.r_year = float(r_year)
        self.nugget_fraction = float(nugget_fraction)

        self.n_neighbors = n_neighbors
        self.jitter = jitter

        self.model = None
        self.kriging = None
        self.fitted = False

    @staticmethod
    def _validate_locations(locations):
        print("I AM USING THE NEW KRIGING ")

        locations = np.asarray(
            locations,
            dtype=float,
        )

        if locations.ndim != 2 or locations.shape[1] != 2:
            raise ValueError(
                "Locations must have shape (n, 2): [longitude, latitude]."
            )

        if not np.isfinite(locations).all():
            raise ValueError(
                "Locations contain NaN or infinite values."
            )

        if (
            np.any(np.abs(locations[:, 0]) > 180)
            or np.any(np.abs(locations[:, 1]) > 90)
        ):
            raise ValueError(
                "Invalid longitude/latitude values."
            )

        return locations

    def _transform_coordinates(self, locations, years):

        longitude = locations[:, 0]
        latitude = locations[:, 1]

        delta_lon = (
            longitude - self.lon_origin + 180.0
        ) % 360.0 - 180.0

        delta_lat = latitude - self.lat_origin

        x = (
            self.EARTH_RADIUS_KM
            * np.cos(np.radians(self.lat_origin))
            * np.radians(delta_lon)
        )

        y = (
            self.EARTH_RADIUS_KM
            * np.radians(delta_lat)
        )

        z = (
            years - self.year_origin
        ) * (
            self.r_loc_km / self.r_year
        )

        return x, y, z

    def fit(self, locations, years, residuals):

        locations = self._validate_locations(locations)

        years = np.asarray(
            years,
            dtype=float,
        ).reshape(-1)

        residuals = np.asarray(
            residuals,
            dtype=float,
        ).reshape(-1)

        if len(locations) != len(residuals):
            raise ValueError(
                "Locations and residuals must match."
            )

        if len(years) != len(residuals):
            raise ValueError(
                "Years and residuals must match."
            )

        if len(residuals) == 0:
            raise ValueError(
                "No residuals supplied."
            )

        if not (
            np.isfinite(years).all()
            and np.isfinite(residuals).all()
        ):
            raise ValueError(
                "Non-finite values detected."
            )

        if self.r_loc_km <= 0 or self.r_year <= 0:
            raise ValueError(
                "Spatial and temporal length scales must be positive."
            )

        if not 0 <= self.nugget_fraction < 1:
            raise ValueError(
                "Nugget out of [0, 1)."
            )

        print(
            f"GSTools: fitting {len(residuals)} residual observations..."
        )

        print(
            "Residual years:",np.sort(np.unique(years)),
        )

        self.residual_mean = np.mean(residuals)
        residual_variance = np.var(residuals)
        self.residual_variance = residual_variance

        if residual_variance < 1e-12:

            self.model = None
            self.kriging = None
            self.fitted = True

            print(
                "Residual variance approximately zero so using mean residual correction."
            )

            return self

        structured_variance = (
            residual_variance
            * (1.0 - self.nugget_fraction)
        )

        nugget = (
            residual_variance
            * self.nugget_fraction
        )

        self.lon_origin = float(
            np.mean(locations[:, 0])
        )

        self.lat_origin = float(
            np.mean(locations[:, 1])
        )

        self.year_origin = float(
            np.mean(years)
        )

        x, y, z = self._transform_coordinates(
            locations,years,
        )

        conditioning_positions = (
            x,y,z,
        )

        self.model = gs.Matern(
            dim=3,
            var=structured_variance,
            len_scale=self.r_loc_km,
            nugget=nugget,
            nu=0.25,
        )

        self.kriging = gs.krige.Ordinary(
            model=self.model,
            cond_pos=conditioning_positions,
            cond_val=residuals,
            exact=False,
            cond_err="nugget",
            pseudo_inv=True,
            pseudo_inv_type="pinv",
        )

        self.fitted = True

        return self

    def predict(self, locations, years):

        if not self.fitted:
            raise RuntimeError(
                "Kriging model must be fitted first."
            )

        locations = self._validate_locations(locations)

        years = np.asarray(
            years,
            dtype=float,
        ).reshape(-1)

        if len(locations) != len(years):
            raise ValueError(
                "Locations and years must match."
            )

        if not np.isfinite(years).all():
            raise ValueError(
                "Prediction years contain non-finite values."
            )

        if len(locations) == 0:
            return np.array([], dtype=float)

        if self.kriging is None:

            return np.full(
                len(locations),
                self.residual_mean,
                dtype=float,
            )

        x, y, z = self._transform_coordinates(
            locations,years,
        )

        prediction_positions = (
            x,y,z,
        )

        predictions = self.kriging(
            prediction_positions,
            mesh_type="unstructured",
            return_var=False,
            store=False,
        )

        predictions = np.asarray(
            predictions,
            dtype=float,
        ).reshape(-1)

        if not np.isfinite(predictions).all():
            raise ValueError(
                "Ummmmm why is this happening."
            )

        return predictions
