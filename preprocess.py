import pandas as pd
import zipfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class FlightPreprocessing:
    """
    A class to preprocess flight pricing dataset.
    Handles data loading, cleaning, feature engineering, and validation.
    """

    def __init__(self, zip_file_path='Flight_prices_dataset.zip', csv_file_name='Data_Train.csv'):
        self.zip_file_path = zip_file_path
        self.csv_file_name = csv_file_name
        self.df = None

    def preprocess(self, verbose=True):
        if verbose:
            print("=" * 60)
            print("Starting Flight Data Preprocessing Pipeline")
            print("=" * 60)

        self._load_data(verbose)
        if verbose:
            self._display_initial_info()
        self._parse_departure_date()
        if verbose:
            print("\nParsed departure dates into day, month, year")

        self._parse_arrival_datetime()
        if verbose:
            print("Parsed arrival dates and times")

        self._validate_dates(verbose)
        self._parse_duration()
        if verbose:
            print("Converted duration to minutes")

        self._parse_total_stops()
        if verbose:
            print("Parsed total stops")

        self._parse_route()
        if verbose:
            print("Extracted route information (source, destination, stops)")

        self._handle_missing_values()
        if verbose:
            print("Handled missing values")

        self._add_festival_column(verbose)

        self._drop_redundant_columns(verbose)

        self._remove_duplicates(verbose)

        if verbose:
            print("\n" + "=" * 60)
            print("Preprocessing Complete!")
            print(f"Final shape: {self.df.shape}")
            print("=" * 60)

        return self.df

    def _load_data(self, verbose=True):
        """Load data from zip file."""
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall()

        self.df = pd.read_csv(self.csv_file_name)

        if verbose:
            print(f"\nLoaded data from {self.csv_file_name}")
            print(f"  Initial shape: {self.df.shape}")

    def _display_initial_info(self):
        """Display initial dataset information."""
        print("\n" + "-" * 60)
        print("Initial Dataset Information")
        print("-" * 60)
        print(f"Shape: {self.df.shape}")
        print(f"\nData Types:")
        print(self.df.info())
        print(f"\nMissing Values:")
        print(self.df.isnull().sum().sort_values(ascending=False).to_frame("missing"))
        print(f"\nDuplicates: {self.df.duplicated().sum()}")

    def _parse_departure_date(self):
        """Parse Date_of_Journey into day, month, and year columns."""
        self.df['Dep_Date'] = self.df['Date_of_Journey'].str.split('/').str[0].astype(int)
        self.df['Dep_Month'] = self.df['Date_of_Journey'].str.split('/').str[1].astype(int)
        self.df['Dep_Year'] = self.df['Date_of_Journey'].str.split('/').str[2].astype(int)

    def _parse_arrival_datetime(self):
        """Parse Arrival_Time into separate date and time columns."""
        # Split arrival time and date
        self.df[['Arrival_time', 'Arrival_Date']] = self.df['Arrival_Time'].str.split(' ', n=1, expand=True)
        self.df = self.df.rename(columns={'Arrival_time': 'Arrival_Time'})

        # Fill missing arrival dates with departure date
        self.df['Arrival_Date'] = self.df['Arrival_Date'].fillna(self.df['Date_of_Journey'])

        # Parse arrival date with year crossover handling
        self.df['Arrival_Date_Parsed'] = self.df.apply(
            lambda row: self._parse_single_arrival_date(
                row['Arrival_Date'],
                row['Date_of_Journey'],
                row['Arrival_Time']
            ),
            axis=1
        )

        # Extract day, month, year
        self.df['Arrival_Day'] = self.df['Arrival_Date_Parsed'].dt.day
        self.df['Arrival_Month'] = self.df['Arrival_Date_Parsed'].dt.month
        self.df['Arrival_Year'] = self.df['Arrival_Date_Parsed'].dt.year

        # Format to DD/MM/YYYY
        self.df['Arrival_Date'] = self.df['Arrival_Date_Parsed'].dt.strftime('%d/%m/%Y')

        # Drop temporary column
        self.df = self.df.drop('Arrival_Date_Parsed', axis=1)

    def _parse_single_arrival_date(self, date_str, journey_date, arrival_time):
        """
        Helper method to parse a single arrival date string.
        Handles year crossover for Dec 31 flights.
        """
        try:
            journey_dt = pd.to_datetime(journey_date, format='%d/%m/%Y')
            journey_year = journey_dt.year

            # If it's already in DD/MM/YYYY format
            if '/' in date_str:
                temp_date = pd.to_datetime(date_str, format='%d/%m/%Y')
                day = temp_date.day
                month = temp_date.month
            # If it's in "DD Mon" format
            else:
                temp_date = pd.to_datetime(f"{date_str} {journey_year}", format='%d %b %Y')
                day = temp_date.day
                month = temp_date.month

            # Default: use journey year
            year = journey_year

            # Check if journey date is Dec 31 and arrival crosses to Jan 1
            if journey_dt.month == 12 and journey_dt.day == 31:
                if month == 1 and day == 1:
                    year = journey_year + 1

            return pd.to_datetime(f"{day}/{month}/{year}", format='%d/%m/%Y')
        except:
            return pd.NaT

    def _validate_dates(self, verbose=True):
        """Validate that arrival dates are not before departure dates."""
        # Create datetime columns for comparison
        self.df['Departure_DateTime'] = pd.to_datetime(self.df['Date_of_Journey'], format='%d/%m/%Y')
        self.df['Arrival_DateTime'] = pd.to_datetime(self.df['Arrival_Date'], format='%d/%m/%Y')

        # Find invalid dates
        invalid_dates = self.df[self.df['Arrival_DateTime'] < self.df['Departure_DateTime']]

        if verbose and len(invalid_dates) > 0:
            print(f"\nFound {len(invalid_dates)} rows with arrival before departure")
            print("  Sample of invalid rows:")
            print(invalid_dates[['Date_of_Journey', 'Dep_Time', 'Arrival_Date', 'Arrival_Time']].head())

        # Create error flag and remove invalid rows
        self.df['Date_Error'] = self.df['Arrival_DateTime'] < self.df['Departure_DateTime']
        self.df = self.df[self.df['Date_Error'] == False]

        # Clean up temporary columns
        self.df = self.df.drop(['Departure_DateTime', 'Arrival_DateTime', 'Date_Error'], axis=1)

        if verbose:
            print(f"Removed {len(invalid_dates)} rows with invalid dates")

    def _parse_duration(self):
        """Convert duration string to minutes."""
        self.df['Duration_min'] = self.df['Duration'].apply(self._duration_to_minutes)

    def _duration_to_minutes(self, duration_str):
        """
        Helper method to convert duration string (e.g., '2h 30m') to minutes.
        """
        if pd.isna(duration_str):
            return np.nan

        duration_str = duration_str.lower().strip()
        hours = 0
        minutes = 0

        # Extract hours if present
        if 'h' in duration_str:
            hours_part = duration_str.split('h')[0].strip()
            try:
                hours = int(hours_part)
            except:
                hours = 0
            minutes_part = duration_str.split('h')[1]
        else:
            minutes_part = duration_str

        # Extract minutes if present
        if 'm' in minutes_part:
            try:
                minutes = int(minutes_part.split('m')[0].strip())
            except:
                minutes = 0

        return hours * 60 + minutes

    def _parse_total_stops(self):
        """Convert Total_Stops categorical values to numeric."""
        self.df['Total_Stops'] = self.df['Total_Stops'].map({
            'non-stop': 0,
            '1 stop': 1,
            '2 stops': 2,
            '3 stops': 3,
            '4 stops': 4,
            np.nan: 1  # Fill missing with mode (1 stop)
        })

    def _parse_route(self):
        """Extract source, destination, and intermediate stops from route."""
        # Split the route by delimiter
        self.df['Route_Split'] = self.df['Route'].str.split(' ? ')

        # Extract source (first element)
        self.df['Source'] = self.df['Route_Split'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )

        # Extract destination (last element)
        self.df['Destination'] = self.df['Route_Split'].apply(
            lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None
        )

        # Extract intermediate stops
        self.df['Stop1'] = self.df['Route_Split'].apply(
            lambda x: x[2] if isinstance(x, list) and len(x) > 3 else None
        )
        self.df['Stop2'] = self.df['Route_Split'].apply(
            lambda x: x[4] if isinstance(x, list) and len(x) > 5 else None
        )

    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        # Replace null values in stops with 'No stop'
        self.df['Stop1'] = self.df['Stop1'].fillna('No stop')
        self.df['Stop2'] = self.df['Stop2'].fillna('No stop')

        # Drop rows where Source or Destination is null
        self.df = self.df.dropna(subset=['Source', 'Destination'])

    def _drop_redundant_columns(self, verbose=True):
        """Drop columns that have been replaced by engineered features."""
        initial_cols = self.df.shape[1]
        redundant_cols = [
            'Date_of_Journey', 'Route', 'Dep_Time',
            'Arrival_Time', 'Duration', 'Route_Split'
        ]
        self.df.drop(columns=redundant_cols, inplace=True, errors='ignore')

        if verbose:
            print(f"Dropped {initial_cols - self.df.shape[1]} redundant columns")

    def _remove_duplicates(self, verbose=True):
        """Remove duplicate rows from the dataset."""
        initial_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates().reset_index(drop=True)

        if verbose:
            print(f"Removed {initial_rows - self.df.shape[0]} duplicate rows")

    def _add_festival_column(self, verbose=True):
        """Add festival/holiday column based on departure and arrival dates."""
        # Define festival date ranges (using day-month format)
        # Format: (start_day, start_month, end_day, end_month, festival_name)
        festivals = [
            # Republic Day
            (24, 1, 28, 1, 'Republic Day'),

            # Holi (varies but typically Feb 25 - Mar 17 range)
            (23, 2, 17, 3, 'Holi'),

            # Good Friday/Easter (varies March-April)
            (28, 3, 22, 4, 'Easter'),

            # Ram Navami (varies March-April)
            (3, 4, 8, 4, 'Ram Navami'),

            # Eid al-Fitr (moves throughout year - approximate ranges)
            (28, 3, 2, 4, 'Eid al-Fitr'),
            (18, 3, 23, 3, 'Eid al-Fitr'),
            (7, 3, 12, 3, 'Eid al-Fitr'),

            # Eid al-Adha (moves throughout year)
            (4, 6, 9, 6, 'Eid al-Adha'),
            (25, 5, 30, 5, 'Eid al-Adha'),
            (14, 5, 19, 5, 'Eid al-Adha'),

            # Independence Day
            (13, 8, 17, 8, 'Independence Day'),

            # Janmashtami (varies August)
            (13, 8, 18, 8, 'Janmashtami'),
            (22, 8, 27, 8, 'Janmashtami'),

            # Ganesh Chaturthi (varies late Aug - early Sept)
            (25, 8, 5, 9, 'Ganesh Chaturthi'),

            # Dussehra (varies Sept-Oct)
            (18, 9, 4, 10, 'Dussehra'),
            (28, 9, 14, 10, 'Dussehra'),
            (8, 10, 24, 10, 'Dussehra'),

            # Diwali (varies Oct-Nov)
            (17, 10, 25, 10, 'Diwali'),
            (25, 10, 2, 11, 'Diwali'),
            (4, 11, 12, 11, 'Diwali'),

            # Chhath Puja (6 days after Diwali)
            (25, 10, 30, 10, 'Chhath Puja'),
            (2, 11, 7, 11, 'Chhath Puja'),

            # Guru Nanak Jayanti (varies November)
            (3, 11, 6, 11, 'Guru Nanak Jayanti'),
            (22, 11, 25, 11, 'Guru Nanak Jayanti'),

            # Christmas/New Year
            (22, 12, 2, 1, 'Christmas-New Year'),

            # Summer Holiday Season (mid-April to mid-June)
            (15, 4, 15, 6, 'Summer Holidays'),
        ]

        def check_festival(dep_day, dep_month, arr_day, arr_month):
            """Check if departure or arrival falls in any festival period."""
            for start_day, start_month, end_day, end_month, festival_name in festivals:
                # Check departure date
                if self._is_date_in_range(dep_day, dep_month, start_day, start_month, end_day, end_month):
                    return festival_name
                # Check arrival date
                if self._is_date_in_range(arr_day, arr_month, start_day, start_month, end_day, end_month):
                    return festival_name
            return 'No Festival'

        # Apply festival detection
        self.df['Season_of_travel'] = self.df.apply(
            lambda row: check_festival(
                row['Dep_Date'], row['Dep_Month'],
                row['Arrival_Day'], row['Arrival_Month']
            ),
            axis=1
        )

        if verbose:
            festival_counts = self.df['Season_of_travel'].value_counts()
            print(f" Added Season_of_travel column")
            print(f"  Flights during festivals: {(self.df['Season_of_travel'] != 'No Festival').sum()}")
            print(f"  Top festivals in dataset:")
            for festival, count in festival_counts.head(5).items():
                print(f"    - {festival}: {count} flights")

    def _is_date_in_range(self, day, month, start_day, start_month, end_day, end_month):
        """
        Check if a date (day, month) falls within a date range.
        Handles ranges that cross year boundaries (e.g., Dec 22 - Jan 2).
        """
        # Convert to comparable format (month * 100 + day)
        date_val = month * 100 + day
        start_val = start_month * 100 + start_day
        end_val = end_month * 100 + end_day

        # If range doesn't cross year boundary
        if start_val <= end_val:
            return start_val <= date_val <= end_val
        # If range crosses year boundary (e.g., Dec to Jan)
        else:
            return date_val >= start_val or date_val <= end_val

    def get_preprocessed_data(self):
        """Return the preprocessed dataframe."""
        if self.df is None:
            raise ValueError("Data has not been preprocessed yet. Call preprocess() first.")
        return self.df
    
# if __name__ == "__main__":
  # Preprocess training data
train_preprocessor = FlightPreprocessing(
    zip_file_path='Flight_prices_dataset.zip',
    csv_file_name='Data_Train.csv'
)
df_train = train_preprocessor.preprocess(verbose=True)

# Preprocess test data
test_preprocessor = FlightPreprocessing(
    zip_file_path='Flight_prices_dataset.zip',
    csv_file_name='Test_set.csv'
)
df_test = test_preprocessor.preprocess(verbose=True)

print("\nFinal train dataset info:")
print(df_train.info())
print("\nFinal test dataset info:")
print(df_test.info())

  

def plot_full_correlation_heatmap(df, figsize=(18, 14)):
    """
    Plot correlation heatmap with ALL features (including encoded categorical).

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    figsize : tuple
        Figure size (width, height)
    """
    # Creating a copy to avoid modifying original
    df_encoded = df.copy()

    # Encode all categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))

    # Drop original categorical columns, keep only encoded versions
    df_encoded = df_encoded.drop(columns=categorical_cols)

    # Calculate correlation matrix
    corr_matrix = df_encoded.corr()

    # Create figure
    plt.figure(figsize=(8, 8))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,           # Show correlation values
        fmt='.2f',            # Format to 2 decimal places
        cmap='coolwarm',      # Color scheme
        center=0,             # Center colormap at 0
        square=True,          # Make cells square
        linewidths=0.5,       # Add gridlines
        cbar_kws={'shrink': 0.8},
        annot_kws={'size': 8}  # Smaller font for annotations
    )

    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Print top correlations with Price
    if 'Price' in corr_matrix.columns:
        price_corr = corr_matrix['Price'].sort_values(ascending=False)
        print("\nTop 10 Features Most Correlated with Price:")
        print("=" * 60)
        for i, (feature, corr_val) in enumerate(price_corr.items(), 1):
            if feature != 'Price' and i <= 11:  # Top 10 (excluding Price itself)
                print(f"  {i:2d}. {feature:30s}: {corr_val:7.3f}")


# Use after preprocessing
plot_full_correlation_heatmap(df_train)