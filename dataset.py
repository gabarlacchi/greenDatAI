from serveo_api import GetDataFromAPIDB
import pandas as pd 
from tqdm import tqdm 
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame as dd 
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
from darts.dataprocessing.transformers import Scaler
import os
import seaborn as sns

class ServeoDataset():
    """
        Utils for the models, refer to serveo dataset.
        It uses API from serveo_api to build the dataset as requested 
    """

    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.df_trips, self.df_stations_docks = None, None
        self.datasets_paths = f"./cache/datasets/stationId=&_{self.config.cold_start_datestart}_{self.config.cold_start_datend}.csv"
        self.stations_occupancy_df_cache = f"./cache/stations_df/cache_stationOccupancy_stationID=&_{self.config.cold_start_datestart}_{self.config.cold_start_datend}.csv"
        self.plots_path_root = f"./plots/serveo"
        self.main_df = None

        self.total_data_trips = None

        self.df_occupancy = None # Call plot statistics to fill it

        self.inop_grouped, self.op_grouped = None, None

        self.status = {}
       

    def __init_dataset__(self, **kwargs):
        """
            Init the dataset. The main task is to predict docks available for one or more station
            The initialization of the dataset start from trips data
        """

        self.serveo_data = GetDataFromAPIDB(config=self.config)
        if self.config.limit_to_station > 1:   
            p = self.serveo_data.occupancy_cache.replace(".csv", "")
            p += f"_stationId={self.config.limit_to_station}.csv"
        else:
            p = self.serveo_data.occupancy_cache

        self.serveo_data.init_data()
        
        if not self.config.use_cache_occupancy:
            # self.occupancy_from_trips()
            self.occupancy_from_trips_v2()
        else:
            self.trips = pd.read_csv(filepath_or_buffer=self.stations_occupancy_df_cache.replace("&", str(self.config.analyze_station))) 
    
    def clean_and_build_dataset(self, df: dd, group_name: str = "", **kwargs):
        """
            Given a dataframe, build a dataset istance for dart
        """ 
        
        df[f"{self.config.timestamp}"] = pd.to_datetime(df[f"{self.config.timestamp}"]).dt.tz_localize(None)
        df.index = pd.DatetimeIndex(df[f"{self.config.timestamp}"])
        df = df.drop_duplicates(subset=[f"{self.config.timestamp}"])
        
        series = TimeSeries.from_dataframe(df=df, time_col=self.config.timestamp, fill_missing_dates=True, freq=self.config.sample_frequency)

        if str(self.config.target_series) not in list(df.columns):
            raise Exception(f"Column of target not in dataframe columns")
        for m in range(2, 150):
            is_seasonal, mseas = check_seasonality(series[self.config.target_series], m=m, alpha=0.05, max_lag=150)
            if is_seasonal:
                break
        if is_seasonal:
            print(f"\n --- Found seasonality of order: {mseas} for group {group_name} --- \n")
        else:
            print(f"\n --- No seasonality found for group {group_name} --- \n")
    
        # Compute train and validation time serie sets
        print(f"\n --- Define training/validation sets --- \n")
        transformer = Scaler()
        trainining, validation = series.split_before(self.config.train_val_split)
        trainining = transformer.fit_transform(trainining)
        validation = transformer.transform(validation)
        self.status[f"{group_name}"] = {
            "training_serie": trainining,
            "validation_serie": validation,
            "seasonality": mseas if is_seasonal else None
        }

        if self.config.verbose_analysis:
            
            # BASE PLOT
            plt.figure(figsize=(15,5))
            sns.lineplot(data=df, x=f"{self.config.timestamp}", y=f"{self.config.target_series}")
            plt.xticks([])
            plt.savefig(os.path.join(self.plots_path_root, f"base_timeSerie_group{group_name}.png"))
            plt.close()

            # PLOT DEMAND GROUPED PER MONTH
            plt.figure(figsize=(15,5))
            sns.boxplot(
                x='month', y=f"{self.config.target_series}",
                data=df.assign(month=df.index.month_name())
            ).set_title(f"{self.config.target_series} montly time serie")
            plt.savefig(os.path.join(self.plots_path_root, f"montly_timeSerie_group{group_name}.png"))
            plt.close()

            # PLOT DEMAND GROUPED PER WEEK
            plt.figure(figsize=(15,5))
            sns.boxplot(
                x='week', y=f"{self.config.target_series}",
                data=df.assign(week=df.index.isocalendar().week)
            ).set_title(f"{self.config.target_series} weekly time serie")
            plt.savefig(os.path.join(self.plots_path_root, f"weekly_timeSerie_group{group_name}.png"))
            plt.close()

            # PLOT DEMAND GROUPED PER WEEK DAY
            plt.figure(figsize=(15,5))
            sns.boxplot(
                x='weekday', y=f"{self.config.target_series}",
                data=df.assign(weekday=df.index.day_name())
            ).set_title(f"{self.config.target_series} week daily time serie")
            plt.savefig(os.path.join(self.plots_path_root, f"weekday_timeSerie_group{group_name}.png"))
            plt.close()

            # PLOT DEMAND GROUPED PER HOUR IN THE DAY
            plt.figure(figsize=(15,5))
            sns.boxplot(
                x='hour', y=f"{self.config.target_series}",
                data=df.assign(hour=df.index.hour)
            ).set_title(f"{self.config.target_series} hour time window")
            plt.savefig(os.path.join(self.plots_path_root, f"hour_demand_{self.config.dataset}_group{group_name}.png"))
            plt.close()

    def occupancy_from_trips_v2(self, **kwargs):
        STARTING_CAPACITY = 0.5
        print(f"\n --- START INFER OCCUPANCY FROM TRIPS DATA - Version2 --- \n")
        freq = self.config.sample_frequency
        trips_cache = self.serveo_data.trips_cache
        if not os.path.exists(trips_cache):
            raise Exception(f"Trips cache data file does not exists")
        else:
            self.trips = pd.read_csv(filepath_or_buffer=trips_cache)
        
        # df = self.trips[["station_id", "Time", "is_end"]]
        self.trips["Time"] = pd.to_datetime(self.trips["Time"])
        # df['time_rounded'] = df['Time'].dt.round(freq)
        # df = df[["station_id", "time_rounded", "is_end"]]
        min_hour_per_station = self.trips.groupby(['station_id', 'Date'])['Hour'].min().reset_index()

        # min_hour_per_station = self.trips.groupby(['station_id', 'Date'])['Hour'].idxmin().reset_index()
        # self.trips.loc[min_hour_idx, 'bikes_available'] = (STARTING_CAPACITY * self.trips.loc[min_hour_idx, 'capacity']).round(0).astype(int)
        for index, row in min_hour_per_station.iterrows():
            station_id = row['station_id']
            date = row['Date']
            min_hour = row['Hour']
            
            
            # Find the row matching the 'station_id' and minimum 'Hour'
            mask = (self.trips['station_id'] == station_id) & (self.trips['Date'] == date) & (self.trips['Hour'] == min_hour)

            # Update the 'bikes_available' value for the matched row
            self.trips.loc[mask, 'bikes_available'] = (STARTING_CAPACITY * self.trips.loc[mask, 'capacity']).round(0).astype(int)  # Set to X% capacity defined above

        self.trips['delta_bikes'] = self.trips['Count'] * self.trips['is_end']
        self.trips = self.trips.sort_values(by=['station_id', 'Time', 'Hour']).reset_index(drop=True)

        self.trips['station_date'] = self.trips['station_id'].astype(str) + '_' + self.trips['Date'].astype(str) #+ '_' + test_march_hourly_startend_trips['Hour'].astype(str)
        self.trips['cumulative_delta_bikes'] = self.trips.groupby('station_date')['delta_bikes'].cumsum()
        self.trips['bikes_available'] = self.trips['bikes_available'].fillna(method='ffill')
        self.trips['bikes_available'] = self.trips['bikes_available'] + self.trips['cumulative_delta_bikes']

        all_hours_df = self.trips.set_index(['station_id', 'Time','is_end'])
        min_time = self.trips['Time'].min()
        max_time = self.trips['Time'].max()
        complete_time_range = pd.date_range(start=min_time, end=max_time, freq='H')

        station_ids = self.trips['station_id'].unique()
        is_end_vals = [-1,1]

        # Create a multi-index using the Cartesian product of station_ids and all hours
        index = pd.MultiIndex.from_product([station_ids, complete_time_range,is_end_vals], names=['station_id', 'Time', 'is_end'])

        # Reindex the DataFrame to include all the 24 hours for each station_id
        all_hours_df = self.trips.set_index(['station_id', 'Time', 'is_end'])
        all_hours_df = all_hours_df.reindex(index).reset_index()

        fill_zero_cols = ['delta_bikes', 'Count', 'is_end']
        ffill_cols = ['lat', 'lon',	'capacity']

        for col in fill_zero_cols:
            all_hours_df[col] = all_hours_df[col].fillna(0)

        for col in ffill_cols:
            all_hours_df[col]= all_hours_df[col].fillna(method='ffill')

        # # all_hours_df['bikes_available']= all_hours_df['bikes_available'].fillna(method='bfill')

        all_hours_df['Hour'] = all_hours_df['Time'].dt.hour
        all_hours_df['Month'] = all_hours_df['Time'].dt.month
        all_hours_df['Date'] = all_hours_df['Time'].dt.date

        # # Drop duplicate rows
        all_hours_df.drop_duplicates(subset=['station_id','Time','is_end'], keep='first', inplace=True)

        # all_hours_with_availability = all_hours_df.sort_values(by=['station_id', 'Time', 'Hour']).reset_index(drop=True)
        all_hours_with_delta = all_hours_df.groupby(['station_id', 'Time','Hour','Date']).agg({'delta_bikes': 'sum','bikes_available':'max'}).reset_index()

        all_hours_with_delta['station_date'] = all_hours_with_delta['station_id'].astype(str) + '_' + all_hours_with_delta['Date'].astype(str) + '_' + all_hours_with_delta['Hour'].astype(str)
        # cumulative sum of delta_bikes within each group
        all_hours_with_delta['cumulative_delta_bikes'] = all_hours_with_delta.groupby('station_date')['delta_bikes'].cumsum()
        all_hours_with_delta['bikes_available'] = all_hours_with_delta['bikes_available'].ffill()
        # Calculate bikes_available
        all_hours_with_delta.loc[all_hours_with_delta.index != 0,'bikes_available'] = all_hours_with_delta.loc[all_hours_with_delta.index != 0, 'bikes_available'] + all_hours_with_delta.loc[all_hours_with_delta.index != 0, 'cumulative_delta_bikes']
        all_hours_with_delta['bikes_available'] = all_hours_with_delta['bikes_available'].fillna(method='bfill')

        df = all_hours_with_delta

        self.total_data_trips = df

        i = 0
        grouped_df = df.groupby('station_id')
        for station_id, station_data in grouped_df:
            p = self.stations_occupancy_df_cache.replace("&", str(station_id))
            station_data.to_csv(p)
            if i == 350:
                break
            i += 1

    def occupancy_from_trips(self, **kwargs):
        """
            Occupancy of each station over the time
            init_data must be called before
        """
        STARTING_CAPACITY = 0.5
        print(f"\n --- START INFER OCCUPANCY FROM TRIPS DATA --- \n")
        freq = self.config.sample_frequency
        trips_cache = self.serveo_data.trips_cache
        if not os.path.exists(trips_cache):
            raise Exception(f"Trips cache data file does not exists")
        else:
            self.trips = pd.read_csv(filepath_or_buffer=trips_cache)
        
        df = self.trips[["station_id", "Time", "is_end"]]
        df["Time"] = pd.to_datetime(df["Time"])
        df['time_rounded'] = df['Time'].dt.round(freq)
        df = df[["station_id", "time_rounded", "is_end"]]

        df = df.groupby(['station_id', 'time_rounded'])['is_end'].sum().reset_index()
        df['bike_change'] = -1 * df['is_end']
        df['bike_count'] = df.groupby(['station_id', 'time_rounded'])['bike_change'].cumsum()
        
        df = df[["station_id", "time_rounded", "bike_count"]]

        stations = self.serveo_data.df_stations
        stations['station_id'] = stations['station_id'].astype(int)
        
        # df_with_capacity = pd.merge(df, stations, on="station_id", how="left")
        df_with_capacity = pd.merge(df, stations[["station_id", "capacity"]], on="station_id", how="left")
        
        i = 0
        grouped_df = df_with_capacity.groupby('station_id')
        for station_id, station_data in grouped_df:
            
            i += 1
            start_time = station_data['time_rounded'].min()
            end_time = station_data['time_rounded'].max()
            complete_index = pd.MultiIndex.from_product(
                [station_data['station_id'].unique(),
                pd.date_range(start=start_time, end=end_time, freq=freq)],
                names=['station_id', 'time_rounded']
            )
            station_data_reindexed = station_data.set_index('time_rounded').reindex(complete_index)
            
            station_data = station_data.set_index(['station_id'])
            
            merged = station_data_reindexed.merge(
                right=station_data,
                how="left",
                on="time_rounded"
            )
            merged["station_id"] = merged.index.get_level_values(0)
            merged = merged.reset_index()
            merged = merged[["station_id", "time_rounded", "bike_count_y", "capacity"]].rename({"bike_count_y": "bike_count"}, inplace=True)
            merged = merged.ffill()

            merged["bikes_available"] = np.nan
            merged.loc[0, 'bikes_available'] = (STARTING_CAPACITY * merged.loc[0, 'capacity']).round(0).astype(int)  # Set to X% 
            
    def plot_statistics(self, **kwargs):

        if self.config.analyze_station < 0:
            raise Exception(f"Set 'analyze_station' for plot statistics")

        path = self.stations_occupancy_df_cache.replace("&", str(self.config.analyze_station))
        if not os.path.exists(path):
            raise Exception(f"Cannot find file of trips for station: {self.config.analyze_station}")
        else:
            self.df_occupancy = pd.read_csv(filepath_or_buffer=path)
        
        # Normalize bike counts
        self.df_occupancy['norm_bike_count'] = (self.df_occupancy['bikes_available'] - self.df_occupancy['bikes_available'].min()) / (self.df_occupancy['bikes_available'].max() - self.df_occupancy['bikes_available'].min())
        self.df_occupancy['Time'] = pd.to_datetime(self.df_occupancy['Time']).dt.tz_localize(None)
        # plt.figure(figsize=(30, 20))
        fig, ax = plt.subplots(nrows=1, figsize=(30, 10))
        self.df_occupancy['x_axis'] = self.df_occupancy['Time'].dt.floor('6h')

        self.df_occupancy.plot(x="x_axis", y="norm_bike_count", ax=ax)
        
        plt.xlabel(f'Timestamp - sample every {self.config.sample_frequency}')
        plt.ylabel('Bikes available')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path_root, f"bike_counts_stationID={self.config.analyze_station}.png"))

        plt.close()

class PaleDataset():
    """
        It uses .csv files only
    """

    def __init__(self, config: dict, **kwarsgs):
        self.config = config 

        self.dataset_path = "data/pale"
        self.plots_path_root = f"./plots/pale"
        self.station_data_path = "stations_data.csv"
        self.stations_energy_supply_df_cache = f"./cache/pale_stations_df/cache_stationEnergySupply_stationID=&.csv"

    def __init_dataset__(self, **kwargs):
        # Load every .csv as pale station. The name must be in the filename
        self.stations = {}
        for filename in os.listdir(self.dataset_path):
            if filename.endswith(".csv"):
                if "MeterData" not in filename:
                    continue
                station_data = pd.read_csv(os.path.join(self.dataset_path, filename))
                station_name = filename.replace(".csv", "").split("_")[-1]
                station_data['station_name'] = station_name
                self.stations[station_name] = station_data
        
        self.df_energy_supply = pd.concat(self.stations.values(), ignore_index=True)

        # Delete first 2 columns and the last one
        self.df_energy_supply = self.df_energy_supply.iloc[:, 2:]
        self.df_energy_supply = self.df_energy_supply.iloc[:, :-1]
        self.df_energy_supply = self.df_energy_supply.drop('-A [MWH]', axis=1)
        self.df_energy_supply['station_id'] = self.df_energy_supply['station_name'].astype('category').cat.codes
        
        self.df_station_names = self.df_energy_supply['station_name'].unique()
        # self.df_station_names = [t.strip().replace(' ', '').lower() for t in self.df_station_names] 
        self.df_station_ids = self.df_energy_supply['station_id'].unique()

        # self.station_data = pd.read_csv(os.path.join(self.dataset_path, self.station_data_path), delimiter=";")
        # self.station_data['Windfarm'] = self.station_data['Windfarm'].str.strip().str.replace(" ", "").str.lower()
        # self.station_data.drop_duplicates(subset=['Windfarm'], inplace=True)
        # self.station_data = self.station_data[self.station_data["Windfarm"].str.replace(" ", "").str.lower().isin(self.df_station_names)]
        # print(self.df_station_names)
        # print(len(self.df_station_names))
        # print(self.station_data)
        # exit()
        # for row in self.station_data.iterrows():
        #     name = row[1]["Windfarm"].replace(" ", "").lower()
        #     if name in self.df_station_names:
        #         self.station_data.loc[row[0], "station_id"] = self.df_station_ids[self.df_station_names.index(name)]
        