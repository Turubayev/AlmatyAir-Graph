"""Issued weather forecasts, indexed by issue/valid time and station.

CSV columns: issue_time, issued_at, valid_time, station_id and WEATHER variables.
All timestamps must use the same timezone as WideData.times. Only a complete
forecast published by the requested issue time is admissible.
"""
import numpy as np
import pandas as pd

WEATHER = ('temp', 'rh', 'wind_speed', 'wind_dir', 'blh')

class ForecastArchive:
    def __init__(self, frame):
        d = frame.copy()
        required = {'issue_time','issued_at','valid_time','station_id',*WEATHER}
        if not required <= set(d.columns): raise ValueError('Incomplete weather archive schema')
        for c in ('issue_time','issued_at','valid_time'): d[c] = pd.to_datetime(d[c])
        if d[['issue_time','issued_at','valid_time','station_id']].isna().any().any(): raise ValueError('Missing forecast identity')
        if (d.issued_at > d.issue_time).any() or (d.valid_time <= d.issue_time).any():
            raise ValueError('Forecast must be published before issue time and predict future hours')
        if d.duplicated(['issue_time','valid_time','station_id']).any(): raise ValueError('Duplicate forecast key')
        if not np.isfinite(d[list(WEATHER)].to_numpy(float)).all(): raise ValueError('Nonfinite weather forecast')
        if (d[['wind_speed','blh']] < 0).any().any() or not d.rh.between(0,100).all(): raise ValueError('Invalid weather range')
        self.frame = d.set_index(['issue_time','valid_time','station_id']).sort_index()

    def window(self, issue_times, station_ids, lead_hours):
        leads = np.arange(1, lead_hours+1)
        keys = [(issue,issue+pd.Timedelta(hours=int(h)),sid) for issue in issue_times for h in leads for sid in station_ids]
        values = self.frame.reindex(pd.MultiIndex.from_tuples(keys, names=self.frame.index.names))[list(WEATHER)]
        if values.isna().any().any(): raise ValueError('Weather archive does not cover requested issues/stations/leads')
        cube = values.to_numpy().reshape(len(issue_times),lead_hours,len(station_ids),len(WEATHER))
        return {v:cube[...,j] for j,v in enumerate(WEATHER)}
