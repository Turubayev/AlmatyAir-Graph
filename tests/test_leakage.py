import copy
import numpy as np
import pandas as pd
import pytest
from almatyair.features import WideData, make_sequences, make_tabular
from almatyair.forecast_archive import ForecastArchive, WEATHER
from almatyair.pipeline import Split, split_indices
from almatyair.synthetic import simulate

def data():
    d=simulate('2024-01-01','2024-01-12',seed=1);d['pm25']=d['pm25_raw']
    return WideData.from_long(d)

def archive(d,i):
    issue=d.times[i]
    return pd.DataFrame([dict(issue_time=issue,issued_at=issue-pd.Timedelta(hours=1),valid_time=issue+pd.Timedelta(hours=h),station_id=sid,
                             temp=9.,rh=60.,wind_speed=3.,wind_dir=90.,blh=1000.) for h in range(1,49) for sid in d.station_ids])

def test_future_pm_cannot_change_history():
    d=data();d.pm[30:34,0]=np.nan
    before=make_sequences(d,np.array([60]),noise=False)[0]
    d.pm[61:]=100000
    np.testing.assert_equal(before,make_sequences(d,np.array([60]),noise=False)[0])

def test_target_windows_purged():
    d=data();s=Split('2024-01-05','2024-01-09');tr,va,te=split_indices(d,s,step=1)
    assert (d.times[tr]+pd.Timedelta(hours=48)<pd.Timestamp(s.train_end)).all()
    assert (d.times[va]+pd.Timedelta(hours=48)<pd.Timestamp(s.val_end)).all()
    assert (d.times[te]>=pd.Timestamp(s.val_end)).all()

def test_archived_forecast_ignores_future_reanalysis():
    d=data();d.forecast_archive=ForecastArchive(archive(d,60))
    before=make_tabular(d,np.array([60]),noise=False)
    for v in WEATHER:d.met[v][61:]=9999
    after=make_tabular(d,np.array([60]),noise=False)
    cols=[c for c in before if '_fc' in c or '_win' in c]
    pd.testing.assert_frame_equal(before[cols],after[cols])
    assert before['temp_fc'].iloc[0]==9
    Xh,Xf,Y,I=make_sequences(d,np.array([60]),noise=False)
    assert np.isfinite(Xf).all() and len(I)==1

def test_late_and_incomplete_archive_rejected():
    d=data();f=archive(d,60);f['issued_at']=f['issue_time']+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):ForecastArchive(f)
    incomplete=ForecastArchive(archive(d,60).iloc[1:])
    with pytest.raises(ValueError):incomplete.window(d.times[[60]],d.station_ids,48)

def test_empty_issues_fail_clearly():
    with pytest.raises(ValueError):make_tabular(data(),np.array([0]))
