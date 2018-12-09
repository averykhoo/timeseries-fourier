# timeseries-fourier
find cyclic patterns of behavior in timeseries

## usage
-   apply to your dataframe
```python
df_fft = df[(df.dstip_private == False) &
            (df.dstport <= 1024) &
            (df.dstip_owner.isna())
           ].groupby(['srcip', 'dstip']).timestamp.apply(get_topk_periods, verbose=True)
```

-   drop extra columns
```python
    df_fft = df_fft.sort_values('magnitude', ascending=False).reset_index().drop(columns='level_2')
```
-   NOTE: drop column 'level_1' if groupby was only called on one column


## notes
-   has issues if there are multiple independent segments of periodic behavior
-   requires a recent version of pandas with `isna`, otherwise, replace with older function call `isnull`
