# tephra-fallout-isopachs
  Most tephra fallout isopachs are hand-drawn so far, but they suffers from unquatifiable subjectivity. 
  In this project I aim to implement as many different numerical methods that solve the isopachs of tephra fallout with easy user interface and input format. Eventually users can choose any numerical method to find an isopach solution at their convenience. 
  
  The first method implemented here is cubic bivariate spline, following from [Engwell et. al. (2015)](https://link.springer.com/article/10.1007/s00445-015-0942-y) and has succeeded. The image below is a reproduction (not exact) of Mazama tephra isopach in [Buckland et. al. (2020)](https://link.springer.com/article/10.1007/s00445-020-1362-1), who used the same numerical method. You can find more isopachs solved withi this method in directory [examples/Isopachs](./examples/Isopachs/) and the corresponding data files in [examples/Data](./examples/Data/). 
![alt text](https://github.com/siyulai21/tephra-fallout-isopachs/blob/main/examples/Isopachs/Mazama_tephra.png)
  The caveat of cubic bivariate spline is that the fit suffers extrapolation instability when a region of no data is large relative to knot spacing, in which case "pseudo data" must be added to ensure fitted surface remains realistic. Where to place and what value (thickness) to choose for the pseudo data is very much subjective and should be clearly labled for reproducibility. My program provides the option to show pseudo data.
## Input data format
  Store your data file within directory [examples/Data](./examples/Data/) in "comma-separated values" (.csv) format, these are my program's assumed path and currently the only understood data format. The csv file must have 6 columns of the names ```Lon,Lat,Thickness,Pseudo,Vent Lon,Vent Lat```. All entries of all columns must be float convertible. The longitudes and latitudes data in the columns ```Lon``` and ```Lat``` must be in decimal format. ```Pseudo``` doesn't need to contain data, but if there are pseudo data to control extrapolation instability, you should clearly write ```Yes``` in the column at the rows which pseudo data are stored. ```Vent Lon,Vent Lat``` must contain 1 row of vent's (source volcano's) longtitude and latitude in decimal format. The data table below is what a row of real data followed by a raw of pseudo data looks like. 
  | Lon     | Lat   | Thickness | Pseudo | Vent Lon | Vent Lat|
  |---------|-------|-----------|--------|----------|---------|
  | -122.37 | 43.05 | 850       |        | -122.1   | 42.95   |
  | -124.19 | 42.03 | 1         | Yes    |          |         |
## Program instructions
