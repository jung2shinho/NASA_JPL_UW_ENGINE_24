*(see full report for more details)*
# UW ENGINE 2024 - NASA JPL - CAPSTONE PROJECT

## Introduction
**Industry Sponsor**: 	NASA Jet Propulsion Laboratory (JPL)

**Project Title**: 	Data Science for Discovery: Machine Learning to Evaluate the Risk and Resilience of Societal Critical Infrastructure to Space Weather

**Objectives**:
- Create a data framework for studying risk and resiliency of societal critical infrastructure from a space weather natural hazard.
- Integrate data from various domains to provide deeper insight into how solar events impact critical infrastructure. 
- Utilize machine learning to predict outcomes in various space weather cases and identify subtle underlying connections.
- Explore the characteristics of space weather impacts on global critical infrastructure; primarily, electrical power grids and communications systems.

## Abstract
A Coronal Mass Ejection (CME) is a solar phenomenon that discharges a cloud of highly magnetized plasma into space, creating significant magnetic fluctuations on Earth’s magnetosphere. This causes geomagnetically-induced currents (GIC) that damage critical, societal  infrastructure, such as electrical transformers, oil pipelines, and long-range telecommunications. Accurately predicting and understanding the frequency, characteristics, and network effects of GICs requires deliberate data consolidation and analysis under a structured framework in order to enable deeper insights into the nature of GICs.

In collaboration with NASA Jet Propulsion Laboratory (JPL) and U.S Geological Survey (USGS), a comprehensive data pipeline was constructed to aggregate and analyze solar data from three major, public datasets (i.e SuperMAG, OMNIWeb, NERC). Combining and restructuring this data into several analysis-ready schema facilitated subsequent extreme value analysis, network analysis, and machine learning models for nowcasting and forecasting magnetic perturbation predictions.

Due to primary interests in high-value GICs, a Generalized Extreme Value (GEV) distribution was utilized to model the tail end of the GIC data. Quantile-quantile plots assisted in presenting comparisons between these GEV and Gaussian distributions, delineating distinct characteristics of each ground-based GIC region.

Network analysis was conducted using cross-correlations of wavelet decomposition coefficients, developing a response network to provide insight into subsequent effects of GICs on the physical U.S power grid. Furthermore, device-specific threshold optimization allowed a more representative, time-based spatial response network that assisted in identifying regions of vulnerability in U.S electrical infrastructure. 

For magnetic perturbation and GIC predictions, three machine learning (ML) models were trained and tested, including a multi-linear regression (MLR), a random forest regression (RFR), and a long-short term memory model (LSTM). Kernel density plots with root means square metrics were used to compare and determine the models' accuracy and precision.
 
## Brief Results
Under extreme value analysis, the Generalized Extreme Value model provided a non-linear, inversely proportional trend between GIC values and predefined return periods (T). The GEV distribution also provided a superior fit for highly active GIC supernodal regions (i.e District of Columbia) when compared to a gaussian distribution, with quantile-quantile plots distinctly differentiating between supernodal and non-supernodal GIC characteristics.

Network analysis confirmed previous determinations of long range east-west connections in the response grid during heightened GIC activity (Orr 2021). Overlaying a physical U.S power grid to visualize the potential impact of GIC variations provided uninteresting conclusions, however, eigenvalue centrality measures of these nodal graphs highlighted previously neglected areas of interests, such as metropolitan areas of Seattle, WA, and Phoenix, AZ. 

In machine learning, the LSTM model had the best performance in nowcasting magnetic perturbations with >95% accuracy given a 48-hour time frame of 5-min intervals. Trained on 50 epochs, batch size was set at 360, with smaller batch sizes inducing a less stable training process while larger batch sizes led to poorer performance.

## Extreme Value Analysis
In the initial phase of our analysis, we concentrated on processing on the North American Electric Reliability Corporation (NERC) raw dataset, spanning the years 2013 to 2022. The GIC (Geomagnetic Induced Current) values consisted of positive and negative data, thus, we converted all values into their absolute magnitudes and then extracted the maximum values from each GIC event from each GIC device across the United States. Subsequently, we visualized these values on a geographic map to access their distributions with other statistical measures. This approach facilitated our initial understanding of the spatial distribution of geomagnetic disturbances across different regions over a long timespan.

![figure 20](/static/images/figure20.png) 

*Figure* 20: Visualization of GIC on USA map

In the second phase of our analysis, we focused on estimating the hazard return levels for periods 1-in-100 years using the Generalized Extreme Value (GEV) method. This approach aimed to predict the worst-case scenarios for geomagnetically induced current (GIC) events based on our dataset. To implement this, we fit our extreme GIC data into the GEV model and specified a return period (T), which the model uses to generate the corresponding return values. A critical aspect of this process involved setting an appropriate threshold for exceedance. We experimented with various thresholds and conducted additional estimation for 1-10 year return levels to validate the trend accuracy and refine our GEV model. In our examination, we acquired a return value of 700.2 amp when setting the thres=15 and T=100, while the max extreme value in our current dataset is 483 amp.

### Data and Model Setup
**Dataset**: Extreme GIC measurements across the US, spanning from 2013 to 2022.
**Extreme Value of Interest**: 483 amps
- The highest GIC measured in the existing dataset.
**Threshold Setting for GEV**: 15 amps
- Data above this value are considered for the GEV fitting.

**Return Periods**: We focus primarily on 100-year return level but also examine shorter periods(1-10 years) for comparative insights.

### GEV Model Fitting
- Fit the GEV Model: Using GIC data (2013 to 2022) that exceeds the thresholds.
**Calculated Return Levels**:

- For T=100 yrs, threshold =15 amps: The model predicted a return level of 700.2A
- For T=10 yrs, threshold =42 amps: The model predicted a return level of 393.2A
*NOTE*: See *Figure*s 21 and 22 to see the trend lines.

### Threshold Sensitivity
Lowering the threshold might include more data points, potentially smoothing out the return level curve but might also dilute the focus on truly extreme events. Conversely, a higher threshold could lead to data scarcity, increasing model uncertainty. 

![figure 21](/static/images/figure21.png)

*Figure* 21: Return level plot for 3 kinds of data filter                       

![figure 22](/static/images/figure22.png)

*Figure* 22: Threshold exceedance of 1-10 years GEV return level

![figure 23](/static/images/figure23.png)

*Figure* 23: ZT: T-year return level equation;
*This is the event value that happens once in every T years*

### Supernodes
After identifying certain regions of the U.S (i.e Washington D.C) that have more activity than others, we focused on how these GICs differentiated based on regions and what their GIC characteristics were. At the start of our research, 18 significant events were identified between 2013 to 2022. We utilized the pyextremes package - a Python library aimed at performing univariate Extreme Value Analysis (EVA) - to examine these events by comparing GEV and Gaussian distributions of GICs to identify supernodal vs non-supernodal characteristics.

### Quantile-Quantile Plots
The quantile-quantile (QQ) plot, a model-independent method for comparing two empirically sampled statistical distributions (Braun & Murdoch, 2016), was employed to investigate changes in the distribution over its full range. This method does not require any prior assumptions of the functional form of the underlying distribution. The QQ plot can be used to test how the functional form changes over time, or whether the samples are drawn from distributions with the same model but with time-dependent moments. Moreover, if the underlying distribution is multicomponent, the QQ plot will reveal the (potentially time-dependent) transitions between regions of parameter space that contain the distinct components or populations of observations​​.

### Supernode & Non-supernode Analysis
We conducted an analysis of the QQ plot for the supernodes, defined as the nodes within a 50-mile radius around Washington, D.C. (latitude 38.8951, longitude -77.0364) and Michigan (latitude 44.3148, longitude -85.6024). The analysis included comparisons between the data and the normal distribution as well as the Generalized Extreme Value (GEV) distribution for both supernodes and non-supernode regions.

![figure 24](/static/images/figure24.png)

*Figure* 24: Supernode 1 (Device 10305, Event 2017 05 27)

For Supernode 1 (Device 10305, Event 20170527) in *Figure* 1, we discovered significant deviation from the normal distribution, particularly in the upper tail. This deviation suggests the presence of extreme values that the normal distribution fails to capture. In contrast, the GEV distribution showed better alignment in the upper tail, indicating that it models extreme values more effectively, which is typical for data with heavy tails.

![figure 25](/static/images/figure25.png)

*Figure* 25: Supernode 2 (Device 10397, Event 2017 05 27)

Similarly, in Supernode 2 (Device 10397, Event 20170527) in *Figure* 2, we observed significant deviations from the normal distribution. Consistent with the findings for the first supernode, the second supernode's data also exhibited a better fit with the GEV distribution, suggesting consistency in data behavior across different supernodes.

![figure 26](/static/images/figure26.png)

*Figure* 26: Non-supernode (Device 10081, Event 2017 05 27)

For the non-supernode (Device 10081, Event 20170527) in *Figure* 3, the QQ plot revealed that the data align more closely with the normal distribution than with the GEV distribution. This pattern indicates fewer occurrences of extreme deviations and a more stable data pattern, typical of less critical infrastructure or differing operational influences.

![figure 27](/static/images/figure27.png)

*Figure* 27: Comparison of non-supernodes

### Results
When comparing QQ plots for Supernode 1 across multiple events (20150622, 20170527, 20180825, 20210512, 20220409) in *Figure* 4, we noted that the supernode data exhibited increasing deviations from the normal model over five years, particularly in the later years. This trend suggests a progression towards greater variability or an increase in operational anomalies as the system ages. 

Conversely, the GEV model consistently provided a better fit, especially in the upper quantiles, across all years, indicating that the GEV distribution is more effective in capturing the extreme values within the dataset.

In most cases, more than 90% of the data fall into a single component, the “core” of the distribution. The remaining 10% of data form a distinct “tail” component, which itself splits into two parts in some cases. Generally, the core component in the distribution shows little sensitivity to the detailed differences across these years. However, the two-part tail dominates the plot on the linear scale due to the heavy-tailed nature of the distribution.

The Generalized Extreme Value (GEV) distribution, designed to model the tail behaviors of datasets, showed that when focusing on the largest sample quantile and observing its proximity and fit to the GEV line in the QQ plot, the data point for the largest sample quantile is closer to the theoretical line provided by the GEV distribution compared to the normal distribution. However, at values outside the 4-amps region, the QQ traces deviated from each other, highlighting the differences between the distributions.

The approximate linearity of each segment indicated that the functional form of the distribution remained similar throughout the years, exhibiting a consistent departure pattern.

## Network Analysis
Network Analysis contains data preprocessing, wavelet decomposition, and cross-correlation. Initially separated by events, measured GIC data from the North American Electric Reliability Corporation (NERC) was firstly preprocessed to prevent errors derived from data gaps and NaN values. Then, wavelet decomposition was applied to the resampled data, identifying strong changes in measured GIC values to infer fluctuations of GICs in the power grid. Lastly, cross-correlations between each Device ID were created using a global and sliding window analysis amongst different thresholds. Centrality measures are calculated to display the interconnectedness of the nodal graphs while overlaying this response network with the physical power grid over time to provide insight into the actual GIC fluctuation within the power grid.

### Software Design

![figure 28a](/static/images/figure28a.png)
![figure 28b](/static/images/figure28b.png)
![figure 28c](/static/images/figure28c.png)

*Figure* 28 a/b/c: Data Flow for Network Analysis

### Data Preprocessing
At the beginning of network analysis, data preprocessing was performed to ensure data quality and consistency. Geomagnetic-induced current (GIC) data was downloaded from the NERC. Firstly, because the sampling rate of the original data is different, to unify the analysis conditions, the original GIC data observed from different monitors of one event were resampled for every one minute to make the data consistent in time. The missing data points were filled by using a three-order spline interpolation method which is a mathematical method used to create a smooth curve through a discrete set of data points.

### Wavelet Transform
The next step is to perform a wavelet transform on the GIC data. Due to the nonlinear and non-stationary characteristics of GIC data, wavelet transform was chosen as the main analysis tool. Wavelet transform is different from Fourier transform in that it can provide localized information in the time-frequency domain. This means that the wavelet transform can simultaneously give the frequency characteristics of the signal at a certain point in time, which is particularly important for the analysis of geomagnetic disturbance which changes rapidly in time. In our project, the Haar wavelet was performed on the discrete wavelet transform for the 4-level decomposition, which extracted detail coefficients and approximate coefficients respectively. The first-level coefficients (approximate coefficients) of decomposition captured the major trend of the signal and the second-level coefficients (detail coefficients) captured some rapid changes representing noise or sudden fluctuations. Then the results of wavelet transform were displayed in the form of heat maps, with different colors representing the different values of coefficients, making the results intuitive and easy to understand.

### Correlation
To gain insight into the interactions and effects between the different monitors, we calculated two types of correlations. Firstly, the maximum overlapping discrete wavelet transform was utilized to extract the wavelet coefficients. Then the correlations were calculated.
Global correlation analysis: Correlation coefficients between every two monitors over one event were calculated to identify overall interdependencies.
Sliding-window cross-correlation: Sliding window cross-correlation of time series by setting different time Windows were calculated to capture dynamic changes more finely

### Threshold optimization
In previous data processing, we have been using the global threshold to determine the connection of two monitors, which was simple and efficient to use. But this approach works better when all monitors are the same and evenly distributed. If this method is applied to our program, it will lead to severely distorted results: those sites with high coefficients are always connected to the network, while those with lower coefficients are rarely connected. Therefore, using a long time series to determine individual station thresholds will account for our local conditions. Firstly, the calculated value of sliding cross-correlations between each pair of stations will be applied to generate an intermediate adjacency matrix Aij*. By setting the global threshold, the connection above the threshold is 1, and the connection below the threshold is 0. The equation for Aij* is:

Aij*(CT , t)=(|Cij(t)-CT|)

where Cij(t) is the correlation between monitor i and monitor j at time t, and CT is the global threshold. Then the average degree ni(CT) at different global threshold CT values for each site need to be calculated:

ni(CT)=(tTjN(t)Aij*(CT , t)N(t)-1)/T

The purpose of this algorithm is to find a specific threshold CTi for each monitor so that the calculated average degree of each monitor is equal to the normalization degree n0. This requires iteratively adjusting CTi for each monitor until the target average degree is reached. The next step is to determine two thresholds, CTi and CTj . If the lower threshold of the two sites is satisfied CTij = min[CTi, CTj], it is considered that there is a connection between sites i and j. Finally, the specific threshold of each pair of monitors will be utilized to analyze the network.

### Results
After applying the Haar wavelet transform to the raw GIC data of even 2013_1002, two heat maps were obtained. *Figure* 29a  shows the major trend of GIC data observed in 2013 Oct 2nd, where the color shows the magnitude of the first-level wavelet coefficients and latitudes represent different monitors. *Figure* 29b shows the rapid changes of GIC data observed in 2013 Oct 2nd, where the color shows the magnitude of the second-level wavelet coefficients, and the redder the color the stronger the change.

![figure 29a](/static/images/figure29a.png)
![figure 29b](/static/images/figure29b.png)

*Figure* 29 a/b: Wavelet heatmaps of first and second level coefficients for each device.

![figure 30](/static/images/figure30.png)

*Figure* 30: One-time cross correlation between coefficients.

*Figure* 30 is to obtain a one-time cross-correlation on different scales of the wavelet transform. It shows a network diagram where the nodes represent different monitors and lines between nodes show correlations. The more nodes are wired, the higher the correlation with the other monitors.

![figure 31](/static/images/figure31.png)

*Figure* 31: Time-based heatmap showing the relationship between GIC devices

*Figure* 31 is a time-based heatmap showing the relationship between GIC devices. The size of the circles indicates the local degree of each node (number of connections to that node in the network at that time). Only cross-correlation greater than 0.8 will increase the node degree at this time point. Then the color plots the magnitude of the correlation, and it will be presented only when the correlation of the node must both (i) exceed 80% of all values obtained at that node and (ii) the wavelet cross-correlation between a pair of nodes must exceed 0.85 within a 30-minute leading edge window, otherwise it will show a circle with no color.

![figure 32](/static/images/figure32.png)

*Figure* 32: Inverse trend between between degree of iteration and device-specific correlation thresholds for 5 different GIC devices or stations

*Figure* 32 shows the relationship between degree and threshold after applying the degree algorithm. Then the n0 was set to equal 0.1, we could find different thresholds for each monitor. Finally, we could use it to find a new correlation between each monitor and make a clearer comparison with the physical power grid.

## Machine Learning
Our objective is to model and predict GIC events to mitigate the impact on the critical infrastructure. Thus, developing nowcasting and predictive models for magnetic perturbation prediction (dB/dt) and GICs required the most meaningful ML algorithms. Our approach was to divide efforts into two different directions for prediction: one is predicting magnetic perturbation (dB/dt) and other is predicting GIC values directly. The reason why we had 2 directions is that “since GIC data is proprietary, the time variability of the horizontal component of the magnetic field perturbation (dB/dt) is used as a proxy for GICs” (Upendran, V. et al., 2022).

### Preprocessing
The preprocessing dataset involved three raw datasets SuperMAG (2010 - 2023), OMNI (2010 - 2023) and NERC GIC (2013 - 2023). As we have predictive direction, we prepared two different type of dataset: one is for magnetic perturbation prediction (joined between SuperMAG and OMNI dataset) and other is for GIC values prediction (joined between SuperMAG, OMNI and NERC GIC dataset). 

### Dataset for Magnetic Perturbation Prediction (dB/dt)
The dataset was created by joining the datetime between SuperMAG and OMNI dataset, both ranging from 2010 - 2023. The training dataset ranged from 2010 - 2019 and the testing dataset ranged from 2020 - 2023 (Upendran, V. et al., 2022). The diagram below shows the details of the data flow from collecting raw data to the complete data set files stored in Google Drive.

![figure 33](/static/images/figure33.png)

*Figure* 33: Diagram showing the process of creating ready for analysis

### SuperMAG Dataset (dB/dt)
SuperMAG is a global network of around 300 magnetometer ground stations employed in measurement of geomagnetic perturbations. The available dataset that we use in this study provides the geomagnetic perturbations measurements of 1-min cadence from around 300 stations around the globe. From the SuperMAG data, we are using the following measurements: datetime, glon, glat, mlon, mlat, mlt, dbn_nez and dbe_nez. But since the provided data are in 1-min cadences and for the purpose of stability and synchronization with OMNI dataset, we resampled the provided into 5-min cadences. Our preprocessing data also includes filling missing values, remove outliers and replacing the default values in the provided data

### OMNI Dataset
OMNI dataset provided near-Earth solar wind magnetic field and plasma parameter data from several spacecraft in geocentric or L1 (Lagrange point) orbits. From OMNI data, we retrieved 5-min cadence data since we believe that 5-min cadence data can provide more stability. And indeed, the fields that we utilized from OMNI data  are: datetime, bx_gse, by_gsm, bz_gsm, flow_speed, proton_density, T, pressure and clock angle of the IMF. 

### Datasets NERC GIC Prediction
The dataset was created by joining the GIC event datetime and by all the devices sensors of GIC measured within the 500 miles radius of the station where Magnetic perturbation measured showed in figure , from the NERC GIC and Supermag datasets, both ranging from 2013 - 2023. The training dataset is ranged from 2013 - 2022 and the testing dataset 2023. 

![figure 34](/static/images/figure34.png)

*Figure* 34: Stations and Devices sensor relationship where Station is 
marked Red and Station is blue

### NERC Geomagnetically Induced Current (GIC)
The North American Electric Reliability Corporation (NERC) has a program to collect data on geomagnetic disturbances (GMDs). These disturbances occur when the sun ejects charged particles that interact with the Earth's magnetic field and atmosphere. When this interaction occurs, it can cause changes in the Earth's magnetic field, which might disrupt or damage important infrastructure, like power systems. In rare but strong GMD events, these changes can create strong direct currents in the electric power grid. These currents can affect the stability of the system, interfere with protective devices, and potentially damage large power transformers. This is why NERC is collecting data to better understand and manage the risks associated with GMDs.

### SuperMAG Dataset (dB/dt)
As described above, we want to understand the geomagnetic perturbations that have an effect on the GIC sensor measurements. Similarly to our other prediction on the magnetic perturbation, we are using the following measurements: datetime, glon, glat, mlon, mlat, mlt, dbn_nez and dbe_nez from the SuperMag datasets to predict the Nerc GIC.

## Training & Testing Models (Results)
Since we have 2 predictions: Magnetic perturbations and GICs. The main 3 models we are using are Multivariate Linear Regression, Random Forest Regressor and Long Short Term Memory Models
Magnetic Perturbation Models

### Multivariate Linear Regression (MLR) Model
The Multivariate Linear Regression model is the first model that tried to understand the underlying data structure of our dataset. It also served as a baseline model for us to compare performance and figure out ways to improve performance of the models trained later on this report.

### Results and Findings for Magnetic Perturbation (db/dt)
For Multivariate Linear Regression, we trained models for the dataset timeline from 2010 - 2019 and testing in 2020 - 2023. Below will be some graphs of our results:
Table 1: RMSE and MAE value for training and testing dataset

|--------| Train dBn/dt | Train dBe/dt | Test dBn/dt | Test dBe/dt |
|--------|---------|---------|---------|---------|
| RMSE   | 7.357   | 7.932   | 7.530   | 7.702   |
| MAE    | 5.095   | 4.954   | 5.298   | 4.905   |


![plot 1](/static/images/plot1.png)

*Plot* 1: Multivariate Linear Model - Plot of truth vs prediction zoom-in two days range of test dataset.

![plot 2](/static/images/plot2.png)

*Plot* 2: Multivariate Linear Model - Residual Error Plot for dBn/dt and dBe/dt prediction
Throughout the results, we recognize that multivariate linear regression is not really doing great in the dataset, especially for the dBe/dt prediction. From the metrics table, we can see that both values for RMSE and MAE of dBe/dt in training and testing dataset are both all higher than those values of dBn/dt. This tells us that the model performs worse on the dBe/dt prediction. Look into plot 2, the line plot of truth vs prediction, in this plot we have zoom into 2 days range data of prediction and we can see that the model is not able to capture the dynamics of dBe/dt really well, meanwhile, it’s able to follow the trend that’s happening in the dBn/dt. One of the hypotheses that we have is that our multivariate linear regression is not able to capture the non-linear relationship between the features. Moreover, since we’re working with the time series data, the order of the data point and historical information are also really important. Since that, we have experimented with Ranform Forest, which is known for capturing non-linear relationships between features, and Long Short Term Memory (LSTM), which is known for keeping the order of data points and maintaining information in memory for longer periods. We have also shown the results and analysis for those models in later sections below.

## Random Forest Regressor (RFR) Model

**Hyperparameters**
- max_depth = 90​
- max_features = 0.5​
- min_samples_leaf = 8​
- min_samples_split = 5​
- n_estimators = 911​
- Loss = RMSE​

### Results and Findings for Magnetic Perturbation (db/dt)

![plot 3](/static/images/plot3.png)

*Plot* 3: Random Forest Model - Plot of truth vs prediction zoom-in two days range of test dataset.

![plot 4](/static/images/plot4.png)

*Plot* 4: Random Forest Model - Residual Error Plot for dBn/dt and dBe/dt prediction
RFRM resulted in mediocre performance in predicting magnetic perturbation values (nT) for the North and East axes.​
However, the results are ​essential in understanding the dynamics and the characteristics of the data.​ The model captures low-magnitude and non-linear trends, however, fails to learn the high-magnitude dynamics of the data.​

### Long Short Term Memory (LSTM) Model
LSTM networks are a type of recurrent neural network (RNN) particularly well-suited for time series forecasting, sequential data, and tasks where the order of data points matters. They were designed to overcome the limitations of standard RNNs. Compared to RNNs, they have a more complex architecture that allows them to maintain information in memory for longer periods.
The core components of LSTM are cell state and gates. *Figure*1 illustrates the mechanisms inside a LSTM unit, where i, f, g, and o denote the input gate, forget gate, cell candidate, and output gate, respectively. 

By default, LSTM uses sigmoid activation to produce a filter for the information. Each gate produces different function:
Forget Gate: Decides what information to discard from the cell state.
Input Gate: Determines which new information to store in the cell state.
Output Gate: Controls what information to output from the cell state to the next time step.
Cell candidate: The cell state is updated with new information scaled by the input gate and old information scaled by the forget gate.

![figure 35](/static/images/figure35.png)

*Figure* 35: Standard LSTM unit

*Figure* 36 illustrates the sequential data process inside a LSTM layer, through this unique architecture, LSTM comes out to be a powerful tool for capturing long-term dependencies and performing future prediction.

![figure 36](/static/images/figure36.png)

*Figure* 36: Standard LSTM Layer


### Results and Findings for Magnetic Perturbation(Db/dt)

![plot 5](/static/images/plot5.png)

*Plot* 5: Long-Short Term Memory Model  - Plot of truth vs prediction zoom-in two days range of test dataset.

![plot 6](/static/images/plot6.png)

*Plot* 6: Long-Short Term Memory Model - Residual Error Plot for dBn/dt and dBe/dt prediction

Our model was trained on the 2010-2019 dataset while validated on 2020 dataset. Our current configuration is: Batch_size = 360, Activation: ReLu, Optimization = Adam, Loss = MSE, Epochs = 50. Through the training process, we find out:
Batch size significantly affects the performance and efficiency of our LSTM model. A smaller batch size induces a slower and less stable training process, while a larger batch size leads to poorer generalization.
The default sigmoid activation did not work well on our dataset, instead, ReLu activation converged faster and achieved a lower training and validation loss.
A single layer can achieve satisfying training performing on a single time step while multiple time steps require more complicated layers setting.

### NERC GIC models
All the models are using a dataset of the station in Washington DC(39.0, 281.8) where the NERC GIC values are the most extreme. In this case, we can see how each of the models' performances on prediction compare to the actual values of the GIC sensor.

### Multivariate Linear Regression (MLR) Model
**Hyperparameters**
- Default Parameters from keras.

**Results and Findings:**

![plot 78](/static/images/plot78.png)

*Plot* 7 & 8: Multivariate Linear Regression Model - GIC Predictions + Residual Error Plot 

The model is able to pick up some trends of the data, which indicates we can definitely improve the models through hyperparameters.

### Random Forest Regressor (RFR) Model
**Hyperparameters:**
- n_estimators=50
- max_depth=5

**Results and Findings**

![plot 910](/static/images/plot910.png)

*Plot* 9 & 10: Random Forest Regressor Model  - GIC Predictions + Residual Error Plot 

This model seems to be flat prediction at around 2 Amperes. Similar to the Linear Regression Model, my prediction is that because we are overpopulated the normal data compare to when the GIC events extreme happened, which cause the models to predict at 2 amperes which give us the most accurate overall but result incorrect.

### Long Short Term Memory (LSTM) Model
**Layers of Neural Network**
- LSTM(128, input_shape=(1, X_train_normalized.shape[1]))
- Dense(64, activation='relu')
- Dense(32, activation='relu')
- Dense(1)

**Model evaluation:**
- optimizer='adam'
- loss='mse'

**Model training**:
- 5 epoch

**Results and Findings**

![plot 1112](/static/images/plot1112.png)

*Plot* 11 & 12: Long-Short Term Memory Model  - GIC Predictions + Residual Error Plot 

The LSTM model seems to be able to pick up the trend of the actual signal, which gives us an indication of these particular models performing better when it comes to our time series dataset prediction compared to random forest and linear regression.

