def plotRaw_D(Data_df, param_list, title):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))
    for parameter in param_list:
        ax.plot(Data_df.loc[:,parameter])  

    #day_month_year_Fmt = mdates.DateFormatter('%d %B %Y')
    #ax.xaxis.set_major_formatter(day_month_year_Fmt)
    plt.xticks(rotation=45)
    plt.ylim(bottom=0)
    plt.legend(param_list)
    plt.title(title)
    plt.show()

def Plot_Outliers(FullData):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))

    data = FullData(channel).values

    AD = FullData(channel).AD
    outlier = FullData(channel).Sec_Result.outlier
    ub = FullData(channel).Sec_Result.UpperLimit_outlier
    lb = FullData(channel).Sec_Result.LowerLimit_outlier


    figure
    hold on
    plot(t,AD,'k',t,lb,'b',t,ub,'g','markersize',6)
    plot(t(outlier),data(outlier,2),'rx','markersize',6)
    xlabel('Time','fontsize',10)
    xtickangle(-20)
    ylabel (FullData(channel).channel, 'fontsize',10)  
    dynamicDateTicks([],[],'dd/mm')
    plt.xticks(rotation=45)
    plt.legend('Raw Data','LowerLimit', 'UpperLimit', 'Outliers', 'location','northoutside', 'Orientation', 'horizontal')
    plt.show()
