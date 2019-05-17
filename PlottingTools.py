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

def Plot_Outliers(df,var_name):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))

    AD = df[var_name+'_Accepted']
    outlier = df.loc[df.outlier,[var_name]]
    ub = df.UpperLimit_outlier
    lb = df.LowerLimit_outlier
    raw=df[var_name]
    
    ax.plot(outlier,'rx')
    ax.plot(lb,'b')
    ax.plot(ub,'g')
    
    ax.plot(AD,'purple')
    ax.plot(raw,'k')
    
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel(var_name)  
    plt.xticks(rotation=45)
    plt.legend(['Outliers','LowerLimit', 'UpperLimit','Accepted Data','Raw'])
    plt.show()
