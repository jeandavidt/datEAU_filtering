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