import numpy as np
import numpy.linalg as alg
import pandas as pd

from sklearn.preprocessing import StandardScaler
# debugging data import



# def multivariate(df,col_name_list, start_cal, end_cal,min_var_exp=0.95, alpha=0.95):
def build_pca(df):
    # #### scaling the features
    DF = df.copy()
    scaler = StandardScaler()
    x_raw = df[col_name_list].loc[start_cal:end_cal].values
    x_bar=scaler.fit_transform(x_raw)
    
    ##### singular value decomposition
    svd_feat={}
    svd_feat['u'],svd_feat['s'],p_T = alg.svd(x_bar,full_matrices=False)
    svd_feat['p'] =p_T.T
    svd_feat['cov']=np.dot(x_bar.T,x_bar)/(len(x_bar)-1)
    svd_feat['eigenvalues'] =  svd_feat['s']**2/(len(x_bar)-1)
    svd_feat['Lambda'] = np.diag(svd_feat['eigenvalues'])
    svd_feat['t']=np.dot(svd_feat['u'],np.diag(svd_feat['s']))
    
    
    ##### How much of the variance is explained by the principal components? 
    t_var = []
    n_components=len(col_name_list)
    
    for col in range(n_components):
        t_var.append(svd_feat['t'][:,col].var())
        
    sum_var = np.sum(t_var)
    svd_feat['exp_var'] = np.divide(t_var,sum_var)
    
    ##### Dimension reduction
    # Initializing a model object (dictionnary)
    model ={}
    #info on how the model was calibrated
    model['start_cal'] = start_cal
    model['end_cal'] = end_cal
    
    #What proportion of the variance is explained by the model? Should be over min_var_exp
    explained=0.
    for i in range(n_components):
        explained+=svd_feat['exp_var'][i]
        if (i>0 and explained>min_var_exp):
            model['n_comp'] = i+1
            model['exp_var']=explained
            break
        else:
            continue
    # Splitting the matrices 
    
    t_hat=svd_feat['t'][:,0:model['n_comp']]
    model['t_hat_stdev']=[t_hat[:,i].std() for i in range(model['n_comp'])]
    model['p_hat']=svd_feat['p'][:,0:model['n_comp']]
    x_hat=np.dot(t_hat,model['p_hat'].T)
    model['Lambda_hat']=svd_feat['Lambda'][0:3,0:3]
    
    
    ##### Calculate the limits for fault detection
    def calc_stats(norm_data, model_load, eigenvals_mat):
        n_comp = norm_data.shape[1]
        T2=[]
        Q=[]
        for i in range(len(norm_data)):
            T2.append(np.dot(norm_data[i].T, \
                             np.dot(model_load,\
                                   np.dot(alg.inv(eigenvals_mat),
                                          np.dot(model_load.T,norm_data[i])))))
            
            R = np.identity(n_comp)-np.dot(model_load, model_load.T)
            Q.append(np.dot(np.dot(norm_data[i],R),norm_data[i].T))
            
        Q=np.array(Q)
        T2=np.array(T2)
        return T2, Q
    
    model['T2'], model['Q'] = calc_stats(x_bar,model['p_hat'],model['Lambda_hat'])
    #calculating the limits
    
    limits={}
    
    def calc_limits(data, n_pc, Alpha, eigenvals_vec):
        from scipy.stats import f as fisher
        from scipy.stats import norm as normal
        
        #### T2 stat limit 
        n_data = len(data)
        F_alpha = fisher.interval(Alpha,n_pc,n_data-n_pc)[1]
        T2_lim = (n_pc*(n_data**2-1)/(n_data*(n_data-n_pc)))*F_alpha
        
        #### Q stat limit prior info
        theta=np.zeros([3,])
        for i in range(3):
            for j in range(n_pc+1):
                theta[i]+=eigenvals_vec[j]**(i+1)        
        h_0= 1-((2*theta[0]*theta[2])/(3*theta[1]**2))
        z_alpha = normal.ppf(Alpha)
        
        #### Actual Q stat limit
        Q_lim= theta[0]*\
        (np.dot(z_alpha,np.divide(np.sqrt(2*theta[1]*h_0**2),theta[0]))\
        +1+ np.divide(theta[1]*h_0*(h_0-1),theta[0]**2))**(1/h_0)
        return T2_lim, Q_lim
    
    limits['t2'], limits['Q'] = calc_limits(x_bar, model['n_comp'], alpha, svd_feat['eigenvalues'])
    limits['alpha']=alpha
    
    ##### Applying the model to the df data
    x_totraw = df[col_name_list].values
    x_tot=scaler.transform(x_totraw) 
    t_tot = np.dot(x_tot, model['p_hat'])
    for i in range(model['n_comp']):
        DF['t_'+str(i+1)]=t_tot[:,i]
        
    DF['T2'], DF ['Q'] = calc_stats(x_tot,model['p_hat'],model['Lambda_hat'])
    if 'fault_count' not in DF.columns:
        DF['fault_count'] = 0
    DF['fault_count'].loc[DF['T2']>limits['t2']] +=1
    DF['fault_count'].loc[DF['Q']>limits['Q']] +=1
    
    return DF, limits, svd_feat, model
