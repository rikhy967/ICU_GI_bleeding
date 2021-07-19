def performance_visualizer(trials_obj,n_models,choice=False,**choice_var):
    
    performance = [1-t['result']['loss'] for t in trials_obj.trials]
    
    
    hyperparam= list(trials_obj.trials[0]['misc']['vals'].keys())
    
    values_dict ={}
    
    for i in hyperparam:
        
        values_dict[i]=[]
        
        for j in trials_obj.trials:
            
            if(len(j['misc']['vals'][i])==0):
                
                values_dict[i].append(np.NaN)
                
            else:
            
                values_dict[i].append(j['misc']['vals'][i][0])
                
    out = pd.DataFrame.from_dict(values_dict)
    
    out['performance'] = performance
    
    out=out.sort_values(by=['performance'])
    
    
    if choice:
        
        for i in list(choice_var.keys()):
        
            for j,_ in enumerate(choice_var[i]):
        
                out[i]=out[i].replace(j,choice_var[i][j])
    
    return out.tail(n_models)




def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14,target = 'Test'):
  
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap="YlGnBu")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    str_title = 'Confusion Matrix '+target
    plt.title(str_title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig



def glasgow_maker(input_df):
    
    list_glasgow = []

    for x in input_df.index:
        gbs_score = 0

        ## HR ##

        if input_df.loc[x]['HEARTRATE']>=100:
            gbs_score = gbs_score+1
        else:
            gbs_score = gbs_score+0

        ## SysBP ##

        if input_df.loc[x]['SysBP']<90:
            gbs_score = gbs_score+3
        elif input_df.loc[x]['SysBP']>=90 and input_df.loc[x]['SysBP']<99:
            gbs_score = gbs_score+2
        else:
            gbs_score = gbs_score+1

        ## BUN ##

        if input_df.loc[x]['BUN']>=70:
            gbs_score = gbs_score+6
        elif input_df.loc[x]['BUN']<70 and input_df.loc[x]['BUN']>=28:
            gbs_score = gbs_score+4
        elif input_df.loc[x]['BUN']<28 and input_df.loc[x]['BUN']>=22.4:
            gbs_score = gbs_score+3
        elif input_df.loc[x]['BUN']<22.4 and input_df.loc[x]['BUN']>=19:
            gbs_score = gbs_score+2
        else:
            gbs_score = gbs_score+0

        ## Heamoglobin (male) ##
        if input_df.loc[x]['gender']==1:
            if input_df[input_df['gender']==1].loc[x]['HEMOGLOBIN']<10:
                gbs_score = gbs_score+6
            elif input_df[input_df['gender']==1].loc[x]['HEMOGLOBIN']>10 and input_df[input_df['gender']==1].loc[x]['HEMOGLOBIN']<12:
                gbs_score = gbs_score+3
            elif input_df[input_df['gender']==1].loc[x]['HEMOGLOBIN']>12 and input_df[input_df['gender']==1].loc[x]['HEMOGLOBIN']<13:
                gbs_score = gbs_score+1
            else:
                gbs_score = gbs_score+0

        ## Heamoglobin (female) ##
        else:
            if input_df[input_df['gender']==0].loc[x]['HEMOGLOBIN']<10:
                gbs_score = gbs_score+6
            elif input_df[input_df['gender']==0].loc[x]['HEMOGLOBIN']>10 and input_df[input_df['gender']==0].loc[x]['HEMOGLOBIN']<12:
                gbs_score = gbs_score+1
            else:
                gbs_score = gbs_score+0

        ##  Commorbidities ##

        #if X.loc[x]['CARDIAC_ARRHYTHMIAS']==1 or X.loc[x]['CONGESTIVE_HEART_FAILURE']==1 or X.loc[x]['LIVER_DISEASE']==1:
            #gbs_score = gbs_score+2
        #else:
            #gbs_score = gbs_score+0


        list_glasgow.append(gbs_score)


    return list_glasgow


def future_checker(df_input,time_treshold):
    
    df2=df_input.copy()
    df=df_input[df_input.time >= time_treshold].copy()
    
    missing = [i for i in df2.index.unique() if i not in df.index.unique()]
    
    return missing
    
    
def objective_megaPipe(all_params,seed=12):
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    ##########################HYPER-PARAMETERS###################################
    
    
    #separate classificator params from general hyperparameters
    
    params_RF={}
    params_BG={}
    
    params_RF['n_estimators'] = all_params['n_estimators']
    params_RF['max_samples'] = all_params['max_samples']
    params_RF['max_features'] = all_params['max_features']
    params_RF['criterion'] = all_params['criterion']
    params_RF['max_depth'] = all_params['max_depth']
    params_RF['ccp_alpha']=all_params['ccp_alpha']
    params_RF['bootstrap']=all_params['bootstrap']
    params_RF['min_samples_leaf']=all_params['min_samples_leaf']
    params_RF['min_samples_split']=all_params['min_samples_split']
    
    
    params_BG['n_estimators'] = all_params['n_estimators_BG']
    params_BG['max_samples'] = all_params['max_samples_BG']
    params_BG['max_features'] = all_params['max_features_BG']
    params_BG['base_estimator__C']= all_params['base_estimator__C']
    params_BG['base_estimator__penalty']= all_params['base_estimator__penalty']
    params_BG['base_estimator__solver'] = all_params['base_estimator__solver']

    
    ############################CLUSTERING###########################################
    
    global X_train_mimic
    global y_train_mimic

    
    
    # VOTING
    

    classifier1 = RandomForestClassifier(class_weight='balanced',n_jobs=-1,random_state=seed)
    classifier1.set_params(**params_RF)
    
    
    lr = LogisticRegression(class_weight = 'balanced',max_iter = 100,random_state=seed)
    classifier2 = BaggingClassifier(lr,n_jobs=-1)
    classifier2.set_params(**params_BG)
    
    
    estimators = [('rf', classifier1),
                ('bg_lr', classifier2)]
    
    
    
    
    classifier = VotingClassifier(estimators=estimators,voting='soft',n_jobs = -1)
    
    

    
    
    
    shuffle = StratifiedKFold(n_splits=4, shuffle=True,random_state=seed)
    
    
    score = cross_val_score(classifier, X_train_mimic, y_train_mimic, cv=shuffle, scoring='roc_auc', n_jobs=-1)
    
    
    
    #progress bar
    global counter
    counter=counter+1
    
    if(counter%100==0):print(counter)
    return 1-score.mean()


    