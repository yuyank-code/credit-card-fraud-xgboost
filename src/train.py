from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
for d in ['data','models','results']: (ROOT/d).mkdir(exist_ok=True)
rng=np.random.default_rng(7); n=100000; fraud_n=600
X=rng.normal(0,1,(n,28)); amount=np.exp(rng.normal(3.2,1.1,n)); time=rng.uniform(0,172800,n); y=np.zeros(n,dtype=int); idx=rng.choice(n,fraud_n,replace=False); y[idx]=1
X[idx,2]+=2.2; X[idx,4]-=1.8; X[idx,9]+=1.7; X[idx,13]-=2.5; X[idx,16]+=1.9; amount[idx]*=1.8
cols=[f'V{i}' for i in range(1,29)]; df=pd.DataFrame(X,columns=cols); df['Time']=time; df['Amount']=amount; df['Class']=y; df.to_csv(ROOT/'data/synthetic_credit_transactions.csv',index=False)
X=df.drop(columns='Class'); y=df.Class
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42); Xtr_sm,ytr_sm=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr,ytr)
model=XGBClassifier(n_estimators=260,max_depth=5,learning_rate=.08,subsample=.85,colsample_bytree=.85,min_child_weight=2,reg_lambda=2,objective='binary:logistic',eval_metric='auc',n_jobs=4,random_state=42)
model.fit(Xtr_sm,ytr_sm); prob=model.predict_proba(Xte)[:,1]
auc=roc_auc_score(yte,prob); ap=average_precision_score(yte,prob); precision,recall,thresholds=precision_recall_curve(yte,prob); f1=2*precision*recall/(precision+recall+1e-12); i=np.argmax(f1[:-1]); threshold=float(thresholds[i]); pred=(prob>=threshold).astype(int); cm=confusion_matrix(yte,pred)
metrics={'n_total':int(n),'fraud_count':int(fraud_n),'fraud_rate':float(y.mean()),'train_after_smote':int(len(ytr_sm)),'roc_auc':auc,'average_precision_pr_auc':ap,'selected_threshold_max_f1':threshold,'precision':precision_score(yte,pred),'recall':recall_score(yte,pred),'f1':f1_score(yte,pred),'confusion_matrix':cm.tolist()}
joblib.dump(model,ROOT/'models/xgboost_smote_fraud.joblib'); json.dump(metrics,open(ROOT/'results/metrics.json','w'),indent=2)
fi=pd.DataFrame({'feature':X.columns,'importance':model.feature_importances_}).sort_values('importance',ascending=False); fi.to_csv(ROOT/'results/feature_importance.csv',index=False)
fpr,tpr,_=roc_curve(yte,prob); plt.figure(figsize=(7,5)); plt.plot(fpr,tpr,label=f'ROC-AUC={auc:.3f}'); plt.plot([0,1],[0,1],'--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Fraud ROC'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'results/roc_curve.png',dpi=160); plt.close()
plt.figure(figsize=(7,5)); plt.plot(recall,precision,label=f'PR-AUC={ap:.3f}'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Fraud Precision-Recall'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'results/pr_curve.png',dpi=160); plt.close()
