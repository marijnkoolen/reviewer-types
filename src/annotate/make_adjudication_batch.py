"""Select reviews for human adjudication (batch 1) and write a sheet."""
import pandas as pd, numpy as np, sys
import glob
n_reviews = int(sys.argv[1]) if len(sys.argv)>1 else 10
name = sys.argv[2] if len(sys.argv)>2 else 'batch1'
seed = int(sys.argv[3]) if len(sys.argv)>3 else 0
used = set()
for f in glob.glob('data/adjudication/*_sheet.tsv'):
    if not f.endswith(name+'_sheet.tsv'): used |= set(pd.read_csv(f,sep='\t').review_id)
d = pd.read_csv('data/raw_annotations.tsv.gz', sep='\t').fillna(0)
labs = [c for c in d.columns[7:] if '--' in c or c in ('Author','Classification','Other_works','Recommendations')]
code = {'Author':'AUT','Classification':'CLS','Other_works':'OTW','Recommendations':'REC',
 'Content--Narrative':'NAR','Content--Other':'COO','Content--Quote':'QUO','Content--Theme':'THM',
 'Reader_response--Evaluation_of_quality':'EVA','Reader_response--Feelings':'FEE',
 'Reader_response--Identification_and_immersion':'IDI','Reader_response--Reading_Context':'CTX',
 'Reader_response--Reception':'RCP','Reader_response--Reflection':'REF','Style--Context':'SCX',
 'Style--Structure':'STR','Style--Stylistic_features':'STY'}
full = d.groupby('review_id').annotator.nunique(); ids = full[full==3].index
d = d[d.review_id.isin(ids) & ~d.review_id.isin(used)]
weak = ['Reader_response--Feelings','Reader_response--Identification_and_immersion','Content--Theme',
        'Reader_response--Reflection','Style--Context','Style--Structure','Content--Other']
P = {l: d.pivot_table(index='sent_id', columns='annotator', values=l, aggfunc='first') for l in labs}
dis = pd.DataFrame({l: (P[l].sum(axis=1).between(1,2)).astype(int) for l in weak})
sent = d.drop_duplicates('sent_id').set_index('sent_id')
dis['review_id'] = sent.loc[dis.index,'review_id']
# F/I swap: one annotator F, another I
F,I = P[weak[0]],P[weak[1]]
swap=[( ((F[a]==1)&(I[b]==1)&(I[a]==0)&(F[b]==0)).astype(int)) for a in F for b in F if a!=b]
dis['swap']=sum(swap)
g = dis.groupby('review_id').sum(); g['n']=sent.groupby('review_id').size()
g = g[(g.n>=8)&(g.n<=16)]
# want coverage: swap-heavy + rare-label disagreements
g['score'] = 3*g.swap + g[weak[2:]].sum(axis=1)*1.0 + g['Style--Context']*2 + g['Style--Structure']*2
rng=np.random.default_rng(0)
top = g.sort_values('score',ascending=False).head(60)
pick = list(top.sample(n_reviews, random_state=seed).index)
rows=[]
for r in pick:
    s = d[d.review_id==r].sort_values('sent_num')
    for sid,grp in s.groupby('sent_id',sort=False):
        row={'review_id':r,'sent_id':sid,'sent_text':grp.sent_text.iloc[0]}
        for a in ['anno1','anno2','anno3']:
            x=grp[grp.annotator==a]
            row[a]=' '.join(code[l] for l in labs if len(x) and x[l].iloc[0]==1) 
        row['adjudicated']=''; row['note']=''
        rows.append(row)
out=pd.DataFrame(rows); out.to_csv('data/adjudication/'+name+'_sheet.tsv',sep='\t',index=False)
# readable markdown
with open('data/adjudication/'+name+'_sheet.md','w') as f:
    f.write('# Adjudication batch 1\n\nCodes: '+', '.join(f'{v}={k}' for k,v in code.items())+'\n\n')
    for r,grp in out.groupby('review_id',sort=False):
        f.write(f'\n## {r}\n\n')
        for _,x in grp.iterrows():
            f.write(f"**{x.sent_id[-3:]}** {x.sent_text}\n  - a1: {x.anno1 or '-'} | a2: {x.anno2 or '-'} | a3: {x.anno3 or '-'} | **gold:** \n\n")
print(g.loc[pick,['n','swap','score']]); print(len(out),'sentences')
