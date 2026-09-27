import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"Liberation Serif","font.size":9,"axes.linewidth":0.7,"savefig.dpi":300})
r=pd.read_csv("../data/paper_tables/rq1_summary.csv")
MODELS=[("Qwen3","Qwen3-4B"),("Qwen2.5","Qwen2.5-3B-Instruct"),("Llama","Llama-3.2-3B-Instruct")]
QS=["F16","Q8_0","Q6_K","Q4_K_M","Q3_K_M","Q2_K"]

# ---------- Figure 1: pipeline ----------
fig,ax=plt.subplots(figsize=(6.6,1.9)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,30)
boxes=[(1,"GGUF weights\n(F16, static and\nimatrix K-quants)"),(21,"Dequantize to\nFP16 tensors\n(Transformers)"),
       (41,"0-shot log-likelihood\nscoring (lm-eval)\nAR & EN items"),(61,"Pair items across\nlanguages; F16 vs.\nquantized"),(81,"Raw & chance-norm.\nloss; bootstrap CI;\nHolm correction")]
for x,t in boxes:
    ax.add_patch(FancyBboxPatch((x,6),17.5,18,boxstyle="round,pad=0.4",fc="#EEEEEE",ec="black",lw=0.8))
    ax.text(x+8.75,15,t,ha="center",va="center",fontsize=8)
for x in [19.2,39.2,59.2,79.2]:
    ax.annotate("",xy=(x+1.5,15),xytext=(x-0.6,15),arrowprops=dict(arrowstyle="-|>",lw=0.9,color="black"))
fig.savefig("fig1_pipeline.png",bbox_inches="tight"); plt.close()

# ---------- Figure 2: normalized loss curves ----------
fig,axes=plt.subplots(2,3,figsize=(6.6,4.3),sharex=True)
xs=np.arange(len(QS))
for j,(m,label) in enumerate(MODELS):
    for i,b in enumerate(["Belebele","GlobalMMLU"]):
        ax=axes[i,j]
        for var,ls in [("static","-"),("imatrix","--")]:
            for lang,col,mk in [("ar","black","o"),("en","#888888","s")]:
                ys=[0.0]; xx=[0]
                for k,q in enumerate(QS[1:],1):
                    row=r[(r.model==m)&(r.variant==var)&(r.quant==q)&(r.bench==b)]
                    if len(row): xx.append(k); ys.append(float(row["lost_"+lang].iloc[0]))
                ax.plot(xx,ys,ls=ls,color=col,marker=mk,ms=3.2,lw=1.1,mfc="white" if var=="imatrix" else col,
                        label=f"{'Arabic' if lang=='ar' else 'English'}, {var}")
        ax.axhline(100,color="black",lw=0.5,ls=":")
        ax.set_ylim(-8,112); ax.set_xticks(xs); ax.set_xticklabels(QS,rotation=45,fontsize=7.5)
        ax.grid(axis="y",lw=0.3,color="#CCCCCC")
        if i==0: ax.set_title(label,fontsize=9)
        if j==0: ax.set_ylabel(f"{'Belebele' if b=='Belebele' else 'Global-MMLU-Lite'}\nknowledge lost (%)",fontsize=8.5)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(0.5,-0.03))
fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig("fig2_loss_curves.png",bbox_inches="tight"); plt.close()

# ---------- Figure 3: forest plot raw vs normalized excess ----------
d=r[r.quant.isin(["Q4_K_M","Q3_K_M","Q2_K"])].copy()
d=d[~((d.model=="Qwen2.5")&(d.variant=="static")&(d.quant=="Q2_K"))]
order={"Q4_K_M":0,"Q3_K_M":1,"Q2_K":2}; mo={"Qwen3":0,"Qwen2.5":1,"Llama":2}
d=d.assign(o=d.quant.map(order),mo=d.model.map(mo)).sort_values(["o","bench","mo","variant"],ascending=[True,True,True,False])
short={"Qwen3":"Qwen3","Qwen2.5":"Qwen2.5","Llama":"Llama"}
labels=[f"{q.replace('_K_M','_K_M')} {('BLB' if b=='Belebele' else 'GMM')} {short[m]} {'st' if v=='static' else 'i1'}" for q,b,m,v in zip(d.quant,d.bench,d.model,d.variant)]
y=np.arange(len(d))[::-1]
fig,axes=plt.subplots(1,2,figsize=(6.6,6.4),sharey=True)
for ax,(c,lo,hi,p,title) in zip(axes,[("excess_raw","raw_lo","raw_hi","p_raw_holm","(a) Raw excess drop (accuracy points)"),
                                      ("excess_norm","norm_lo","norm_hi","p_norm_holm","(b) Chance-normalized excess (% of knowledge)")]):
    sig=d[p].values<0.05
    ax.hlines(y,d[lo],d[hi],color="black",lw=0.8)
    ax.scatter(d[c][~sig],y[~sig],marker="o",s=14,facecolor="white",edgecolor="black",zorder=3,label="n.s. (Holm)")
    ax.scatter(d[c][sig],y[sig],marker="o",s=16,facecolor="black",edgecolor="black",zorder=3,label="p < 0.05 (Holm)")
    ax.axvline(0,color="black",lw=0.6,ls="--"); ax.set_title(title,fontsize=8.5); ax.grid(axis="x",lw=0.3,color="#CCCCCC")
    for yy in [len(d)-12-0.5, len(d)-24-0.5]: ax.axhline(yy,color="#999999",lw=0.5)
axes[0].set_yticks(y); axes[0].set_yticklabels(labels,fontsize=6.5)
axes[1].legend(loc="upper right",fontsize=7,frameon=True)
fig.tight_layout(); fig.savefig("fig3_forest.png",bbox_inches="tight"); plt.close()
print(len(d),"rows in forest")
