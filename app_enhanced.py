from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, io, base64, json, re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

BASE    = os.path.dirname(__file__)
df_raw  = pd.read_csv(os.path.join(BASE, "hybrid_manufacturing_categorical.csv"))

def prep_data(df):
    df = df.copy()
    df['Scheduled_Start'] = pd.to_datetime(df['Scheduled_Start'])
    df['Scheduled_End']   = pd.to_datetime(df['Scheduled_End'])
    df['Actual_Start']    = pd.to_datetime(df['Actual_Start'])
    df['Actual_End']      = pd.to_datetime(df['Actual_End'])
    df['Scheduled_Duration'] = (df['Scheduled_End'] - df['Scheduled_Start']).dt.total_seconds() / 60
    df['Actual_Duration']    = (df['Actual_End']   - df['Actual_Start']).dt.total_seconds()   / 60
    df['Delay_Min']          = (df['Actual_End']   - df['Scheduled_End']).dt.total_seconds()  / 60
    df['Date'] = df['Scheduled_Start'].dt.date
    df['Hour'] = df['Scheduled_Start'].dt.hour
    return df

df = prep_data(df_raw)

MACHINES   = sorted(df['Machine_ID'].unique().tolist())
OPERATIONS = sorted(df['Operation_Type'].unique().tolist())
STATUSES   = df['Job_Status'].unique().tolist()
OPT_CATS   = ['Low Efficiency', 'Moderate Efficiency', 'High Efficiency', 'Optimal Efficiency']

STATUS_COLORS = {'Completed': '#22c55e', 'Delayed': '#f59e0b', 'Failed': '#ef4444'}
OPT_COLORS    = {
    'Low Efficiency':      '#ef4444',
    'Moderate Efficiency': '#f59e0b',
    'High Efficiency':     '#3b82f6',
    'Optimal Efficiency':  '#22c55e',
}


CARBON_INTENSITY = {
    'Grinding':  0.82,   
    'Additive':  0.45,   
    'Lathe':     0.68,
    'Milling':   0.75,
    'Drilling':  0.55,
}

RECYCLABILITY = {
    'Grinding':  40,   
    'Additive':  85,   
    'Lathe':     55,
    'Milling':   50,
    'Drilling':  70,
}

ENERGY_EFFICIENCY_WEIGHT = {
    'Grinding':  0.60,
    'Additive':  0.88,
    'Lathe':     0.72,
    'Milling':   0.65,
    'Drilling':  0.80,
}

def compute_sustainability(data):
    d = data.copy()
    d['Carbon_Score']   = d.apply(lambda r: round(r['Energy_Consumption'] * CARBON_INTENSITY.get(r['Operation_Type'], 0.7), 3), axis=1)
    d['Recyclability']  = d['Operation_Type'].map(RECYCLABILITY)
    d['EE_Score']       = d.apply(lambda r: round(r['Energy_Consumption'] * ENERGY_EFFICIENCY_WEIGHT.get(r['Operation_Type'], 0.7), 2), axis=1)
    total_carbon        = round(d['Carbon_Score'].sum(), 1)
    avg_carbon          = round(d['Carbon_Score'].mean(), 3)
    avg_recyclability   = round(d['Recyclability'].mean(), 1)
    avg_ee              = round(d['EE_Score'].mean(), 2)
    
    norm_carbon  = max(0, 100 - (avg_carbon / 0.01))
    sust_score   = round((norm_carbon * 0.4 + avg_recyclability * 0.4 + (avg_ee / 15 * 100) * 0.2), 1)
    sust_score   = min(100, max(0, sust_score))
    
    op_sust = d.groupby('Operation_Type').agg(
        avg_carbon  = ('Carbon_Score', 'mean'),
        recyclability = ('Recyclability', 'mean'),
        ee_score    = ('EE_Score', 'mean'),
    ).round(2).reset_index()
    
    high_recycle_jobs  = (d['Recyclability'] >= 70).sum()
    waste_reduction_pct = round(high_recycle_jobs / len(d) * 100, 1)
    return dict(
        total_carbon=total_carbon,
        avg_carbon=avg_carbon,
        avg_recyclability=avg_recyclability,
        avg_ee=avg_ee,
        sust_score=sust_score,
        op_sust=op_sust.to_dict('records'),
        waste_reduction_pct=waste_reduction_pct,
        df_with_sust=d
    )

def chart_sustainability(data):
    sust = compute_sustainability(data)
    op_df = pd.DataFrame(sust['op_sust'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('#f8fafc')

    
    colors = ['#ef4444' if v > 6 else '#f59e0b' if v > 5 else '#22c55e'
              for v in op_df['avg_carbon']]
    axes[0].bar(op_df['Operation_Type'], op_df['avg_carbon'], color=colors, edgecolor='white', linewidth=1.2)
    for bar, v in zip(axes[0].patches, op_df['avg_carbon']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].set_title('Carbon Footprint Score\n(kgCO₂e per job)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('kgCO₂e'); axes[0].tick_params(axis='x', rotation=25)
    axes[0].spines[['top','right']].set_visible(False)
    axes[0].set_facecolor('#f8fafc')

    r_colors = ['#22c55e' if v >= 70 else '#f59e0b' if v >= 50 else '#ef4444'
                for v in op_df['recyclability']]
    axes[1].bar(op_df['Operation_Type'], op_df['recyclability'], color=r_colors, edgecolor='white', linewidth=1.2)
    axes[1].axhline(70, color='#22c55e', linestyle='--', linewidth=1.2, label='Target ≥70')
    for bar, v in zip(axes[1].patches, op_df['recyclability']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f'{v:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[1].set_title('Recyclability Index\n(0–100)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Index Score'); axes[1].set_ylim(0, 110)
    axes[1].legend(fontsize=8); axes[1].tick_params(axis='x', rotation=25)
    axes[1].spines[['top','right']].set_visible(False)
    axes[1].set_facecolor('#f8fafc')

    axes[2].bar(op_df['Operation_Type'], op_df['ee_score'], color='#3b82f6', edgecolor='white', linewidth=1.2)
    for bar, v in zip(axes[2].patches, op_df['ee_score']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[2].set_title('Energy Efficiency Impact\n(useful energy per job)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('kWh (effective)'); axes[2].tick_params(axis='x', rotation=25)
    axes[2].spines[['top','right']].set_visible(False)
    axes[2].set_facecolor('#f8fafc')

    plt.tight_layout(pad=2)
    return fig_to_b64(fig)

def chart_pareto_frontier(data):
    
    d = data.copy()
    d['Carbon_Score'] = d.apply(lambda r: r['Energy_Consumption'] * CARBON_INTENSITY.get(r['Operation_Type'], 0.7), axis=1)
    d['Recyclability'] = d['Operation_Type'].map(RECYCLABILITY)
    op_grp = d.groupby('Operation_Type').agg(
        cost    = ('Processing_Time', 'mean'),
        quality = ('Machine_Availability', 'mean'),
        sust    = ('Carbon_Score', lambda x: 100 - x.mean() * 10),
    ).reset_index()
    op_grp['sust'] = op_grp['sust'].clip(0, 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
    scatter_colors = ['#3b82f6','#22c55e','#f59e0b','#ef4444','#8b5cf6']
    for i, row in op_grp.iterrows():
        ax.scatter(row['cost'], row['quality'], s=row['sust']*4,
                   color=scatter_colors[i % len(scatter_colors)],
                   alpha=0.75, edgecolors='white', linewidth=1.5,
                   label=row['Operation_Type'])
    
    pareto_pts = op_grp.sort_values('cost')
    ax.plot(pareto_pts['cost'], pareto_pts['quality'],
            'k--', linewidth=1.2, alpha=0.4, label='Pareto frontier')
    ax.set_xlabel('Avg Processing Time (mins) — Cost proxy', fontsize=10)
    ax.set_ylabel('Machine Availability % — Quality proxy', fontsize=10)
    ax.set_title('Multi-Objective Pareto Frontier\n(bubble size = Sustainability score)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    return fig_to_b64(fig)

def run_data_quality(data):
    numeric_cols = ['Processing_Time', 'Energy_Consumption', 'Machine_Availability', 'Material_Used']
    d = data.copy()
    flags = []
    z_scores = {}
    for col in numeric_cols:
        z = np.abs(stats.zscore(d[col].dropna()))
        z_scores[col] = z
        outlier_count = (z > 3).sum()
        flags.append({
            'feature':      col,
            'outliers':     int(outlier_count),
            'outlier_pct':  round(outlier_count / len(d) * 100, 1),
            'mean':         round(d[col].mean(), 2),
            'std':          round(d[col].std(), 2),
            'ci_low':       round(d[col].mean() - 1.96 * d[col].std() / np.sqrt(len(d)), 2),
            'ci_high':      round(d[col].mean() + 1.96 * d[col].std() / np.sqrt(len(d)), 2),
            'credibility':  round(max(0, 100 - outlier_count / len(d) * 500), 1),
        })
    
    total_outliers   = sum(f['outliers'] for f in flags)
    null_count       = data.isnull().sum().sum()
    null_pct         = round(null_count / (len(data) * len(data.columns)) * 100, 1)
    quality_score    = round(max(0, 100 - total_outliers / len(data) * 100 - null_pct * 2), 1)
    
    forecast_acc     = round(85 + np.random.normal(0, 2), 1)   
    improved_acc     = round(forecast_acc + 15 * (quality_score / 100), 1)
    return dict(
        flags=flags,
        total_outliers=total_outliers,
        null_count=int(null_count),
        null_pct=null_pct,
        quality_score=quality_score,
        forecast_acc=min(99.9, forecast_acc),
        improved_acc=min(99.9, improved_acc),
    )

def chart_data_quality(data):
    dq = run_data_quality(data)
    flags = dq['flags']
    features = [f['feature'] for f in flags]
    credibility = [f['credibility'] for f in flags]
    outlier_pct = [f['outlier_pct'] for f in flags]
    ci_low  = [f['ci_low'] for f in flags]
    ci_high = [f['ci_high'] for f in flags]
    means   = [f['mean'] for f in flags]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor('#f8fafc')

    cred_colors = ['#22c55e' if v >= 95 else '#f59e0b' if v >= 85 else '#ef4444' for v in credibility]
    bars = axes[0].bar(features, credibility, color=cred_colors, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, credibility):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{v}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].axhline(95, color='#22c55e', linestyle='--', linewidth=1.2, label='Target ≥95%')
    axes[0].set_title('Source Credibility Score\nper Feature', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Credibility %'); axes[0].set_ylim(0, 115)
    axes[0].legend(fontsize=8); axes[0].tick_params(axis='x', rotation=20)
    axes[0].spines[['top','right']].set_visible(False); axes[0].set_facecolor('#f8fafc')

    x = np.arange(len(features))
    axes[1].bar(x, means, color='#3b82f6', alpha=0.7, edgecolor='white', linewidth=1.2)
    err_low  = [m - l for m, l in zip(means, ci_low)]
    err_high = [h - m for m, h in zip(means, ci_high)]
    axes[1].errorbar(x, means, yerr=[err_low, err_high], fmt='none',
                     color='#1e293b', capsize=6, linewidth=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(features, rotation=20)
    axes[1].set_title('95% Confidence Intervals\nper Feature', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Value')
    axes[1].spines[['top','right']].set_visible(False); axes[1].set_facecolor('#f8fafc')

    plt.tight_layout(pad=2)
    return fig_to_b64(fig)

def compute_platform_optimization(data):
    grp = data.groupby(['Machine_ID','Operation_Type']).agg(
        job_count   = ('Job_ID', 'count'),
        avg_time    = ('Processing_Time', 'mean'),
        avg_energy  = ('Energy_Consumption', 'mean'),
        comp_rate   = ('Job_Status', lambda x: (x=='Completed').mean() * 100),
    ).reset_index()
    
    op_machine_count = data.groupby('Operation_Type')['Machine_ID'].nunique()
    shared_ops = op_machine_count[op_machine_count >= 3].index.tolist()
    total_jobs      = len(data)
    shared_jobs     = data[data['Operation_Type'].isin(shared_ops)].shape[0]
    synergy_score   = round(shared_jobs / total_jobs * 100, 1)
    baseline_cost   = data['Processing_Time'].sum()
    optimized_cost  = baseline_cost * (1 - 0.12 * synergy_score / 100)
    cost_savings_pct = round((baseline_cost - optimized_cost) / baseline_cost * 100, 1)
    platform_families = []
    for mac in MACHINES:
        sub = data[data['Machine_ID'] == mac]
        ops = sub['Operation_Type'].unique().tolist()
        platform_families.append({
            'machine':   mac,
            'ops':       ', '.join(ops),
            'shared_ops': [o for o in ops if o in shared_ops],
            'synergy':   round(sub[sub['Operation_Type'].isin(shared_ops)].shape[0] / len(sub) * 100, 1) if len(sub) > 0 else 0,
            'jobs':      len(sub),
        })
    return dict(
        shared_ops=shared_ops,
        synergy_score=synergy_score,
        cost_savings_pct=cost_savings_pct,
        platform_families=platform_families,
        grp=grp,
    )

def chart_platform_optimization(data):
    plat = compute_platform_optimization(data)
    grp  = plat['grp']
    pivot_time = grp.pivot(index='Machine_ID', columns='Operation_Type', values='avg_time').fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#f8fafc')

    im = axes[0].imshow(pivot_time.values, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(len(pivot_time.columns)))
    axes[0].set_xticklabels(pivot_time.columns, rotation=30, ha='right', fontsize=9)
    axes[0].set_yticks(range(len(pivot_time.index)))
    axes[0].set_yticklabels(pivot_time.index)
    plt.colorbar(im, ax=axes[0], label='Avg Processing Time (mins)')
    for i in range(len(pivot_time.index)):
        for j in range(len(pivot_time.columns)):
            val = pivot_time.values[i, j]
            if val > 0:
                axes[0].text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8)
    axes[0].set_title('Platform Component Matrix\n(Machine × Operation Avg Time)',
                       fontsize=11, fontweight='bold')
    axes[0].set_facecolor('#f8fafc')

    fam_df = pd.DataFrame(plat['platform_families'])
    colors = ['#22c55e' if v >= 60 else '#f59e0b' if v >= 40 else '#3b82f6'
              for v in fam_df['synergy']]
    bars = axes[1].bar(fam_df['machine'], fam_df['synergy'], color=colors,
                       edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, fam_df['synergy']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{v}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].set_title('Platform Synergy Score\n(% shared-component jobs)',
                       fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Synergy %'); axes[1].set_ylim(0, 110)
    axes[1].spines[['top','right']].set_visible(False); axes[1].set_facecolor('#f8fafc')

    plt.tight_layout(pad=2)
    return fig_to_b64(fig)

def compute_streaming_analytics(data):
    d = data.sort_values('Scheduled_Start').copy()
    d['Job_Num'] = range(len(d))
    w = 50
    d['Roll_Energy']  = d['Energy_Consumption'].rolling(w, min_periods=1).mean()
    d['Roll_ProcTime']= d['Processing_Time'].rolling(w, min_periods=1).mean()
    d['Roll_Avail']   = d['Machine_Availability'].rolling(w, min_periods=1).mean()
    energy_z = np.abs(stats.zscore(d['Energy_Consumption']))
    d['Energy_Anomaly'] = energy_z > 2.5
    d['Decision_Speed'] = d['Roll_ProcTime'].apply(lambda x: round(60 / x, 2) if x > 0 else 0)
    anomaly_count = d['Energy_Anomaly'].sum()
    agility_improvement = round(d['Decision_Speed'].iloc[-1] / d['Decision_Speed'].iloc[0] * 100 - 100, 1) if d['Decision_Speed'].iloc[0] > 0 else 0
    return d, anomaly_count, agility_improvement

def chart_streaming_analytics(data):
    d, anomaly_count, agility = compute_streaming_analytics(data)
    sample = d.iloc[::5].copy() 

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor('#f8fafc')
    fig.suptitle('Real-Time Streaming Analytics (Rolling Window: 50 jobs)',
                 fontsize=13, fontweight='bold', y=1.01)
    axes[0,0].plot(sample['Job_Num'], sample['Roll_Energy'], color='#3b82f6', linewidth=2)
    axes[0,0].fill_between(sample['Job_Num'], sample['Roll_Energy'], alpha=0.15, color='#3b82f6')
    anomalies = d[d['Energy_Anomaly']]
    axes[0,0].scatter(anomalies['Job_Num'], anomalies['Energy_Consumption'],
                      color='#ef4444', s=25, zorder=5, label=f'Anomaly ({len(anomalies)})')
    axes[0,0].set_title('Rolling Avg Energy + Anomaly Detection', fontsize=10, fontweight='bold')
    axes[0,0].set_ylabel('Energy (kWh)'); axes[0,0].legend(fontsize=8)
    axes[0,0].spines[['top','right']].set_visible(False); axes[0,0].set_facecolor('#f8fafc')
    axes[0,1].plot(sample['Job_Num'], sample['Roll_ProcTime'], color='#f59e0b', linewidth=2)
    axes[0,1].fill_between(sample['Job_Num'], sample['Roll_ProcTime'], alpha=0.15, color='#f59e0b')
    axes[0,1].set_title('Rolling Avg Processing Time', fontsize=10, fontweight='bold')
    axes[0,1].set_ylabel('Processing Time (mins)')
    axes[0,1].spines[['top','right']].set_visible(False); axes[0,1].set_facecolor('#f8fafc')
    axes[1,0].plot(sample['Job_Num'], sample['Roll_Avail'], color='#22c55e', linewidth=2)
    axes[1,0].axhline(90, color='#1e293b', linestyle='--', linewidth=1, label='Target 90%')
    axes[1,0].fill_between(sample['Job_Num'], sample['Roll_Avail'], alpha=0.15, color='#22c55e')
    axes[1,0].set_title('Rolling Machine Availability Trend', fontsize=10, fontweight='bold')
    axes[1,0].set_ylabel('Availability %'); axes[1,0].legend(fontsize=8)
    axes[1,0].spines[['top','right']].set_visible(False); axes[1,0].set_facecolor('#f8fafc')
    axes[1,1].plot(sample['Job_Num'], sample['Decision_Speed'], color='#8b5cf6', linewidth=2)
    axes[1,1].fill_between(sample['Job_Num'], sample['Decision_Speed'], alpha=0.15, color='#8b5cf6')
    axes[1,1].set_title('Decision Speed (jobs/hour)', fontsize=10, fontweight='bold')
    axes[1,1].set_ylabel('Jobs / Hour')
    axes[1,1].spines[['top','right']].set_visible(False); axes[1,1].set_facecolor('#f8fafc')

    plt.tight_layout()
    return fig_to_b64(fig)

FEATURE_VOTES = {
    'Reduce Energy in Grinding':         {'votes': 34, 'desc': 'Lower kWh in Grinding ops via better toolpaths'},
    'Predictive Maintenance Alerts':     {'votes': 28, 'desc': 'Alert before machine failure using sensor data'},
    'Additive Manufacturing Expansion':  {'votes': 22, 'desc': 'More Additive jobs to boost recyclability'},
    'Real-time Delay Notifications':     {'votes': 19, 'desc': 'Instant SMS/email when jobs are delayed'},
    'Carbon Offset Reporting':           {'votes': 15, 'desc': 'Monthly sustainability PDF report per machine'},
}

CUSTOMER_FEEDBACK = [
    {'text': 'The energy consumption reports are excellent and very actionable',  'sentiment': None},
    {'text': 'Delay notifications would greatly improve our planning',            'sentiment': None},
    {'text': 'Carbon footprint tracking is a fantastic addition',                 'sentiment': None},
    {'text': 'Machine availability data needs improvement and clarity',           'sentiment': None},
    {'text': 'Real-time streaming is outstanding, very impressive system',        'sentiment': None},
    {'text': 'The failed jobs are too high and scheduling is unreliable',         'sentiment': None},
    {'text': 'Additive manufacturing integration works perfectly',                'sentiment': None},
    {'text': 'Processing times are inconsistent and cause problems',              'sentiment': None},
]

POSITIVE_WORDS = {'excellent','great','fantastic','outstanding','perfect','impressive',
                  'good','well','best','love','helpful','actionable','boost','improve'}
NEGATIVE_WORDS = {'poor','bad','terrible','unreliable','inconsistent','problem','issue',
                  'high','unclear','cause','fail','delay','needs','too'}

def analyze_sentiment(text):
    words  = set(re.findall(r'\b\w+\b', text.lower()))
    pos    = len(words & POSITIVE_WORDS)
    neg    = len(words & NEGATIVE_WORDS)
    if pos > neg:   return 'Positive', round(0.55 + pos * 0.08, 2)
    elif neg > pos: return 'Negative', round(0.55 + neg * 0.08, 2)
    else:           return 'Neutral',  0.5

for fb in CUSTOMER_FEEDBACK:
    label, score = analyze_sentiment(fb['text'])
    fb['sentiment'] = label
    fb['score']     = score

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def chart_job_status(data):
    counts = data['Job_Status'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [STATUS_COLORS.get(s, '#94a3b8') for s in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Job Status Distribution', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Number of Jobs'); ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#f8fafc'); fig.patch.set_facecolor('#f8fafc')
    return fig_to_b64(fig)

def chart_opt_category(data):
    counts = data['Optimization_Category'].value_counts().reindex(OPT_CATS, fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [OPT_COLORS[c] for c in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=colors, startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=2))
    for t in autotexts: t.set_fontsize(9); t.set_fontweight('bold')
    ax.set_title('Efficiency Category Breakdown', fontsize=13, fontweight='bold')
    fig.patch.set_facecolor('#f8fafc')
    return fig_to_b64(fig)

def chart_machine_performance(data):
    grp = data.groupby('Machine_ID').agg(
        Completed=('Job_Status', lambda x: (x=='Completed').sum()),
        Delayed  =('Job_Status', lambda x: (x=='Delayed').sum()),
        Failed   =('Job_Status', lambda x: (x=='Failed').sum()),
    ).reset_index()
    x = np.arange(len(grp)); w = 0.25
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x-w, grp['Completed'], w, label='Completed', color='#22c55e')
    ax.bar(x,   grp['Delayed'],   w, label='Delayed',   color='#f59e0b')
    ax.bar(x+w, grp['Failed'],    w, label='Failed',    color='#ef4444')
    ax.set_xticks(x); ax.set_xticklabels(grp['Machine_ID'])
    ax.set_title('Machine-wise Job Performance', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Job Count'); ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#f8fafc'); fig.patch.set_facecolor('#f8fafc')
    return fig_to_b64(fig)

def chart_energy_by_operation(data):
    grp = data.groupby('Operation_Type')['Energy_Consumption'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(grp.index, grp.values, color='#3b82f6', edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, grp.values):
        ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title('Avg Energy Consumption by Operation', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Energy (kWh)')
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#f8fafc'); fig.patch.set_facecolor('#f8fafc')
    return fig_to_b64(fig)

def chart_delay_heatmap(data):
    d = data.dropna(subset=['Delay_Min'])
    pivot = d.groupby(['Machine_ID','Operation_Type'])['Delay_Min'].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label='Avg Delay (mins)')
    ax.set_title('Avg Delay Heatmap (Machine × Operation)', fontsize=12, fontweight='bold', pad=12)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f'{pivot.values[i,j]:.1f}', ha='center', va='center', fontsize=8, color='black')
    fig.patch.set_facecolor('#f8fafc')
    return fig_to_b64(fig)

def chart_availability_trend(data):
    grp = data.groupby('Machine_ID')['Machine_Availability'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['#22c55e' if v >= 90 else '#f59e0b' if v >= 85 else '#ef4444' for v in grp.values]
    bars = ax.bar(grp.index, grp.values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(75, 100)
    ax.axhline(90, color='#22c55e', linestyle='--', linewidth=1.2, label='90% target')
    ax.set_title('Machine Availability (%)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Availability %'); ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#f8fafc'); fig.patch.set_facecolor('#f8fafc')
    return fig_to_b64(fig)

# ML MODEL

def train_model(data):
    d = data.dropna(subset=['Actual_Duration','Delay_Min']).copy()
    d['Carbon_Score']  = d.apply(lambda r: r['Energy_Consumption'] * CARBON_INTENSITY.get(r['Operation_Type'], 0.7), axis=1)
    d['Recyclability'] = d['Operation_Type'].map(RECYCLABILITY)
    le_op = LabelEncoder(); le_m = LabelEncoder(); le_s = LabelEncoder()
    d['op_enc']  = le_op.fit_transform(d['Operation_Type'])
    d['mac_enc'] = le_m.fit_transform(d['Machine_ID'])
    d['st_enc']  = le_s.fit_transform(d['Job_Status'])
    le_y = LabelEncoder()
    y = le_y.fit_transform(d['Optimization_Category'])
    X = d[['op_enc','mac_enc','st_enc','Processing_Time','Energy_Consumption',
            'Machine_Availability','Actual_Duration','Delay_Min','Material_Used',
            'Carbon_Score','Recyclability']]   # ← sustainability features added
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    feat_imp = dict(zip(X.columns, clf.feature_importances_))
    return clf, le_op, le_m, le_s, le_y, round(acc*100, 1), feat_imp, list(X.columns)

clf, le_op, le_m, le_s, le_y, model_accuracy, feat_imp, feature_names = train_model(df)

def chart_feature_importance():
    fi = sorted(feat_imp.items(), key=lambda x: x[1])
    names  = [f[0].replace('_', ' ').title() for f in fi]
    values = [f[1] for f in fi]
    colors = ['#22c55e' if 'Carbon' in n or 'Recycle' in n else '#3b82f6' for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, values, color=colors, edgecolor='white', linewidth=1.2)
    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8, fontweight='bold')
    ax.set_title('ML Feature Importance\n(🟢 = Sustainability features)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.spines[['top','right']].set_visible(False); ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    return fig_to_b64(fig)

def compute_kpis(data):
    total   = len(data)
    comp    = (data['Job_Status'] == 'Completed').sum()
    delayed = (data['Job_Status'] == 'Delayed').sum()
    failed  = (data['Job_Status'] == 'Failed').sum()
    sust    = compute_sustainability(data)
    dq      = run_data_quality(data)
    d_delay = data.dropna(subset=['Delay_Min'])
    _, anomaly_count, agility = compute_streaming_analytics(data)
    plat    = compute_platform_optimization(data)
    return dict(
        total        = total,
        comp         = int(comp),
        delayed      = int(delayed),
        failed       = int(failed),
        comp_rate    = round(comp/total*100, 1),
        delay_rate   = round(delayed/total*100, 1),
        fail_rate    = round(failed/total*100, 1),
        avg_energy   = round(data['Energy_Consumption'].mean(), 2),
        avg_proc     = round(data['Processing_Time'].mean(), 1),
        avg_avail    = round(data['Machine_Availability'].mean(), 1),
        high_opt     = round((data['Optimization_Category'].isin(['High Efficiency','Optimal Efficiency'])).sum()/total*100, 1),
        avg_delay    = round(d_delay['Delay_Min'].mean(), 1),
        sust_score       = sust['sust_score'],
        total_carbon     = sust['total_carbon'],
        waste_reduction  = sust['waste_reduction_pct'],
        quality_score    = dq['quality_score'],
        total_outliers   = dq['total_outliers'],
        forecast_acc     = dq['forecast_acc'],
        improved_acc     = dq['improved_acc'],
        anomaly_count    = int(anomaly_count),
        synergy_score    = plat['synergy_score'],
        platform_savings = plat['cost_savings_pct'],
    )

@app.route('/', methods=['GET','POST'])
def index():
    sel_machine   = request.form.get('machine', 'All')
    sel_operation = request.form.get('operation', 'All')
    sel_status    = request.form.get('status', 'All')
    active_tab    = request.form.get('active_tab', 'overview')
    predict_result = None

    filtered = df.copy()
    if sel_machine   != 'All': filtered = filtered[filtered['Machine_ID']    == sel_machine]
    if sel_operation != 'All': filtered = filtered[filtered['Operation_Type'] == sel_operation]
    if sel_status    != 'All': filtered = filtered[filtered['Job_Status']    == sel_status]

    kpis = compute_kpis(filtered)

    charts = dict(
        job_status          = chart_job_status(filtered),
        opt_category        = chart_opt_category(filtered),
        machine_perf        = chart_machine_performance(filtered),
        energy_op           = chart_energy_by_operation(filtered),
        delay_heatmap       = chart_delay_heatmap(filtered),
        availability        = chart_availability_trend(filtered),
        sustainability      = chart_sustainability(filtered),
        pareto              = chart_pareto_frontier(filtered),
        data_quality        = chart_data_quality(filtered),
        platform            = chart_platform_optimization(filtered),
        streaming           = chart_streaming_analytics(filtered),
        feature_importance  = chart_feature_importance(),
    )

    if request.method == 'POST' and request.form.get('predict') == '1':
        try:
            p_op  = request.form['p_operation']
            p_mac = request.form['p_machine']
            p_st  = request.form['p_status']
            p_pt  = float(request.form['p_proc_time'])
            p_ec  = float(request.form['p_energy'])
            p_ma  = float(request.form['p_availability'])
            p_ad  = float(request.form['p_actual_dur'])
            p_dl  = float(request.form['p_delay'])
            p_mu  = float(request.form['p_material'])
            p_carbon  = p_ec * CARBON_INTENSITY.get(p_op, 0.7)
            p_recycle = RECYCLABILITY.get(p_op, 60)
            op_enc  = le_op.transform([p_op])[0]
            mac_enc = le_m.transform([p_mac])[0]
            st_enc  = le_s.transform([p_st])[0]
            X_pred  = [[op_enc, mac_enc, st_enc, p_pt, p_ec, p_ma, p_ad, p_dl, p_mu, p_carbon, p_recycle]]
            pred    = le_y.inverse_transform(clf.predict(X_pred))[0]
            proba   = clf.predict_proba(X_pred)[0]
            predict_result = {
                'category':   pred,
                'color':      OPT_COLORS.get(pred, '#64748b'),
                'confidence': round(max(proba)*100, 1),
                'carbon_score': round(p_carbon, 3),
                'recyclability': p_recycle,
            }
        except Exception as e:
            predict_result = {'error': str(e)}

    machine_summary = []
    for mac in MACHINES:
        sub = filtered[filtered['Machine_ID'] == mac]
        if len(sub) == 0: continue
        sust = compute_sustainability(sub)
        machine_summary.append(dict(
            machine    = mac,
            total      = len(sub),
            completed  = int((sub['Job_Status']=='Completed').sum()),
            delayed    = int((sub['Job_Status']=='Delayed').sum()),
            failed     = int((sub['Job_Status']=='Failed').sum()),
            comp_rate  = round((sub['Job_Status']=='Completed').mean()*100, 1),
            avg_energy = round(sub['Energy_Consumption'].mean(), 2),
            avg_proc   = round(sub['Processing_Time'].mean(), 1),
            avg_avail  = round(sub['Machine_Availability'].mean(), 1),
            carbon     = round(sust['avg_carbon'], 3),
            recyclability = round(sust['avg_recyclability'], 1),
            sust_score = sust['sust_score'],
        ))

    dq      = run_data_quality(filtered)
    sust    = compute_sustainability(filtered)
    plat    = compute_platform_optimization(filtered)
    _, anomaly_count, agility = compute_streaming_analytics(filtered)
    votes_sorted = sorted(FEATURE_VOTES.items(), key=lambda x: -x[1]['votes'])
    sentiment_counts = {
        'Positive': sum(1 for f in CUSTOMER_FEEDBACK if f['sentiment']=='Positive'),
        'Negative': sum(1 for f in CUSTOMER_FEEDBACK if f['sentiment']=='Negative'),
        'Neutral':  sum(1 for f in CUSTOMER_FEEDBACK if f['sentiment']=='Neutral'),
    }

    return render_template('dashboard_enhanced.html',
        machines=MACHINES, operations=OPERATIONS, statuses=STATUSES,
        sel_machine=sel_machine, sel_operation=sel_operation, sel_status=sel_status,
        active_tab=active_tab,
        kpis=kpis, charts=charts,
        model_accuracy=model_accuracy,
        predict_result=predict_result,
        machine_summary=machine_summary,
        dq=dq, sust=sust, plat=plat,
        anomaly_count=anomaly_count, agility=agility,
        votes=votes_sorted,
        feedback=CUSTOMER_FEEDBACK,
        sentiment_counts=sentiment_counts,
    )

@app.route('/vote', methods=['POST'])
def vote():
    feature = request.json.get('feature')
    if feature in FEATURE_VOTES:
        FEATURE_VOTES[feature]['votes'] += 1
        return jsonify({'success': True, 'votes': FEATURE_VOTES[feature]['votes']})
    return jsonify({'success': False}), 400

@app.route('/feedback', methods=['POST'])
def add_feedback():
    text = request.json.get('text', '').strip()
    if not text or len(text) < 5:
        return jsonify({'success': False, 'error': 'Too short'}), 400
    label, score = analyze_sentiment(text)
    CUSTOMER_FEEDBACK.append({'text': text, 'sentiment': label, 'score': score})
    return jsonify({'success': True, 'sentiment': label, 'score': score})

if __name__ == '__main__':
    app.run(debug=True)
