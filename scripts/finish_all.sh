#!/bin/bash
echo "Waiting for seed 1b to finish..."
while kill -0 60791 2>/dev/null; do
    sleep 60
done
echo "Seed 1b finished."

echo "Merging 3-seed CSV..."
/home/zeyufu/Desktop/.conda/bin/python3 -c "
import pandas as pd, glob
seed1 = sorted(glob.glob('/home/zeyufu/Desktop/PanODE-LAB/benchmarks/benchmark_results/crossdata/csv/results_*_20260221_*.csv'))
df1 = pd.concat([pd.read_csv(f) for f in seed1])
df1['seed'] = 1
df_multiseed = pd.read_csv('/home/zeyufu/Desktop/PanODE-LAB/benchmarks/benchmark_results/crossdata/csv/results_combined_multiseed.csv')
df_3seed = pd.concat([df_multiseed, df1])
df_3seed.to_csv('/home/zeyufu/Desktop/PanODE-LAB/benchmarks/benchmark_results/crossdata/csv/results_combined_3seed.csv', index=False)
print(f'Saved {len(df_3seed)} rows to 3-seed CSV')
"

echo "Regenerating Wilcoxon 3-seed figures..."
/home/zeyufu/Desktop/.conda/bin/python3 /home/zeyufu/Desktop/PanODE-LAB/scripts/generate_statistical_figures_2seed.py --csv /home/zeyufu/Desktop/PanODE-LAB/benchmarks/benchmark_results/crossdata/csv/results_combined_3seed.csv

echo "All done."

echo "Updating IEEE blueprints TABLE IV..."
/home/zeyufu/Desktop/.conda/bin/python3 -c "
import pandas as pd
df = pd.read_csv('/home/zeyufu/Desktop/PanODE-LAB/benchmarks/benchmark_results/crossdata/csv/results_combined_3seed.csv')
# Compute mean and std for each model
stats = df.groupby('Model')['Score'].agg(['mean', 'std']).reset_index()
stats = stats.sort_values('mean', ascending=False)
print('3-seed stats:')
print(stats)

# Update blueprints
for bp in ['/home/zeyufu/Desktop/PanODE-LAB/docs/PAPER_BLUEPRINT_DPMM_IEEE.md', '/home/zeyufu/Desktop/PanODE-LAB/docs/PAPER_BLUEPRINT_TOPIC_IEEE.md']:
    with open(bp, 'r') as f:
        content = f.read()
    
    # Replace 0.4565±0.0026 with the new Topic-Transformer score
    tt_stat = stats[stats['Model'] == 'Topic-Transformer'].iloc[0]
    new_score = f\"{tt_stat['mean']:.4f}±{tt_stat['std']:.4f}\"
    content = content.replace('0.4565±0.0026', new_score)
    
    # Add footnote
    if 'Mean ± SD over 3 independent seeds (42, 0, 1)' not in content:
        content = content.replace('TABLE IV', 'TABLE IV\\n*Mean ± SD over 3 independent seeds (42, 0, 1).*')
        
    with open(bp, 'w') as f:
        f.write(content)
    print(f'Updated {bp}')
"
echo "All done."
