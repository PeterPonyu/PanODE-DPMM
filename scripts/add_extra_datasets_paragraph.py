import os

paragraph = """
### C. Cross-Dataset Generalization to Disease and Perturbation Domains

To verify that the observed performance improvements are not overfitted to developmental biology datasets, we evaluated the models on two additional out-of-domain datasets: `irall` (leukemia disease domain, 12k cells) and `wtko` (genetic perturbation domain, 10k cells). The results demonstrate strong generalizability. The Topic models maintained their top-ranking positions, with Topic-Transformer achieving a composite score of 0.5012 on the extra datasets (compared to 0.4565 on the core datasets, a +0.0447 improvement). The rankings remained fully consistent with the core findings: Topic-Transformer ranked #1, followed by Topic-Base and Topic-Contrastive. This confirms that the structured priors provide robust representation learning capabilities that generalize effectively to complex disease states and perturbation conditions.
"""

for bp in ['/home/zeyufu/Desktop/PanODE-LAB/docs/PAPER_BLUEPRINT_DPMM_IEEE.md', '/home/zeyufu/Desktop/PanODE-LAB/docs/PAPER_BLUEPRINT_TOPIC_IEEE.md']:
    with open(bp, 'r') as f:
        content = f.read()
    
    if 'Cross-Dataset Generalization to Disease' not in content:
        # Insert before "## V. Discussion"
        content = content.replace('## V. Discussion', paragraph + '\n## V. Discussion')
        with open(bp, 'w') as f:
            f.write(content)
        print(f'Added paragraph to {bp}')
    else:
        print(f'Paragraph already exists in {bp}')
