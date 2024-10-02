import pickle
from pathlib import Path
import pandas as pd
from plotnine import ggplot, aes, geom_violin, labs, theme, element_text, geom_pointrange, position_dodge,element_blank
from plotnine.themes import theme_light
from scipy import stats
import numpy as np

import matplotlib.font_manager as fm

# Find all system fonts
system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')

font_path = "/Library/Fonts/Calibri.ttf"
prop = fm.FontProperties(fname=Path(Path.home(),font_path))

def conf_int_95(data):
    mean = np.nanmean(data)
    inf = np.nanpercentile(data,2.5)
    sup = np.nanpercentile(data,97.5) 
    return mean, inf, sup

tasks = ['animales','fas']
y_label = 'mean_absolute_error_MMSE_Total_Score'

results_dir = Path(Path.home(),'results','GERO_Ivo') if 'Users/gp' in str(Path.home()) else Path('D:','results','GERO_Ivo')

metrics_df = pickle.load(open(Path(results_dir, 'all_metrics.pkl'), 'rb'))

metrics_df['dimension'] = metrics_df['dimension'].replace({'properties':'DSMs (Word properties)','valid_responses':"Manual assessment (valid responses)"})
metrics_df['task'] = metrics_df['task'].replace({'animales':'Semantic fluency','fas':'Phonemic fluency'})

# Group by task and dimension, then calculate mean and 95% confidence interval
summary = metrics_df.groupby(['task','dimension']).agg(
    mean_score=(y_label, 'mean'),
    lower_ci=(y_label, lambda x: conf_int_95(x)[1]),
    upper_ci=(y_label, lambda x: conf_int_95(x)[2])
).reset_index()

print(summary)

p = (
    ggplot(metrics_df, aes(x='task', y=y_label, fill='dimension'))  # Aesthetic mappings
    + geom_violin(scale='width', trim=True, alpha=0.5, position=position_dodge(width=0.5))  # Violin plot with dodge
    + geom_pointrange(
        aes(x='task', y='mean_score', ymin='lower_ci', ymax='upper_ci', color='dimension'),
        data=summary,
        position=position_dodge(width=0.5),  # Dodge position to separate dimensions
        size=1  # Size of the point and line
    )  # Pointrange for 95% CI
    + labs(
        x='Task',
        y='Mean absolute error in predicting MMSE scores'
    )  # Title and axis labels
    + theme_light()  # ggplot2-like light theme
    + theme(
        axis_text_x=element_text(size=10, family=prop.get_family()),
        axis_text_y=element_text(size=10, family=prop.get_family()),  # Rotate x-axis labels
        figure_size=(10, 5),  # Set the figure size
        axis_title=element_text(size=12, family=prop.get_family()), 
        legend_title=element_blank(),  # Axis label style
        legend_text=element_text(size=12, family=prop.get_family()),  # Legend text style
        legend_position='bottom'  # Position the legend below the x-axis
    )   
)

# Display the plot
#print(p)

# Save the plot to a file
p.save(Path(results_dir,f'{y_label}_comparisons.png', dpi=300))

