import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


means_o     = [ 0.3967, 0.4113, 0.7461, 0.7908, 0.7647, 0.6498, 0.6939, 0.5688, 0.8220, 0.8638, 0.5957, 0.5780, 0.8292, 0.6869, 0.9241 ]
stddevs     = [ 0.2015, 0.2758, 0.2320, 0.1476, 0.1735, 0.2715, 0.2001, 0.2382, 0.1106, 0.1519, 0.2785, 0.3006, 0.2317, 0.2344, 0.1561 ]
predictions = [ 0.4691, 0.4890, 0.8708, 0.7450, 0.7656, 0.6820, 0.7519, 0.6025, 0.7352, 0.8248, 0.7528, 0.6426, 0.8452, 0.8235, 0.8787]


x = [x for x in range(len(means_o))]

means, predictions = zip(*sorted(zip(means_o, predictions)))
means, x_sort,     = zip(*sorted(zip(means_o, x)))
means, stddevs     = zip(*sorted(zip(means_o, stddevs)))

errors = [p-m for m,p in zip(means,predictions)]
errorsP = [ [0,          max(0, e)] for e in errors ]
errorsN = [ [-min(0, e),         0] for e in errors ]

x, x_sort, means, predictions = np.array(x), np.array(x_sort), np.array(means), np.array(predictions)
errors, errorsP, errorsN = np.array(errors), np.array(errorsP), np.array(errorsN)

from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Custom handler that draws crossing lines
class HandlerCross(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx = xdescent + width / 2
        cy = ydescent + height / 2
        # size = min(width, height) / 2.5
        DX=width/9
        DY=height/3

        style = orig_handle.get('style', '+')

        if style == "x":
            line1 = Line2D([cx-DX, cx+DX],     [cy, cy], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)
            line2 = Line2D([cx+DX, cx+DX*1.8], [cy, cy+DY], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)
            line3 = Line2D([cx+DX, cx+DX*1.8], [cy, cy-DY], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)
            line4 = Line2D([cx-DX, cx-DX*1.8], [cy, cy+DY], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)
            line5 = Line2D([cx-DX, cx-DX*1.8], [cy, cy-DY], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)

            return [line1, line2, line3, line4, line5]
        else:
            ddx = DX*0.1
            linewidth = orig_handle.get('linewidth', 2)
            line1 = Line2D([cx-ddx, cx+ddx], [cy, cy], color=orig_handle.get('color', 'darkorange'), linewidth=linewidth, transform=trans)
            return [line1]


mse = np.mean((means - predictions) ** 2)
mae = np.mean(np.abs(means - predictions))
print(f"MSE={mse}  MAE={mae}")



fig, ax = plt.subplots(figsize=(5, 3.5))
plt.ylim(0, 1.1)
plt.xlabel("control trajectories (sorted)")
plt.ylabel("score")


plt.axhline(y=1, color='gray', linestyle='-', linewidth=1, zorder=1)
plt.errorbar(x, means, yerr=stddevs,   fmt='s', color='black', ecolor='black', elinewidth=1, capsize=3, ms=4.5,  zorder=2)
# plt.scatter(x, predictions, facecolors=None,  zorder=3)

LW=1.1
DX=0.1
DY=0.01

for xi, yi in zip(x, predictions):
    plt.plot([xi-DX, xi+DX], [yi, yi], color='darkorange', linewidth=LW, zorder=3)
    plt.plot([xi+DX, xi+DX*1.8], [yi, yi+DY], color='darkorange', linewidth=LW, zorder=3)
    plt.plot([xi+DX, xi+DX*1.8], [yi, yi-DY], color='darkorange', linewidth=LW, zorder=3)
    plt.plot([xi-DX, xi-DX*1.8], [yi, yi+DY], color='darkorange', linewidth=LW, zorder=3)
    plt.plot([xi-DX, xi-DX*1.8], [yi, yi-DY], color='darkorange', linewidth=LW, zorder=3)


cross_handle = {'color': 'darkorange', 'linewidth': LW, "style":"x"}
box_handle = {'color': 'black', 'linewidth': LW*6, "style":"b"}

plt.legend(handles=[cross_handle, box_handle], labels=['estimations', 'mean scores'],
           handler_map={dict: HandlerCross()},
           loc='lower right')


ax.set_xticks(x)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels(x_sort)
plt.tight_layout()

plt.savefig("control_results.pdf", bbox_inches='tight', pad_inches=0)
plt.show()