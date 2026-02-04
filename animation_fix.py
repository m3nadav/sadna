# ============================================================================
# SOLUTION: Embed Animations Directly in Notebook/HTML
# ============================================================================

"""
Replace your current animation code with one of these approaches:
"""

# ============================================================================
# OPTION 1: HTML5 Video (Recommended - works in most browsers)
# ============================================================================

import matplotlib.animation as animation
from IPython.display import HTML

# Your existing animation setup...
fig, ax = plt.subplots()
# ... your plotting code ...

ani = animation.FuncAnimation(
    fig,
    update,
    frames=flipped_data_3d,
    interval=200
)

# Instead of saving and loading GIF, directly display as HTML5 video
HTML(ani.to_html5_video())

# ============================================================================
# OPTION 2: JavaScript HTML (Interactive, but larger file size)
# ============================================================================

ani = animation.FuncAnimation(
    fig,
    update,
    frames=flipped_data_3d,
    interval=200
)

# Display as interactive JavaScript
HTML(ani.to_jshtml())

# ============================================================================
# OPTION 3: Embed GIF as Base64 (If you need GIF format)
# ============================================================================

import io
import base64
from IPython.display import HTML

# Save to BytesIO instead of file
buf = io.BytesIO()
ani.save(buf, writer='pillow', format='gif')
buf.seek(0)

# Encode as base64 and display
gif_base64 = base64.b64encode(buf.read()).decode('ascii')
HTML(f'<img src="data:image/gif;base64,{gif_base64}" />')

# ============================================================================
# FOR YOUR frames_to_animation FUNCTION
# ============================================================================

def frames_to_animation(frames, animation_name, sequence_id):
    """Modified to return embeddable animation"""
    from IPython.display import HTML
    
    ani = animate_frames(frames, sequence_id)
    
    # Return HTML5 video (embeds in notebook and HTML export)
    return HTML(ani.to_html5_video())

# Usage:
# frames_to_animation(overlapping_frames, "SEQ_000007_overlapping.gif", "SEQ_000007")

# ============================================================================
# COMPLETE EXAMPLE FOR YOUR HEATMAP ANIMATION
# ============================================================================

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    sns.heatmap(frame, ax=ax, cbar=False)
    ax.set_title("Heatmap over time")

ani = animation.FuncAnimation(
    fig,
    update,
    frames=flipped_data_3d,
    interval=200,
    blit=False
)

# Display directly without saving to file
display(HTML(ani.to_html5_video()))
plt.close(fig)
