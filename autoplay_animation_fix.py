# ============================================================================
# SOLUTION: Make HTML5 Videos Behave Like GIFs (Autoplay + Loop)
# ============================================================================

"""
Instead of using ani.to_html5_video() which creates a video with controls,
we can customize the HTML to make it autoplay and loop like a GIF.
"""

from IPython.display import HTML
import base64

# ============================================================================
# OPTION 1: Custom HTML5 Video with GIF-like behavior (Recommended)
# ============================================================================

def animation_to_html5_gif(ani):
    """
    Convert matplotlib animation to HTML5 video that behaves like a GIF:
    - Autoplays
    - Loops continuously
    - No controls
    - Muted (required for autoplay in most browsers)
    """
    from io import BytesIO
    import base64
    
    # Save animation to video format in memory
    f = BytesIO()
    ani.save(f, writer='ffmpeg', format='mp4', codec='libx264', 
             extra_args=['-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2'])
    f.seek(0)
    
    # Encode as base64
    video_base64 = base64.b64encode(f.read()).decode('ascii')
    
    # Create HTML with autoplay, loop, and no controls (like a GIF)
    html = f'''
    <video autoplay loop muted playsinline style="max-width: 100%;">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    '''
    
    return HTML(html)

# ============================================================================
# OPTION 2: Using matplotlib's built-in with custom parameters
# ============================================================================

def animation_to_autoplay_video(ani):
    """
    Similar to above but uses matplotlib's to_html5_video with custom embed
    """
    from matplotlib.animation import HTMLWriter
    from io import StringIO
    import base64
    
    # Get the video data
    video = ani.to_html5_video()
    
    # Modify the HTML to add autoplay, loop, muted attributes
    # The default output is a <video> tag, we just need to add attributes
    modified = video.replace('<video ', '<video autoplay loop muted playsinline ')
    # Remove controls if present
    modified = modified.replace(' controls', '')
    
    return HTML(modified)

# ============================================================================
# OPTION 3: Simplified wrapper function
# ============================================================================

def display_animation_as_gif(ani):
    """
    Display matplotlib animation as auto-playing looping video (GIF-like)
    
    Usage:
        ani = animation.FuncAnimation(...)
        display_animation_as_gif(ani)
    """
    from IPython.display import HTML, display
    
    # Get HTML5 video
    html = ani.to_html5_video()
    
    # Add autoplay, loop, and muted attributes
    html = html.replace('<video ', '<video autoplay loop muted playsinline ')
    html = html.replace(' controls', '')  # Remove controls bar
    
    display(HTML(html))

# ============================================================================
# COMPLETE UPDATED CODE FOR YOUR NOTEBOOK
# ============================================================================

# Cell 19: Heatmap Animation (Updated)
import matplotlib.animation as animation
from IPython.display import HTML, display

fig, ax = plt.subplots()
heatmap = sns.heatmap(flipped_data_3d[0], ax=ax, cbar=False)

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

# Make it behave like a GIF: autoplay, loop, no controls
html = ani.to_html5_video()
html = html.replace('<video ', '<video autoplay loop muted playsinline ')
html = html.replace(' controls', '')
display(HTML(html))
plt.close(fig)

# ============================================================================
# Cell 29: Updated Functions
# ============================================================================

def animate_frames(frames, sequence_id):
    n_samples = frames.shape[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    vmin, vmax = 0, 260

    im = ax.imshow(frames[0], cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, ax=ax)

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        ax.set_title(f"Sequence {sequence_id}\nSample {frame_idx}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=n_samples, interval=100, blit=False)
    plt.close(fig)
    return ani

def frames_to_animation(frames, animation_name, sequence_id):
    """Creates animation and embeds it as auto-playing video (GIF-like)"""
    from IPython.display import HTML, display
    
    ani = animate_frames(frames, sequence_id)
    
    # Convert to HTML5 video that behaves like a GIF
    html = ani.to_html5_video()
    html = html.replace('<video ', '<video autoplay loop muted playsinline ')
    html = html.replace(' controls', '')
    
    display(HTML(html))

def animate_overlapping(df, sequence_id):
    overlapping_frames = df_to_overlapping_frames(df, sequence_id)
    frames_to_animation(overlapping_frames, f"{sequence_id}_overlapping", sequence_id)

def animate_separated(df, sequence_id):
    separated_frames = df_to_separated_frames(df, sequence_id)
    frames_to_animation(separated_frames, f"{sequence_id}_separated", sequence_id)

# ============================================================================
# HTML Attributes Explained
# ============================================================================

"""
autoplay      - Video starts playing automatically (like a GIF)
loop          - Video loops forever (like a GIF)
muted         - Video is muted (required for autoplay in modern browsers)
playsinline   - Plays inline on mobile devices (iOS requirement)
controls      - Removed to hide play/pause buttons (optional)

These attributes make the video behave exactly like a GIF!
"""
