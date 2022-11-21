# Import everything needed to edit video clips
from moviepy.editor import *

# Open video files.
clip1 = VideoFileClip("Videos/x0_y0_Tile.avi")
clip2 = VideoFileClip("Videos/x1_y0_Tile.avi")
clip3 = VideoFileClip("Videos/x2_y0_Tile.avi")
clip4 = VideoFileClip("Videos/x0_y1_Tile.avi")
clip5 = VideoFileClip("Videos/x1_y1_Tile.avi")
clip6 = VideoFileClip("Videos/x2_y1_Tile.avi")
clip7 = VideoFileClip("Videos/x0_y2_Tile.avi")
clip8 = VideoFileClip("Videos/x1_y2_Tile.avi")
clip9 = VideoFileClip("Videos/x2_y2_Tile.avi")

# list of clips in their arrangement
clips = [[clip7, clip8, clip9],
         [clip4, clip5, clip6],
         [clip1, clip2, clip3]]

# stacking clips
final = clips_array(clips)
final.write_videofile("Output.mp4",fps=25)
# showing final clip
final.ipython_display(width=480)# Import everything needed to edit video clips
