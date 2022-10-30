# Plot method as shown on PyShine Youtube channel https://pyshine.com/How-to-plot-real-time-frame-rate-in-opencv-and-matplotlib/
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Get the Figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_facecolor((1,1,1)) # Set the background to white
# 
def animate(i):
    ax.clear()
    ts = []
    xs = []
    ys = []
    graph_data = open('data/output.csv','r').read() # Open file.csv generated by pupil_track_ui.py
    lines = graph_data.split('\n')
    for line in lines[1:]:
        if len(line) > 1: # Skip the first labels line in csv file
            t, x, y = line.split(',')
            ts.append(float(t))
            xs.append(float(x))
            ys.append(float(y))

            # I want a running representation of eyemovement do just the last 128 samples (10s) need to be graphed
            ts = ts[-128:] 
            xs = xs[-128:]
            ys = ys[-128:]
	# Lets add these lists ts, xs, ys to the plot		
    ax.clear()
    ax.plot(ts, xs,'-', color = (0,0,1)) #blue for horizontal
    ax.plot(ts, ys,'-', color = (1,0,0.25)) #red for vertical
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement [x/blue, y/red]")
    ax.set_title("Pupil Movement")
    fig.tight_layout() # To remove outside borders
    ax.yaxis.grid(True)
# Lets call the animation function 	
ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()


