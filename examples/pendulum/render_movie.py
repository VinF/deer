""" The script is slightly adapted from:

https://github.com/toddsifleet/inverted_pendulum/blob/master/render_movie.py

Author: Aaron Zixiao Qiu
"""

import os
import numpy as np
from math import sin, cos, pi

import matplotlib.pyplot as plt

PI = np.pi

def save_mp4(data, n):
    if (not os.path.exists('./img')):
        os.makedirs('./img')

    if (not os.path.exists('./video')):
        os.makedirs('./video')

    # Create temporal layout at the bottom
    fig = plt.figure(0)
    fig.suptitle("Pendulum on Cart")

    cart_time_line = plt.subplot2grid(
        (12, 12),
        (9, 0),
        colspan=12,
        rowspan=3
    )

    # Draw displacement curve
    t_max = max(data[:,0])
    cart_time_line.axis([
        0,
        t_max,
        min(data[:,1])*1.1,
        max(data[:,1])*1.1+.1,
    ])
    cart_time_line.set_xlabel('time (s)')
    cart_time_line.set_ylabel('x (m)')
    cart_time_line.plot(data[:,0], data[:,1],'r-')

    # Draw theta curve
    pendulum_time_line = cart_time_line.twinx()
    pendulum_time_line.axis([
        0,
        t_max,
        min(data[:,3])*1.1-.1,
        max(data[:,3])*1.1
    ])
    pendulum_time_line.set_ylabel('theta (rad)')
    pendulum_time_line.plot(data[:,0], data[:,3],'g-')

    # Cart layout
    cart_plot = plt.subplot2grid(
        (12,12),
        (0,0),
        rowspan=8,
        colspan=12
    )
    cart_plot.axes.get_yaxis().set_visible(False)

    # Draw cart and pole
    t = 0
    fps = 25.
    frame_number = 1
    x_min = min([min(data[:,1]), -1.1])
    x_max = max([max(data[:,1]), 1.1])

    time_bar, = cart_time_line.plot([0,0], [10000, -10000], lw=3)
    for point in data:
        if point[0] >= t + 1./fps or not t:
            _draw_point(point, time_bar, t, x_min, x_max, cart_plot)
            t = point[0]
            fig.savefig('img/_tmp%03d.png' % frame_number)
            frame_number += 1

    print(os.system("ffmpeg -framerate 25 -i img/_tmp%03d.png "  \
          + "-c:v libx264 -pix_fmt yuv420p video/out" + str(n) + ".mp4"))

    return

def _draw_point(point, time_bar, t, x_min, x_max, cart_plot):
    # Draw cart
    time_bar.set_xdata([t, t])
    cart_plot.cla()
    cart_plot.axis([x_min,x_max,-.5,.5])
    l_cart = 0.05 * (x_max + abs(x_min))
    cart_plot.plot([point[1]-l_cart,point[1]+l_cart], [0,0], 'r-', lw=5)

    # Draw pole
    theta = point[3] 
    x = sin(theta)
    y = cos(theta)
    l_pole = 0.2 * (x_max + abs(x_min))
    cart_plot.plot([point[1],point[1]+l_pole*x],[0,.4*y],'g-', lw=4)

    return 
