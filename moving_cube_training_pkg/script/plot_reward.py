"""
A simple example of an animated plot
"""
import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from theconstruct_msgs.msg import RLExperimentInfo

class RTPlotReward(object):
    def __init__(self):

        rospy.Subscriber("/openai/reward", RLExperimentInfo, self._reward_callback)

        fig, ax = plt.subplots()

        self.x = 0.0
        self.y = 0.0
        self.line, = ax.plot(self.x, self.y)

        ani = animation.FuncAnimation(fig, self.animate, np.arange(1, 200), init_func=self.init,
                                      interval=25, blit=True)
        plt.show()
        print "Plot Line Ready"

    def _reward_callback(self,msg):
        print str(msg)
        self.y = msg.episode_reward
        self.x += 1.0

    def animate(self,i):
        self.line.set_ydata(self.y)  # update the data
        return self.line,

    # Init only required for blitting to give a clean slate.
    def init(self):
        self.line.set_ydata(np.ma.array(self.x, mask=True))
        return self.line,


if __name__ == "__main__":
    rtplotline_obj = RTPlotReward()